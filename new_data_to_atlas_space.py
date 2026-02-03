#!/usr/bin/env python3
import os
import os.path
import shutil
import subprocess
from copy import deepcopy
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces import utility as utils
from nipype.interfaces.io import SelectFiles, DataSink, DataGrabber
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.minc import Resample, BigAverage, VolSymm, Volcentre, Norm, Volpad, Voliso
import argparse


def check_minc_available():
    """Check if MINC tools are available on PATH, try to load via 'ml minc' if not."""
    if shutil.which('mincresample') is not None:
        return True
    
    # Try to load minc module
    print("MINC tools not found on PATH, attempting to load via 'ml minc'...")
    try:
        result = subprocess.run(
            ['bash', '-c', 'source /etc/profile.d/modules.sh 2>/dev/null || true; ml minc && echo $PATH'],
            capture_output=True,
            text=True,
            check=True
        )
        # Update PATH with the module's additions
        new_path = result.stdout.strip()
        if new_path:
            os.environ['PATH'] = new_path
        
        if shutil.which('mincresample') is not None:
            print("Successfully loaded MINC module.")
            return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to load MINC module: {e}")
    except FileNotFoundError:
        print("Module system not available.")
    
    raise RuntimeError(
        "MINC tools are not available. Please ensure MINC is installed and on your PATH, "
        "or that the 'minc' module can be loaded via 'ml minc'."
    )


def identity_file(input_file):
    """Pass through file unchanged."""
    output_file = 'IdentityFile_copy' + os.path.splitext(input_file)[1]
    shutil.copyfile(input_file, output_file)
    return os.path.abspath(output_file)


def _convert_nii_to_mnc(input_file):
    """Convert a NIfTI file to MINC format using nii2mnc."""
    lower_file = input_file.lower()
    
    if lower_file.endswith('.nii.gz'):
        output_file = os.path.basename(input_file)[:-7] + '.mnc'
    elif lower_file.endswith('.nii'):
        output_file = os.path.basename(input_file)[:-4] + '.mnc'
    else:
        # Already MINC or unknown format, return as-is
        return input_file
    
    output_path = os.path.abspath(output_file)
    cmd = ['nii2mnc', '-float', '-clobber', input_file, output_path]
    print(f"+++ Converting NIfTI to MINC: {input_file} -> {output_path}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nii2mnc failed: {result.stderr}")
    
    return output_path


convert_nii_to_mnc = Function(
    input_names=['input_file'],
    output_names=['output_file'],
    function=_convert_nii_to_mnc
)


def _calc_threshold_blur_preprocess(input_file):
    """Calculate threshold blur based on step sizes."""
    def get_step_sizes(mincfile):
        import subprocess
        xcmd = f'mincinfo -attvalue xspace:step {mincfile}'
        ycmd = f'mincinfo -attvalue yspace:step {mincfile}'
        zcmd = f'mincinfo -attvalue zspace:step {mincfile}'
        
        xstep = float(subprocess.check_output(xcmd, shell=True).decode().split()[0])
        ystep = float(subprocess.check_output(ycmd, shell=True).decode().split()[0])
        zstep = float(subprocess.check_output(zcmd, shell=True).decode().split()[0])
        return (xstep, ystep, zstep)
    
    (step_x, step_y, step_z) = get_step_sizes(input_file)
    return abs(step_x + step_y + step_z)


calc_threshold_blur_preprocess = Function(
    input_names=['input_file'],
    output_names=['threshold_blur'],
    function=_calc_threshold_blur_preprocess
)


def create_workflow(
    xfm_dir,
    xfm_pattern,
    atlas_dir,
    atlas_pattern,
    source_dir,
    source_pattern,
    work_dir,
    out_dir,
    normalise=True,
    model_norm_thresh=0.1,
    pad=5,
    iso=True,
    name="new_data_to_atlas_space"
):

    wf = Workflow(name=name)
    wf.base_dir = os.path.join(work_dir)

    datasource_source = Node(
        interface=DataGrabber(
            infields=[],
            outfields=['outfiles'],
            sort_filelist=True
        ),
        name='datasource_source'
    )
    datasource_source.inputs.base_directory = os.path.abspath(source_dir)
    datasource_source.inputs.template = '*'
    datasource_source.inputs.field_template = {'outfiles': source_pattern}
    datasource_source.inputs.template_args = {'outfiles': [[]]}

    # ========== PREPROCESSING PIPELINE ==========
    # Convert NIfTI to MINC if needed
    nii_to_mnc_converter = MapNode(
        interface=deepcopy(convert_nii_to_mnc),
        name='nii_to_mnc_converter',
        iterfield=['input_file']
    )
    wf.connect(datasource_source, 'outfiles', nii_to_mnc_converter, 'input_file')

    # Volume centering
    preprocess_volcentre = MapNode(
        interface=Volcentre(zero_dircos=True),
        name='preprocess_volcentre',
        iterfield=['input_file']
    )
    wf.connect(nii_to_mnc_converter, 'output_file', preprocess_volcentre, 'input_file')

    # Normalization
    if normalise:
        preprocess_threshold_blur = MapNode(
            interface=deepcopy(calc_threshold_blur_preprocess),
            name='preprocess_threshold_blur',
            iterfield=['input_file']
        )
        wf.connect(preprocess_volcentre, 'output_file', preprocess_threshold_blur, 'input_file')

        preprocess_normalise = MapNode(
            interface=Norm(
                cutoff=model_norm_thresh,
                threshold=True,
                threshold_perc=model_norm_thresh
            ),
            name='preprocess_normalise',
            iterfield=['input_file', 'threshold_blur']
        )
        wf.connect(preprocess_volcentre, 'output_file', preprocess_normalise, 'input_file')
        wf.connect(preprocess_threshold_blur, 'threshold_blur', preprocess_normalise, 'threshold_blur')
    else:
        preprocess_normalise_id = utils.Function(
            input_names=['input_file'],
            output_names=['output_file'],
            function=identity_file,
        )
        preprocess_normalise = MapNode(
            interface=preprocess_normalise_id,
            name='preprocess_normalise',
            iterfield=['input_file']
        )
        wf.connect(preprocess_volcentre, 'output_file', preprocess_normalise, 'input_file')

    # Padding
    if pad > 0:
        preprocess_volpad = MapNode(
            interface=Volpad(
                distance=pad,
                smooth=True,
                smooth_distance=5
            ),
            name='preprocess_volpad',
            iterfield=['input_file']
        )
    else:
        preprocess_volpad_id = utils.Function(
            input_names=['input_file'],
            output_names=['output_file'],
            function=identity_file,
        )
        preprocess_volpad = MapNode(
            interface=preprocess_volpad_id,
            name='preprocess_volpad',
            iterfield=['input_file']
        )
    wf.connect(preprocess_normalise, 'output_file', preprocess_volpad, 'input_file')

    # Isotropic resampling
    if iso:
        preprocess_voliso = MapNode(
            interface=Voliso(avgstep=True),
            name='preprocess_voliso',
            iterfield=['input_file']
        )
    else:
        preprocess_voliso_id = utils.Function(
            input_names=['input_file'],
            output_names=['output_file'],
            function=identity_file,
        )
        preprocess_voliso = MapNode(
            interface=preprocess_voliso_id,
            name='preprocess_voliso',
            iterfield=['input_file']
        )
    wf.connect(preprocess_volpad, 'output_file', preprocess_voliso, 'input_file')
    # ========== END PREPROCESSING ==========

    datasource_xfm = Node(
        interface=DataGrabber(
            infields=[],
            outfields=['outfiles'],
            sort_filelist=True
        ),
        name='datasource_xfm'
    )
    datasource_xfm.inputs.base_directory = os.path.abspath(xfm_dir)
    datasource_xfm.inputs.template = '*'
    datasource_xfm.inputs.field_template = {'outfiles': xfm_pattern}
    datasource_xfm.inputs.template_args = {'outfiles': [[]]}

    datasource_atlas = Node(
        interface=DataGrabber(
            infields=[],
            outfields=['outfiles'],
            sort_filelist=True
        ),
        name='datasource_atlas'
    )
    datasource_atlas.inputs.base_directory = os.path.abspath(atlas_dir)
    datasource_atlas.inputs.template = '*'
    datasource_atlas.inputs.field_template = {'outfiles': atlas_pattern}
    datasource_atlas.inputs.template_args = {'outfiles': [[]]}

    resample = MapNode(
        interface=Resample(
            sinc_interpolation=True,
            invert_transformation=True  # Transform is native->atlas, but resample needs atlas->native
        ),
        name='resample_',
        iterfield=['input_file', 'transformation']
    )
    wf.connect(preprocess_voliso, 'output_file', resample, 'input_file')
    wf.connect(datasource_xfm, 'outfiles', resample, 'transformation')
    wf.connect(datasource_atlas, 'outfiles', resample, 'like')

    bigaverage = Node(
        interface=BigAverage(
            output_float=True,
            robust=False
        ),
        name='bigaverage',
        iterfield=['input_file']
    )

    wf.connect(resample, 'output_file', bigaverage, 'input_files')

    datasink = Node(
        interface=DataSink(
            base_directory=out_dir,
            container=out_dir
        ),
        name='datasink'
    )

    wf.connect([(bigaverage, datasink, [('output_file', 'average')])])
    wf.connect([(resample, datasink, [('output_file', 'atlas_space')])])
    wf.connect([(datasource_xfm, datasink, [('outfiles', 'transforms')])])

    return wf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
        required=True
    )

    parser.add_argument(
        "--xfm_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--xfm_pattern",
        type=str,
        required=True
    )

    parser.add_argument(
        "--source_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--source_pattern",
        type=str,
        required=True
    )

    parser.add_argument(
        "--atlas_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--atlas_pattern",
        type=str,
        required=True
    )

    parser.add_argument(
        "--work_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        '--normalise',
        type=int,
        default=1,
        choices=[0, 1],
        help='Normalise input data via histogram clamping (default: 1)'
    )

    parser.add_argument(
        '--model_norm_thresh',
        type=float,
        default=0.1,
        help='Thresholding of normalized image to remove background noise (default: 0.1)'
    )

    parser.add_argument(
        '--pad',
        type=int,
        default=5,
        help='Zero padding around image (default: 5)'
    )

    parser.add_argument(
        '--iso',
        type=int,
        default=1,
        choices=[0, 1],
        help='Resample image to be isotropic (default: 1)'
    )

    parser.add_argument(
        '--debug',
        dest='debug',
        action='store_true',
        help='debug mode'
    )

    args = parser.parse_args()

    # Check MINC availability before running workflow
    check_minc_available()

    if args.debug:
        from nipype import config
        config.enable_debug_mode()
        config.set('execution', 'stop_on_first_crash', 'true')
        config.set('execution', 'remove_unnecessary_outputs', 'false')
        config.set('execution', 'keep_inputs', 'true')
        config.set('logging', 'workflow_level', 'DEBUG')
        config.set('logging', 'interface_level', 'DEBUG')
        config.set('logging', 'utils_level', 'DEBUG')

    wf = create_workflow(
        xfm_dir=os.path.abspath(args.xfm_dir),
        xfm_pattern=args.xfm_pattern,
        atlas_dir=os.path.abspath(args.atlas_dir),
        atlas_pattern=args.atlas_pattern,
        source_dir=os.path.abspath(args.source_dir),
        source_pattern=args.source_pattern,
        work_dir=os.path.abspath(args.work_dir),
        out_dir=os.path.abspath(args.out_dir),
        normalise=bool(args.normalise),
        model_norm_thresh=args.model_norm_thresh,
        pad=args.pad,
        iso=bool(args.iso),
        name=args.name
    )

    wf.run(
        plugin='MultiProc',
        plugin_args={
            'n_procs': int(
                os.environ["NCPUS"] if "NCPUS" in os.environ else os.cpu_count()
            )
        }
    )
