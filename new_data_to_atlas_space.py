#!/usr/bin/env python3
import os
import os.path
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink, DataGrabber
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.minc import Resample, BigAverage, VolSymm
import argparse


def create_workflow(
    xfm_dir,
    xfm_pattern,
    atlas_dir,
    atlas_pattern,
    source_dir,
    source_pattern,
    work_dir,
    out_dir,
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
            sinc_interpolation=True
        ),
        name='resample_',
        iterfield=['input_file', 'transformation']
    )
    wf.connect(datasource_source, 'outfiles', resample, 'input_file')
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
        '--debug',
        dest='debug',
        action='store_true',
        help='debug mode'
    )

    args = parser.parse_args()

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
