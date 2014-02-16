from nipype import config
config.enable_debug_mode()

import glob
import os
import os.path
import subprocess
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nipype.interfaces.utility as utils

from nipypeminc import Volcentre, Norm, Volpad, Voliso, Math, Pik, Blur, Gennlxfm, XfmConcat, BestLinReg, NlpFit, XfmAvg, XfmInvert, Resample, BigAverage

def run_command(cmd):
    """
    Run a shell command and return all stdout, throwing an error
    if anything appears on stderr. Only used for commands that are
    expected to be short running, e.g. mincinfo.
    """

    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            shell=True)
    stdout, stderr = proc.communicate()

    if stderr == '':
        return stdout
    else:
        assert False, 'Stuff on stderr: ' + str(stderr) # FIXME Change to an appropriate exception?

def get_step_sizes(mincfile):
    """
    Get the x, y, and z step sizes from a Minc file.
    """

    xcmd = 'mincinfo -attvalue xspace:step ' + mincfile
    ycmd = 'mincinfo -attvalue yspace:step ' + mincfile
    zcmd = 'mincinfo -attvalue zspace:step ' + mincfile

    xstep = float(run_command(xcmd).split()[0])
    ystep = float(run_command(ycmd).split()[0])
    zstep = float(run_command(zcmd).split()[0])

    return (xstep, ystep, zstep)

# Hard coded things, should be parameters:
FAST_EXAMPLE_BASE_DIR = '/scratch/fast-example'
NORM_CUTOFF = 0.1
NORM_THRESHOLD_PERC = 0.1
NORM_THRESHOLD_BLUR = 7.5
NORMALISE = True
PAD_DISTANCE = 5
PAD_SMOOTH_DISTANCE = 1

ISOTROPIC_RESAMPLING = True
CHECK_FILE = True

FIT_STAGES = ['lin', 1, 3]

MODEL_MIN_STEP = 0.5
MODEL_NORM_THRESH = 0.1

LINMETHOD = 'bestlinreg'

# fit.10-genmodel.conf:
#
# ICBM nlin conf
CONF = [ {'step': 32, 'blur_fwhm': 16, 'iterations': 20},
         {'step': 16, 'blur_fwhm': 8,  'iterations': 20},
         {'step': 12, 'blur_fwhm': 6,  'iterations': 20},
         {'step': 8,  'blur_fwhm': 4,  'iterations': 20},
         {'step': 6,  'blur_fwhm': 3,  'iterations': 20},
         {'step': 4,  'blur_fwhm': 2,  'iterations': 10},
         {'step': 2,  'blur_fwhm': 1,  'iterations': 10}, ]

# Top level workflow.
workflow = pe.Workflow(name="workflow")
workflow.base_dir = os.path.abspath(FAST_EXAMPLE_BASE_DIR)

# Glob all the sml*pre.mnc files in the current directory.
datasource = pe.Node(interface=nio.DataGrabber(sort_filelist=True), name='datasource')
datasource.inputs.base_directory = os.path.abspath(FAST_EXAMPLE_BASE_DIR)
datasource.inputs.template = 'sml*pre.mnc'

# Glob for the first sml*pre.mnc file in the current directory.
# FIXME Use Select from utils?
sorted_files = sorted(glob.glob(os.path.join(os.path.abspath(FAST_EXAMPLE_BASE_DIR), 'sml*pre.mnc')))
first_file_name     = sorted_files[0]
first_file_template = os.path.split(sorted_files[0])[1]
datasource_first_file = pe.Node(interface=nio.DataGrabber(sort_filelist=True), name='datasource_first_file')
datasource_first_file.inputs.base_directory = os.path.abspath(FAST_EXAMPLE_BASE_DIR)
datasource_first_file.inputs.template = first_file_template

# Sink for all the results (the subdirectory volgenmodel_output).
datasink = pe.Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = os.path.abspath(os.path.join(FAST_EXAMPLE_BASE_DIR, 'volgenmodel_output'))

# Preprocessing: centre volumes.
volcentre = pe.MapNode(interface=Volcentre(zero_dircos=True, verbose=True),
                       name='volcentre',
                       iterfield=['input_file'])

# Preprocessing: norm step.
if NORMALISE:
    # FIXME this is not complete, need to also get the step size for each file!!!!!
    """
         # get step sizes
         chomp($step_x = `mincinfo -attvalue xspace:step $infiles[$f]`);
         chomp($step_y = `mincinfo -attvalue yspace:step $infiles[$f]`);
         chomp($step_z = `mincinfo -attvalue zspace:step $infiles[$f]`);

         printf STDOUT "   | $nrmfile - normalising\n";
         &do_cmd_batch("NRM$$-$f", "CEN$$-$f",
                       'mincnorm', '-clobber',
                       '-cutoff', $opt{'model_norm_thresh'},
                       '-threshold',
                       '-threshold_perc', $opt{'model_norm_thresh'},
                       '-threshold_blur', abs($step_x + $step_y + $step_z),
                       $resfiles[$f], $nrmfile);
    """

    norm = pe.MapNode(interface=Norm(cutoff=NORM_CUTOFF,
                                     threshold_perc=NORM_THRESHOLD_PERC,
                                     threshold_blur=NORM_THRESHOLD_BLUR),
                      name='norm',
                      iterfield=['input_file'])
else:
    norm = pe.MapNode(interface=utils.IdentityInterface(fields=['input_file']),
                      name='identity_norm',
                      iterfield=['input_file'])

# Preprocessing: padding.
if PAD_DISTANCE > 0:
    pad = pe.MapNode(interface=Volpad(distance=PAD_DISTANCE, smooth=True, smooth_distance=PAD_SMOOTH_DISTANCE),
                     name='volpad',
                     iterfield=['input_file'])
else:
    pad = pe.MapNode(interface=utils.IdentityInterface(fields=['input_file']),
                      name='identity_pad',
                      iterfield=['input_file'])

# Preprocessing: isotropic resampling.
if ISOTROPIC_RESAMPLING:
    iso = pe.MapNode(interface=Voliso(avgstep=True),
                     name='voliso',
                     iterfield=['input_file'])
else:
    iso = pe.MapNode(interface=utils.IdentityInterface(fields=['input_file']),
                      name='identity_voliso',
                      iterfield=['input_file'])

# Preprocessing: check file.
if CHECK_FILE:
    pik = pe.MapNode(interface=Pik(triplanar=True, sagittal_offset=10),
                     name='pik',
                     iterfield=['input_file'])
else:
    pik = pe.MapNode(interface=utils.IdentityInterface(fields=['input_file']),
                      name='identity_pik',
                      iterfield=['input_file'])

# Preprocessing: initial model from the "first" file.
(xstep, ystep, zstep) = get_step_sizes(first_file_name)
initial_model = pe.Node(interface=Blur(fwhm3d=(abs(xstep*4), abs(ystep*4), abs(zstep*4))),
                        name='initial_model')


# Identity transformation.
# FIXME link in input field:    '-like', $cmodel,
identity_gennlxfm = pe.Node(interface=Gennlxfm(step=CONF[0]['step']),
                            name='identity_gennlxfm')

# Determine the last linear stage.
lastlin = FIT_STAGES[::-1].index('lin') - len(FIT_STAGES) + 1

# Connect up the workflow for the nodes that we have
# defined so far. More connections come later when
# we loop through the stages.

# sml*pre.mnc => volcentre
workflow.connect(datasource,            'outfiles', volcentre,     'input_file')

# first file => initial model
workflow.connect(datasource_first_file, 'outfiles', initial_model, 'input_file')

# The initial model is used to construct the identity transformation.
workflow.connect(initial_model, 'output_file', identity_gennlxfm, 'like')

# volcentre => norm => pad => iso => pik
workflow.connect(volcentre,     'output_file',  norm,       'input_file')
workflow.connect(norm,          'output_file',  pad,        'input_file')
workflow.connect(pad,           'output_file',  iso,        'input_file')
workflow.connect(iso,           'output_file',  pik,        'input_file')

# Keep the picture.
workflow.connect(pik,           'output_file',  datasink,   'final_of_pik')

# Keep the initial model.
workflow.connect(initial_model,     'output_file',  datasink,   'final_initial_model')

# Keep the identity transformation.
workflow.connect(identity_gennlxfm, 'output_file',  datasink,   'final_identity_gennlxfm')

# Construct the workflow for each stage.

# FIXME we need to store these - not sure what the final output actually is.
stage_models = [None] * len(FIT_STAGES)

# At some point we need the last linear stage's xfm outputs.
last_linear_stage_xfm = None

datasource_fit_confs = [None] * len(FIT_STAGES)

# Generate the config files for each stage.
for snum in range(0, len(FIT_STAGES)):
    end_stage = FIT_STAGES[snum]

    print 'Generating config file for stage %02d' % snum

    if end_stage == 'lin':
       print "---Linear fit---"
    else:
        print "---Non Linear fit---"

        conf_fname = os.path.join(FAST_EXAMPLE_BASE_DIR, 'fit-stage-%02d.conf' % snum)  #   "$cworkdir/fit.conf";
        print 'Creating', conf_fname

        with open(conf_fname, 'w') as f:
            f.write( ("# {conf_fname} -- created by {me}\n#\n"
                      "# End stage: {end_stage}\n"
                      "# Stage Num: {snum_txt}\n\n").format(conf_fname=conf_fname,
                                                            me='FIXME',
                                                            end_stage=end_stage,
                                                            snum_txt='%02d' % snum))

            f.write("@conf = (\n")
            for s in range(0, end_stage + 1):
               f.write("   {'step' => " + str(CONF[s]['step']))
               f.write(", 'blur_fwhm' => " + str(CONF[s]['blur_fwhm']))
               f.write(", 'iterations' => " + str(CONF[s]['iterations']))
               f.write("},\n")
            f.write("   );\n")


        datasource_fit_confs[snum] = pe.Node(interface=nio.DataGrabber(sort_filelist=True), name='datasource_fit_confs_stage_%02d' % snum)
        datasource_fit_confs[snum].inputs.base_directory = os.path.abspath(FAST_EXAMPLE_BASE_DIR)
        datasource_fit_confs[snum].inputs.template = os.path.split(conf_fname)[1]

for snum in range(0, len(FIT_STAGES)):
    # creating fit-model-iso...
    end_stage = FIT_STAGES[snum]

    print 'Stage: %02d, end stage: %s' % (snum, str(end_stage),)
    modelmaxstep = CONF[0 if end_stage == 'lin' else end_stage]['step']/4

    # Check that the resulting model won't be too large.
    if modelmaxstep < MODEL_MIN_STEP:
        modelmaxstep = MODEL_MIN_STEP

    print 'modelmaxstep: %d' % (modelmaxstep,)

    # prev_model is the previous stage's model, or the initial model if
    # we are on snum == 0. See also 'cmodel' in volgenmodel.
    if snum == 0:
        prev_model = initial_model
    else:
        prev_model = stage_models[snum - 1]

    # Normalisation.
    norm_tmp = pe.Node(interface=Norm(
                                    cutoff=MODEL_NORM_THRESH,
                                    threshold=True,
                                    threshold_perc=MODEL_NORM_THRESH,
                                    threshold_blur=3),
                                    # ==> threshold_mask="$isomodel_base.msk.mnc"),
                                    # ==> "$isomodel_base.nrm.mnc");
                       name='norm_stage_%02d' % snum)
    workflow.connect(prev_model, 'output_file', norm_tmp, 'input_file')

    # ISO
    iso_tmp = pe.Node(interface=Voliso(maxstep=modelmaxstep),
                      name='iso_stage_%02d' % snum) # ==> "$isomodel_base.mnc"

    workflow.connect(norm_tmp, 'output_file', iso_tmp, 'input_file')

    # ISO model pik
    pik_tmp = pe.Node(interface=Pik(
                                    triplanar=True,
                                    horizontal_triplanar_view=True,
                                    scale=4,
                                    tile_size=400,
                                    sagittal_offset=10), # ==> "$isomodel_base.jpg"
                      name='iso_pik_stage_%02d' % snum)

    workflow.connect(iso_tmp, 'output_file', pik_tmp,  'input_file')

    workflow.connect(pik_tmp, 'output_file', datasink, 'final_of_pik_tmp_stage_%02d' % snum)


    # Isomodel fit mask.
    #    #chomp($step_x = `mincinfo -attvalue xspace:step $isomodel_base.msk.mnc`);
    #    $step_x = 1;
    step_x = 1
    iso_fit_mask_blur = pe.Node(interface=Blur(fwhm=step_x*15),
                                name='iso_fit_mask_blur_stage_%02d' % snum) # ==> $isomodel_base.msk
    workflow.connect(norm_tmp, 'output_file', iso_fit_mask_blur, 'input_file')

    iso_fit_mask_math = pe.Node(interface=Math(test_gt=0.1),
                                name='iso_fit_mask_math_stage_%02d' % snum) # ==> $isomodel_base.fit-msk.mnc
    workflow.connect(iso_fit_mask_blur, 'output_file', iso_fit_mask_math, 'input_files') # FIXME does Nipype pass a single file as a list?

    # Register each file in the input series.
    if end_stage == 'lin':
        assert LINMETHOD == 'bestlinreg' # other interfaces not implemented

        print 'end_stage == \'lin\' so doing bestlinreg...'

        registers_tmp = pe.MapNode(interface=BestLinReg(),
                       name='bestlinreg_stage_%02d' % snum,
                       iterfield=['target']) # ==> $modxfm[$f]

        if snum == lastlin:
            last_linear_stage_xfm = registers_tmp # FIXME use this!

        # The single iso model is the source.
        workflow.connect(iso_tmp, 'output_file', registers_tmp, 'source')

        # We iterate over each possible target, namely all of the
        # iso models from the preprocessing step.
        workflow.connect(iso, 'output_file', registers_tmp, 'target')
    else:
        # Use the last linear xfm as a starting point. In particular,
        # we concat together the identity model from 'identity_gennlxfm'
        # along with each xfm file from the last linear stage. To get these
        # inputs together we use a Merge node.

        # Create a node to merge the inputs.
        merge_xfm_tmp = pe.Node(interface=utils.Merge(2),
                                name='merge_xfm_stage_%02d' % snum)
        workflow.connect(identity_gennlxfm,     'output_file', merge_xfm_tmp, 'in1')
        workflow.connect(last_linear_stage_xfm, 'output_xfm',  merge_xfm_tmp, 'in2')

        # Now we can concat them.
        xfm_concat_tmp = pe.Node(interface=XfmConcat(),
                                 name='xfm_concat_stage_%02d' % snum) # ==> $initcnctxfm
        workflow.connect(merge_xfm_tmp, 'out', xfm_concat_tmp, 'input_files')

        nlpfit_tmp = pe.MapNode(interface=NlpFit(),
                                name='nlpfit_tmp_stage_%02d' % snum,
                                iterfield=['target']) # ==> modxfm[$f]

        assert datasource_fit_confs[snum] is not None
        workflow.connect(datasource_fit_confs[snum], 'outfiles', nlpfit_tmp, 'config_file') # this is a single config file

        workflow.connect(xfm_concat_tmp,     'output_file', nlpfit_tmp, 'init_xfm')

        # FIXME why doesn't this work???
        workflow.connect(iso_fit_mask_math , 'output_file', nlpfit_tmp, 'source_mask')

        workflow.connect(iso, 'output_file', nlpfit_tmp, 'target') # iteration; FIXME Is this coming from the right source!?
        workflow.connect(iso_tmp, 'output_file', nlpfit_tmp, 'source')

    # Average xfms.
    if end_stage == 'lin':
        mod_xfm = registers_tmp
        xfm_avg_tmp = pe.Node(interface=XfmAvg(ignore_nonlinear=True),
                                               name='xfm_avg_stage_%02d' % snum)
        workflow.connect(mod_xfm, 'output_xfm', xfm_avg_tmp, 'input_files') # FIXME does Nipype pass a single file as a list?
    else:
        mod_xfm = nlpfit_tmp
        xfm_avg_tmp = pe.Node(interface=XfmAvg(ignore_linear=True),
                                               name='xfm_avg_stage_%02d' % snum)
        workflow.connect(mod_xfm, 'output_xfm', xfm_avg_tmp, 'input_files') # FIXME does Nipype pass a single file as a list?

    # Resample each file in the input series.


    # 1. Invert model xfm.
    resample_invert_mod_xfm = pe.MapNode(interface=XfmInvert(),
                                         name='resample_xfm_invert_stage_%02d' % snum,
                                         iterfield=['input_file']) # ==> $invxfm
    workflow.connect(mod_xfm, 'output_xfm', resample_invert_mod_xfm, 'input_file')

    # 2. Concat $invxfm and $avgxfm.
    #
    # 2a. Merge $invxfm and $avgxfm.
    resample_merge_invxfm_avgxfm = pe.Node(interface=utils.Merge(2),
                                           name='resample_merge_invxfm_avgxfm_stage_%02d' % snum)
    workflow.connect(resample_invert_mod_xfm, 'output_file', resample_merge_invxfm_avgxfm, 'in1')
    workflow.connect(xfm_avg_tmp,             'output_file', resample_merge_invxfm_avgxfm, 'in2')

    # 2b. Run xfmConcat on the merged input.
    resample_concat = pe.Node(interface=XfmConcat(),
                              name='resample_concat_stage_%02d' % snum) # ==> resxfm
    workflow.connect(resample_merge_invxfm_avgxfm, 'out', resample_concat, 'input_files')

    # 3. Resample.
    resample = pe.MapNode(interface=Resample(sinc_interpolation=True),
                          name='resample_stage_%02d' % snum,
                          iterfield=['input_file']) # ==> $rsmpl[$f]
    workflow.connect(resample_concat, 'output_file', resample, 'transformation')
    workflow.connect(iso_tmp,         'output_file', resample, 'like')
    workflow.connect(norm,            'output_file', resample, 'input_file') # this is iterated over

    # Pictures.
    resample_pik = pe.MapNode(interface=Pik(triplanar=True, sagittal_offset=10),
                              name='resample_pik_stage_%02d' % snum,
                              iterfield=['input_file'])
    workflow.connect(resample, 'output_file', resample_pik, 'input_file')

    # Create model.
    #
    # FIXME ignoring tmp dir... '-tmpdir', "$opt{'workdir'}/tmp",
    big_average = pe.Node(interface=BigAverage(output_float=True, robust=True),
                          name='big_average_stage_%02d' % snum) # ==> sd_file, output is: iavgfile

    workflow.connect(resample, 'output_file', big_average, 'input_files')

    # TODO
    # &do_cmd_batch("cIAV$$-$snum_txt", "IAV$$-$snum_txt",
    # 'mincpik', '-clobber',
    # '-triplanar', '-horizontal',
    # '-scale', 4, '-tilesize', 400,
    # '-sagittal_offset', 10,
    # $iavgfile, $iavgfilechk) if $opt{'check'};
    # &do_cmd_batch("cIAV$$-$snum_txt", "IAV$$-$snum_txt",
    # 'mincpik', '-clobber',
    # '-triplanar', '-horizontal',
    # '-scale', 4, '-tilesize', 400,
    # '-lookup', '-hotmetal',
    # '-sagittal_offset', 10,
    # $istdfile, $istdfilechk) if $opt{'check'};

    # FIXME Various not done bits from the end of volgenmodel.

    # FIXME Use output of big_average as the new model; for the moment just send it to the sink.


    workflow.connect(big_average, 'output_file', datasink, 'big_average_stage_%02d' % snum)

    # FIXME Change this to the actual final model blah, may need symmetric stuff applied.
    stage_models[snum] = big_average


# Run single-core:
#
# workflow.run()
#
# Run multicore:
#
# workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 4})
