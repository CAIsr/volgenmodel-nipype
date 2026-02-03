# volgenmodel-nipype
volgenmodel-nipype is the port of [volgenmodel](https://github.com/andrewjanke/volgenmodel) to [Nipype](https://github.com/nipy/nipype). It creates nonlinear models from a series of input MINC files.

## Use Volgenmodel inside a Neurodesk.org environment
```bash
git clone https://github.com/CAIsr/volgenmodel-nipype.git
git clone https://github.com/CAIsr/volgenmodel-fast-example.git 
ml minc
python3 volgenmodel-nipype/volgenmodel.py --input_dir volgenmodel-fast-example
```

## Adjusting memory
Every nipype node has a default for the memory consumption, but this might not be enough depending on the input data. If you get out of memory errors, adjust the memory scale, e.g. by 2:
```
python3 volgenmodel.py --input_dir data/ --work_dir work/ --memory_scale 2.0
```

## Reducing SLURM job count
When running with `--run SLURMGraph`, each workflow node becomes a separate SLURM job. This can create hundreds of jobs that all wait in the queue. There are several ways to reduce job count:

### Option 1: Combined preprocessing (recommended)
Use `--combine_jobs` to merge all preprocessing steps (NIfTI conversion, volcentre, normalize, volpad, voliso) into a single job per input file:
```bash
python3 volgenmodel.py --run SLURMGraph --combine_jobs --input_dir data/
```
This typically reduces preprocessing from 5-6 jobs per input file to just 1.

### Option 2: Use a faster fit_stages preset
The default pipeline uses 20 fit stages. Use presets to reduce:
```bash
# Fast: 7 stages (~65% fewer jobs, good for testing)
python3 volgenmodel.py --run SLURMGraph --fit_stages fast --input_dir data/

# Medium: 13 stages (~35% fewer jobs, balanced quality/speed)
python3 volgenmodel.py --run SLURMGraph --fit_stages medium --input_dir data/
```

### Option 3: Combine both approaches
```bash
python3 volgenmodel.py --run SLURMGraph --combine_jobs --fit_stages medium \
    --slurm_account myaccount --slurm_partition normal --input_dir data/
```

### Estimating job count
With default settings and N input files, you get approximately:
- Default: ~5-6 preprocessing + ~20 fit stages × (multiple nodes per stage) = many hundreds of jobs
- `--combine_jobs`: Reduces preprocessing to N jobs
- `--fit_stages fast`: Reduces fit stages from 20 to 7
- Combined: Can reduce total jobs by 60-70%

## Use Volgenmodel as a docker container
this project maintains a docker container with volgenmodel and minc setup and configured: https://github.com/SaibotMagd/volgenmodel-docker


## Install for Windows Subsystem for Linux or Ubuntu 16.04
Install minc: https://bic-mni.github.io
```bash
    wget http://packages.bic.mni.mcgill.ca/minc-toolkit/Debian/minc-toolkit-1.9.16-20180117-Ubuntu_16.04-x86_64.deb
    sudo apt-get install libc6 libstdc++6 imagemagick perl octave
    sudo dpkg -i minc-toolkit-1.9.16-20180117-Ubuntu_16.04-x86_64.deb
    sudo apt-get install libgl1-mesa-glx libglu1-mesa
    rm minc-toolkit-1.9.16-20180117-Ubuntu_16.04-x86_64.deb

    vi .bashrc
    source /opt/minc/1.9.16/minc-toolkit-config.sh
    export PERL5LIB=/opt/minc/1.9.16/perl
```

Clone code and test-data:    
```bash
git clone https://github.com/CAIsr/volgenmodel-nipype.git
git clone https://github.com/CAIsr/volgenmodel-fast-example.git
```

install miniconda:
```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

start new terminal and install packages required:
```bash
conda install --channel conda-forge nipype
pip install pydot
```

add additonal scripts to PATH
```bash
cd volgenmodel-nipype/extra-scripts
echo "export PATH="`pwd -P`":\$PATH" >> ~/.bashrc
```

you need a working octave installed. Somehow the minc libs break octave. Sometimes this fixes it:
select /usr/lib/lapack/liblapack.so.3
```bash
sudo update-alternatives --config liblapack.so.3
```
or load octave using a module:
```bash
module load octave/4.2.1
```

start new temrinal and run volgenmodel with the test data:
```bash
cd volgenmodel-nipype/
python3 volgenmodel.py --input_dir ../volgenmodel-fast-example
```



The final model should look like this:

![mouse model triplanar](https://raw.githubusercontent.com/carlohamalainen/volgenmodel-fast-example/master/model-2016-01-09.png)

It's possible to apply the estimated flow fields to other images that where aligned with the original scans. Here is one example how this can be done (atlas was estimated on Magnitude data and applied to QSM data):
```
./new_data_to_atlas_space.py \
    --name "QSM template" \
    --xfm_dir "data/magnitude_template/workflow/xfmconcat_9" \
    --xfm_pattern "*/*.xfm" \
    --source_dir "data/qsm/qsm_final/mnc" \
    --source_pattern "*.mnc" \
    --atlas_dir "data/magnitude_template/workflow/voliso_9" \
    --atlas_pattern "*.mnc" \
    --work_dir "data/qsm_template/work" \
    --out_dir "data/qsm_template/out" 
```

# Citation
This method is an implementation of the technique described in this paper:

   http://www.ncbi.nlm.nih.gov/pubmed/25620005

If you use it in a publication please cite:

   Janke AL, Ullmann JF, Robust methods to create ex vivo minimum
deformation atlases for brain mapping.
   Methods. 2015 Feb;73:18-26. doi: [10.1016/j.ymeth.2015.01.005](http://dx.doi.org/10.1016/j.ymeth.2015.01.005)
