# GeneEx2Conn

## Description
This repository contains code for predicting the connectome from the transcriptome.

## Repo overview 
- /sim
    - notebook driver files
    - single and multi sim methods for different features, cross-validation styles, and model types
    - /sim_results
        - contains pickle files with the results of different simulation runs
- /data
    - data loading and processing
    - data expansion functions for the transcriptome and connectome
    - custom train-test split classes (random, community, anatomical subnetworks)
- /models
    - base model class with parameter grids and distributions for hyperparameter search
    - dynamic MLP class and sweep configs
    - /metrics
        - classes and functions for evaluating the accuracy of connectome predictions 
- /harmonize
    - classes and functions for loading and harmonizing gene expression datasets
    - transcriptome_harmonization_EDA.ipynb demos harmonizer utilities
- /notebooks
    - notebooks demonstrating repo functionality. need to be moved to the root directory to run 
    - data_demo shows how to load and split the data
    - plot_demo shows the plotting modalities for outputted simulations
    - sim_demo shows how to run single and multi sims
    - finetune_demo shows single sim runs with specified parameter grids

## Setting up Jupyter notebook on NYU HPC (Greene)

To setup your Jupyter notebook on the HPC cluster, follow the instructions here [here](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/open-ondemand-ood-with-condasingularity?authuser=0). The key difference is that instead of setting up your own Singularity image, you can point your Jupyter config to our existing Singularity image (which contains all the necessary python packages).

Specifically, all you need to do is 

```
mkdir -p ~/.local/share/jupyter/kernels
cd ~/.local/share/jupyter/kernels
cp -R /share/apps/mypy/src/kernel_template ./my_env # this should be the name of your Singularity env
cd ./my_env 
```

And then in the file called `python` modify the existing singularity command at the bottom of the file to be

```
OVERLAY_FILE=/scratch/asr655/main_env/overlay-50G-10M.ext3
SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif

singularity exec $nv \
  --overlay $OVERLAY_FILE:rw \
  $SINGULARITY_IMAGE \
  /bin/bash -c "source /ext3/env.sh; $cmd $args"
```

This tells your Jupyter notebook to load its environment from this Singularity image. Now, when you launch your Jupyter notebook from the [OnDemand interface](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/getting-started?authuser=0#h.n3n5bitemm3g) you can select the kernel that you just defined. 

## Developer notes
- If doing any substantial development with the repo, the assumption is that there is a '/GeneEx2Conn_data' folder one directory up from where the repo is setup. If you don't have access to this folder or if there are issues in pulling data from here, reach out to asr655 for access to this folder.
- To use this repo outside of NYU HPC Greene, setup a conda environment using `GeneEx2Conn_env_combined.yml` from 'env' directory. If there are missing packages delete and recreate the environment with `GeneEx2Conn_env_all.yml`. Then setup ENIGMA below. If necessary, manually install the latest version of torch for deep learning functionality, `pip3 install torch torchvision torchaudio`.
- A few extra steps may be required to get `enigmatoolbox` working:
    - Manually install enigma: `git clone https://github.com/MICA-MNI/ENIGMA.git; cd ENIGMA; python setup.py install`
    - Replace all references to `error_bad_lines=False` in code with `on_bad_lines='skip'` following [this](https://stackoverflow.com/questions/69513799/pandas-read-csv-the-error-bad-lines-argument-has-been-deprecated-and-will-be-re).
    - In `enigmatoolbox/datasets/matrices/hcp_connectivity/` rename all `funcMatrix_with_ctx_schaefer_{100,200,300,400}` to `funcMatrix_with_sctx_schaefer_{100,200,300,400}`.
