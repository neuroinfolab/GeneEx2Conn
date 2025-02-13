# GeneEx2Conn
This repository contains code for predicting human brain connectivity (_the connectome_) from regional gene expression (_the transcriptome_). 

## Description
At a high-level, the connectome prediction problem is a small-network link prediction problem. The nodes of the network (brain regions) come from a high-dimensional feature space; $O(n^4)$ gene expression measures per region. The goal is to predict edge connectivity strength between any pair of regions across the brain based solely on their node-wise features. This repo implements several architectures for solving the connectome prediction problem, with emphasis on encoder-based models that capture intra and inter-region genetic interactions.

<div style="display: flex; justify-content: center;">
  <table>
    <tr>
      <td><img src="https://github.com/neuroinfolab/GeneEx2Conn/blob/master/glass/S400_LAMA1_expression.png" height="300"></td>
      <td><img src="https://github.com/neuroinfolab/GeneEx2Conn/blob/master/glass/cv_split_s400_structural_connectivity_spatial_split_10-3.gif" height="300"></td>
    </tr>
    <tr>
      <td align="center"><strong>Normalized expression of LAMA1 across brain</strong></td>
      <td align="center"><strong>10-fold spatial split visualization</strong></td>
    </tr>
  </table>
</div>

## Running a sim
The core functionality of `GeneEx2Conn` is the sim class. The sim class enables users to run detailed experiments targetting a variety of neuroscientific hypothesis. For example, a user could setup an experiment to evaluate how well linear vs non-linear models can predict connectivity in different subnetworks of the left-hemisphere. See `sim/sim_run.py` for details on all possible options. 

Here's an example of how to run a sim:
```
single_sim_run(
              cv_type='spatial',
              random_seed=42,
              model_type='shared_transformer',
              use_gpu=True,
              feature_type=[{'transcriptome': None}], 
              connectome_target='SC',
              omit_subcortical=False,
              parcellation='S400', 
              gene_list='0.2',
              hemisphere='left',
              search_method=('wandb', 'mse', 3),
              track_wandb=True,
              skip_cv=False
              )
```

## Repo Overview

- **`/sim`** → Simulation class, cross-validation logic, wandb tracking  
- **`/data`** → Data loading, preprocessing, custom train-test splits
- **`/glass`** → Select visualizations  
- **`/models`** → ML & DL models, hyperparameter tuning, training pipeline  
- **`/models/metrics`** → Evaluation functions for connectome predictions 
- **`/notebooks`** → Experiment notebooks _(move to root to run)_  
  - **`/demos`** → Pulling data, cross-validation logic, visualization, simulation demos  
  - Organized folders for **presentations & deadlines**
 

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
- If doing any substantial development with the repo, the assumption is that there is a `/GeneEx2Conn_data` folder one directory up from where the repo is setup. If you don't have access to this folder or if there are issues in pulling data from here, reach out to asr655 for access to this folder.
- To use this repo outside of NYU HPC Greene, setup a conda environment using `GeneEx2Conn_env_combined.yml` from 'env' directory. If there are missing packages delete and recreate the environment with `GeneEx2Conn_env_all.yml`. Then setup ENIGMA below. If necessary, manually install the latest version of torch for deep learning functionality, `pip3 install torch torchvision torchaudio`.
- A few extra steps may be required to get `enigmatoolbox` working:
    - Manually install enigma: `git clone https://github.com/MICA-MNI/ENIGMA.git; cd ENIGMA; python setup.py install`
    - Replace all references to `error_bad_lines=False` in code with `on_bad_lines='skip'` following [this](https://stackoverflow.com/questions/69513799/pandas-read-csv-the-error-bad-lines-argument-has-been-deprecated-and-will-be-re).
    - In `enigmatoolbox/datasets/matrices/hcp_connectivity/` rename all `funcMatrix_with_ctx_schaefer_{100,200,300,400}` to `funcMatrix_with_sctx_schaefer_{100,200,300,400}`.
