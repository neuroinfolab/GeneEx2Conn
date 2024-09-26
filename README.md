# GeneEx2Conn

## Description
This repository contains code for linking gene expression in the human brain to the connectome. 
The repo contains two primary modalities: transcriptome harmonization and connectome prediction. See overleaf for more details.

## Setup instructions
- To use this repo, setup a conda environment using GeneEx2Conn_env.yml from 'env' directory. If there are missing packages delete and recreate the environment with GeneEx2Conn_env_all.yml. Then setup ENIGMA as specified in 'Developer notes' below. Also, manually install the latest version of torch for MLP functionality, pip3 install torch torchvision torchaudio.
- If using harmonize.py (or any other substantial development) the assumption is that there is a data folder stored locally, one directory up from where the repo is setup. Reach out to asr655 for access to this data folder. 

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
- /metrics
    - classes and functions for evaluating the accuracy of connectome predictions 
- /models
    - base class for all the models
    - parameter grids and distributions for hyperparameter search
- /harmonize
    - classes and functions for loading and harmonizing gene expression datasets
    - transcriptome_harmonization_EDA.ipynb demos harmonizer utilities
- /notebooks
    - notebooks demonstrating repo functionality. need to be moved to the root directory to run 
    - data_demo shows how to load and split the data
    - plot_demo shows the plotting modalities for outputted simulations
    - sim_demo shows how to run single and multi sims
    - finetune_demo shows single sim runs with specified parameter grids

## Developer notes
- A few extra steps are required to get `enigmatoolbox` working:
    - Manually install enigma: `git clone https://github.com/MICA-MNI/ENIGMA.git; cd ENIGMA; python setup.py install`
    - Replace all references to `error_bad_lines=False` in code with `on_bad_lines='skip'` following [this](https://stackoverflow.com/questions/69513799/pandas-read-csv-the-error-bad-lines-argument-has-been-deprecated-and-will-be-re).
    - In `enigmatoolbox/datasets/matrices/hcp_connectivity/` rename all `funcMatrix_with_ctx_schaefer_{100,200,300,400}` to `funcMatrix_with_sctx_schaefer_{100,200,300,400}`.
