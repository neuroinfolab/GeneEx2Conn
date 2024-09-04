# GeneExp2Conn

## Description
This repository contains code for linking gene expression in the human brain to the connectome. The repo contains two primary modalities: transcriptome harmonization and connectome prediction. See overleaf for more details. 

## Repo overview 
- /data  contains methods for loading and processing the data. This includes all data expansion functions for the transcriptome and connectome.
- /cv_split contains custom train-test split classes. The data can be partitioned into folds randomly, based on community detection, or based on anatomical subnetworks.
- /sim is the notebook driver. A full sim includes training and testing for the connectivity-connectivity, transcriptome-connectivity, and the combined-connectivity models. Single sims can also be specified for one of the 3 model feature types. A sim takes in key information including the type of split/cross-validation to be used, what parts of the connectome to use for training and testing, the model to use, and whether to GPU accelerate model training and inference. 
- /metrics contains custom classes and functions for evaluating the accuracy of connectome predictions. 
- /models contains a base class for all the models including parameter grids to be used in random or gridsearch.
- /demo_notebooks contains various notebooks utilizing and displaying the above modalities. To run the demo notebooks they need to be moved out of their folder to the main directory. 
    - /data_demo shows how to load and split the data
    - /plot_demo shows the plotting modalities for outputted simulations
    - /sim_demo shows how to run single and multi sims
    - /finetune_demo shows single sim runs with specified parameter grids
- /sim_results contains pickle files with the results of different simulation runs

## Usage instructions
To use this repo, setup a conda environment using Gene2Conn_env.yml

## Developer notes
- Note on replacing .csv file in abagen package
- location of data dir
