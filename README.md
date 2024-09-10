# Gene2Conn
Repository for linking gene expression in the human brain to the connectome

## Basic overview 
This notebook contains modalities for predicting the connectome from the transcriptome. 
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

To use this repo, setup a conda environment using Gene2Conn_env.yml

A few extra steps are required to get `enigmatoolbox` working:

- Manually install enigma: `git clone https://github.com/MICA-MNI/ENIGMA.git; cd ENIGMA; python setup.py install`
- Replace all references to `error_bad_lines=False` in code with `on_bad_lines='skip'` following [this](https://stackoverflow.com/questions/69513799/pandas-read-csv-the-error-bad-lines-argument-has-been-deprecated-and-will-be-re).
- In `enigmatoolbox/datasets/matrices/hcp_connectivity/` rename all `funcMatrix_with_ctx_schaefer_{100,200,300,400}` to `funcMatrix_with_sctx_schaefer_{100,200,300,400}`.