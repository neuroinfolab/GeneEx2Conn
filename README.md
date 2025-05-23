This repository contains code for predicting human brain connectivity from regional gene expression. 

The core functionality of `SMT` is the sim class. The sim class enables users to run detailed experiments targetting a variety of neuroscientific hypothesis. See `sim/sim_run.py` for details on all possible options. 

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

- **`/sim`** → Simulation class, cross-validation logic, wandb tracking, null spin generation
- **`/data`** → Data loading, preprocessing, custom train-test splits
- **`/models`** → ML & DL models, hyperparameter tuning, training pipeline  
- **`/models/metrics`** → Evaluation functions for connectome predictions 
- **`/notebooks`** → Experiment notebooks
  - **`/NeurIPS`** → Notebooks used for data analysis and figure generation, model checkpoints and artifacts