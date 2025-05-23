import sys
import os
import argparse
relative_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(relative_root_path)

from env.imports import *
from sim.sim import Simulation


def single_sim_run(
    feature_type='transcriptome',
    use_shared_regions=False,
    test_shared_regions=False,
    omit_subcortical=False,
    parcellation='S100',
    hemisphere='both',
    connectome_target='FC',
    binarize=False,
    impute_strategy=None,
    sort_genes=None,
    gene_list='0.2',
    cv_type='random',
    resolution=1.0,
    random_seed=42,
    search_method=('random', 'mse', 5),
    track_wandb=False,
    skip_cv=False,
    model_type='dynamic_mlp',
    use_gpu=True, 
    null_model='none',
    use_folds=[0, 1, 2, 3]
):
    """
    Runs a single simulation for a given feature type and model configuration.

    Running options: 
    - Jupyter notebook: 
        Start HPC job with GPU, activate environment, run single_sim_run(params) (no wandb tracking available)
    - Command line: 
        Start gpu node in Greene terminal, ssh into GPU node, activate singularity environment with -nv support and -ssl certs, run python -m sim.sim_run example_config.yml
    - SBATCH job: 
        Setup SBATCH job in sim/experiment_configs/example_sbatch_config.yml
        Run sbatch run_sim.sh from root
    
    Parameters:
    ----------
    feature_type : list of dict
        List of feature dictionaries specifying the features to use.
        Each dict maps feature name to parameters (or None if no parameters).
        E.g. [{'transcriptome': None}, {'euclidean': None}]
    
    use_shared_regions : bool, optional
        Whether to use shared regions between hemispheres. Default: False
    
    test_shared_regions : bool, optional
        Whether to test on shared regions. Default: False
    
    omit_subcortical : bool, optional
        Whether to omit subcortical regions. Default: False
    
    parcellation : str, optional
        Parcellation scheme to use. Default: 'S100'
    
    hemisphere : str, optional
        Which hemisphere to use. Options: 'left', 'right', 'both'. Default: 'both'
    
    connectome_target : str, optional
        Target connectome type to predict. Options: 'FC' (functional) or 'SC' (structural).
        Default: 'FC'
    
    binarize : bool, optional
        Whether to binarize the connectome. Default: False

    impute_strategy : str, optional
        Imputation strategy to use. Default: None. Options: None, 'mirror', 'interpolate', 'mirror_interpolate'

    sort_genes : str, optional
        Sort genes based on reference genome order 'refgenome', or by mean expression across brain 'expression', or alphabetically (None). Default: 'expression'
    
    gene_list : str, optional
        Gene list identifier to use. Default: '0.2'
    
    cv_type : str
        Type of cross-validation to use. Options: 'random', 'community', 'schaefer', 'spatial'
    
    resolution : float, optional
        Resolution parameter for community detection. Default: 1.0
    
    random_seed : int, optional
        Random seed for reproducibility. Default: 42
    
    search_method : tuple, optional
        Hyperparameter search configuration as (method, metric, n_iter).
        method: 'random', 'grid', 'bayes', 'wandb'
        metric: 'mse', 'pearson', 'r2'
        n_iter: number of search iterations
        Default: ('random', 'mse', 5)
    
    track_wandb : bool, optional
        Whether to log metrics to Weights & Biases. Default: False
    
    skip_cv : bool, optional
        Whether to skip cross-validation. Default: False
    
    model_type : str
        Type of model to use. Options: 'linear', 'ridge', 'pls', 'xgboost', 'dynamic_mlp',
        'random_forest', 'bilinear_lowrank', 'bilinear_SCM', 'shared_mlp_encoder',
        'shared_linear_encoder', 'shared_transformer'
    
    use_gpu : bool
        Whether to use GPU acceleration where available

    null_model : str
        Whether to generate a spatial null model of the transcriptome
        Options: 'none', 'random', 'spatial'
    Returns:
    -------
    single_model_results : list
        List containing simulation results for the specified model configuration
    """
    start_time = time.time()
    sim = Simulation(
                    feature_type=feature_type,
                    use_shared_regions=use_shared_regions,
                    test_shared_regions=test_shared_regions,
                    omit_subcortical=omit_subcortical,
                    parcellation=parcellation,
                    hemisphere=hemisphere,
                    connectome_target=connectome_target,
                    binarize=binarize,
                    impute_strategy=impute_strategy,
                    sort_genes=sort_genes,
                    gene_list=gene_list,
                    cv_type=cv_type,
                    resolution=resolution,
                    random_seed=random_seed,
                    model_type=model_type,
                    gpu_acceleration=use_gpu,
                    skip_cv=skip_cv,
                    null_model=null_model
                )
    
    if search_method[0] == 'wandb':
        sim.run_sim_torch(search_method, track_wandb, use_folds)
    else:
        sim.run_sim(search_method, track_wandb)
    
    elapsed_time = time.time() - start_time
    print(f'Simulation completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)')
    return

def load_config(config_path):
    config_path = os.path.join(absolute_root_path, 'sim/experiment_configs', config_path)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simulation with config file and optional overrides')
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('--model_type', help='Override model type from config')
    parser.add_argument('--n_cvs', type=int, help='Override number of CVs from config')
    parser.add_argument('--feature_type', nargs='+', help='Override feature type from config. Options: transcriptome, euclidean')
    parser.add_argument('--random_seed', type=int, help='Override random seed from config')
    parser.add_argument('--null_model', help='Override null model type from config')
    parser.add_argument('--use_folds', nargs='+', type=int, help='Override use folds from config')
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Override config values if command line arguments are provided
    if args.model_type:
        config['model_type'] = args.model_type
    if args.n_cvs is not None:
        # Update the search_method tuple with new n_cvs value
        search_method = config.get('search_method', ('wandb', 'mse', 5))
        config['search_method'] = (search_method[0], search_method[1], args.n_cvs)
    if args.feature_type:
        # Split feature types and convert to list of dicts
        config['feature_type'] = [{ft: None} for ft in args.feature_type]
    if args.random_seed is not None:
        config['random_seed'] = args.random_seed
    if args.null_model:
        config['null_model'] = args.null_model
    if args.use_folds:
        config['use_folds'] = args.use_folds

    print(f"Running simulation with config: {args.config}")
    print(f"Model type: {config['model_type']}")
    print(f"Search method: {config['search_method']}")
    print(f"Feature type: {config['feature_type']}")
    print(f"Random seed: {config['random_seed']}")
    print(f"Null model: {config['null_model']}")
    print(f"Use folds: {config['use_folds']}")
    print(torch.cuda.is_available())    
    print(os.environ.get("CUDA_VISIBLE_DEVICES"))
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} - Memory Allocated: {torch.cuda.memory_allocated(i)/1024**3:.2f} GB")

    results = single_sim_run(**config)