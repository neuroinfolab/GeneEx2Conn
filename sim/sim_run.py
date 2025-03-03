# imports
from env.imports import *

from sim.sim import Simulation

def open_pickled_results(file, added_dir='', backup=False): # Specify the path to your pickle file
    """
    Function to open any pickle file from sim_results directory
    If in subdirectory of sim_results pass to added dir as 'subdir/'
    """
    
    if backup:
        pickle_file_path = "/scratch/asr655/neuroinformatics/GeneEx2Conn_backup/sim/sim_results/" + added_dir + file
    else:
        pickle_file_path = os.getcwd() + '/sim/sim_results/' + added_dir + file

    # Open the pickle file in read mode
    with open(pickle_file_path, "rb") as file:
        # Load the data from the pickle file
        pickle_results = pickle.load(file)
    
    return pickle_results
    
def save_sims(multi_model_results, feature_type, cv_type, model_type, use_shared_regions, test_shared_regions, search_method, resolution, random_seed, connectome_target='FC'): 
    """
    Function to save all sim results to a pickle file
    """

    sim_results_file_path = os.getcwd() + '/sim/sim_results/'
    
    # Build filename components
    results_file_str = ""
    for feature in feature_type:
        for key, value in feature.items():
            if value is None:
                results_file_str += str(key)
            else:
                results_file_str += f"{key}_{value}"
        results_file_str += "+"
    results_file_str = results_file_str.rstrip("_")  # Remove trailing underscore
    
    results_file_str += f"_{connectome_target}_{model_type}_{cv_type}"    
    
    if cv_type == "community":
        results_file_str += str(resolution)
    
    results_file_str += '_' + str(random_seed)
    results_file_str += "_" + search_method[0] + "_" + search_method[1] + "_search"

    if use_shared_regions: 
        results_file_str += "_useshared"
        if test_shared_regions: 
            results_file_str += "_testshared"
        else: 
            results_file_str += "_trainshared"        

    results_file_str = re.sub(r'[^\w\s_-]', '', str(results_file_str))

    results_file_path = os.path.join(str(sim_results_file_path), results_file_str)
    results_file_path_pickle = results_file_path + '.pickle'
    
    # Save the list to a file using pickle
    with open(results_file_path_pickle, 'wb') as f:
        pickle.dump(multi_model_results, f)
    
    print("Simulation results have been saved to ", results_file_path_pickle)
    return


def single_sim_run(feature_type, cv_type, model_type, use_gpu, connectome_target='FC', binarize=False, feature_interactions=None, use_shared_regions=False, test_shared_regions=False, omit_subcortical=False, parcellation='S100', gene_list='0.2', hemisphere='both', resolution=1.0, random_seed=42, save_sim=False, search_method=('random', 'mse', 5), save_model_json=False, track_wandb=False, skip_cv=False):
    """
    Runs a single simulation for a given feature type and model configuration.

    Parameters:
    ----------
    feature_type : list of dict
        List of feature dictionaries specifying the features to use.
        Each dict maps feature name to parameters (or None if no parameters).
        E.g. [{'transcriptome': None}, {'euclidean': None}]
    
    cv_type : str
        Type of cross-validation to use. Options: 'random', 'community', 'schaefer', 'spatial'
    
    model_type : str
        Type of model to use. Options: 'linear', 'ridge', 'pls', 'xgboost', 'dynamic_mlp',
        'random_forest', 'bilinear_lowrank', 'bilinear_SCM', 'shared_mlp_encoder',
        'shared_linear_encoder', 'shared_transformer'
    
    use_gpu : bool
        Whether to use GPU acceleration where available
    
    connectome_target : str, optional
        Target connectome type to predict. Options: 'FC' (functional) or 'SC' (structural).
        Default: 'FC'
    
    binarize : bool, optional
        Whether to binarize the connectome. Default: False

    feature_interactions : bool, optional
        Whether to include feature interactions. Default: None
    
    use_shared_regions : bool, optional
        Whether to use shared regions between hemispheres. Default: False
    
    test_shared_regions : bool, optional
        Whether to test on shared regions. Default: False
    
    omit_subcortical : bool, optional
        Whether to omit subcortical regions. Default: False
    
    parcellation : str, optional
        Parcellation scheme to use. Default: 'S100'
    
    gene_list : str, optional
        Gene list identifier to use. Default: '0.2'
    
    hemisphere : str, optional
        Which hemisphere to use. Options: 'left', 'right', 'both'. Default: 'both'
    
    resolution : float, optional
        Resolution parameter for community detection. Default: 1.0
    
    random_seed : int, optional
        Random seed for reproducibility. Default: 42
    
    save_sim : bool, optional
        Whether to save simulation results. Default: False
    
    search_method : tuple, optional
        Hyperparameter search configuration as (method, metric, n_iter).
        method: 'random', 'grid', 'bayes', 'wandb'
        metric: 'mse', 'pearson', 'r2'
        n_iter: number of search iterations
        Default: ('random', 'mse', 5)
    
    save_model_json : bool, optional
        Whether to save model architecture as JSON (XGBoost only). Default: False
    
    track_wandb : bool, optional
        Whether to log metrics to Weights & Biases. Default: False
    
    skip_cv : bool, optional
        Whether to skip cross-validation. Default: False

    Returns:
    -------
    single_model_results : list
        List containing simulation results for the specified model configuration
    """

    # List to store each model types results
    single_model_results = []

    # Structural
    sim = Simulation(
                    feature_type=feature_type,
                    cv_type=cv_type,
                    model_type=model_type, 
                    feature_interactions=feature_interactions, 
                    gpu_acceleration=use_gpu,
                    resolution=resolution,
                    random_seed=random_seed,
                    use_shared_regions=use_shared_regions,
                    test_shared_regions=test_shared_regions,
                    omit_subcortical=omit_subcortical,
                    parcellation=parcellation,
                    gene_list=gene_list,
                    hemisphere=hemisphere,
                    connectome_target=connectome_target,
                    binarize=binarize,
                    save_model_json=save_model_json,
                    skip_cv=skip_cv
                )
    
    if search_method[0] == 'wandb':
        sim.run_sim_torch(search_method, track_wandb)
    else:
        sim.run_sim(search_method, track_wandb)
    
    try: 
        single_model_results.append(sim.results)
    except:
        print('No results to append')
    
    # Save sim data
    if save_sim:
        save_sims(single_model_results, feature_type, cv_type, model_type, use_shared_regions, test_shared_regions, search_method, resolution, random_seed, connectome_target)
    
    return single_model_results


def run_simulation_set(model_types,
                      feature_types,
                      parcellations,
                      connectome_targets,
                      random_seeds,
                      cv_types=['random', 'spatial'], 
                      inner_cv_runs=3,
                      skip_cv=False):
    """
    Run a set of simulations with different combinations of parameters.
    
    Args:
        model_types (list): List of model types to test (e.g. ['dynamic_mlp', 'shared_transformer'])
        feature_types (list): List of features to include (e.g. ['transcriptome', 'euclidean'])
        parcellations (list): List of parcellation schemes (e.g. ['S100', 'S400'])
        connectome_targets (list): List of connectome targets (e.g. ['FC', 'SC'])
        random_seeds (list): List of random seeds for multiple runs
        cv_types (list): List of cross-validation types, defaults to ['random', 'spatial']
        inner_cv_runs (int): Number of inner cross-validation runs to perform
        skip_cv (bool): Whether to skip cross validation, defaults to False
    """
    
    for model in model_types:
        for feat in feature_types:
            for parc in parcellations:
                current_hemisphere = 'both' if parc == 'S100' else 'left'
                for target in connectome_targets:
                    for cv in cv_types:
                        for seed in random_seeds:
                            if feat == 'transcriptome':
                                feat_dict = [{'transcriptome': None}]
                            elif feat == 'euclidean':
                                feat_dict = [{'euclidean': None}]
                            elif feat == 'transcriptome+euclidean':
                                feat_dict = [
                                    {'transcriptome': None},
                                    {'euclidean': None}
                                ]

                            print(f"Running simulation with: {model}, {cv}, {parc}, {target}, {feat}, seed={seed}")
                            if model == 'pls':
                                single_sim_run(
                                    cv_type=cv,
                                    random_seed=seed,
                                    model_type=model,
                                    feature_type=feat_dict,
                                    connectome_target=target,
                                    use_gpu=False,
                                    use_shared_regions=False,
                                    test_shared_regions=False,
                                    omit_subcortical=False,
                                    parcellation=parc,
                                    gene_list='0.2',
                                    hemisphere=current_hemisphere,
                                    search_method=('grid', 'mse', 10),
                                    save_sim=False,
                                    track_wandb=True,
                                    skip_cv=skip_cv
                                )
                            elif model == 'xgboost':
                                # Run single simulation
                                single_sim_run(
                                    cv_type=cv,
                                    random_seed=seed,
                                    model_type=model,
                                    feature_type=feat_dict,
                                    connectome_target=target,
                                    use_gpu=True,
                                    use_shared_regions=False,
                                    test_shared_regions=False,
                                    omit_subcortical=False,
                                    parcellation=parc,
                                    gene_list='0.2',
                                    hemisphere=current_hemisphere,
                                    search_method=('bayes', 'mse', 5),
                                    save_sim=False,
                                    track_wandb=True,
                                    skip_cv=skip_cv
                                )
                            elif model == 'shared_transformer':   
                                # Run single simulation
                                single_sim_run(
                                    cv_type=cv,
                                    random_seed=seed,
                                    model_type=model,
                                    feature_type=feat_dict,
                                    connectome_target=target,
                                    use_gpu=True,
                                    use_shared_regions=False,
                                    test_shared_regions=False,
                                    omit_subcortical=False,
                                    parcellation=parc,
                                    gene_list='0.2',
                                    hemisphere=current_hemisphere,
                                    search_method=('wandb', 'mse', inner_cv_runs),
                                    save_sim=False,
                                    track_wandb=True,
                                    skip_cv=skip_cv
                                )
                            else: 
                                # Run single simulation
                                single_sim_run(
                                    cv_type=cv,
                                    random_seed=seed,
                                    model_type=model,
                                    feature_type=feat_dict,
                                    connectome_target=target,
                                    use_gpu=True,
                                    use_shared_regions=False,
                                    test_shared_regions=False,
                                    omit_subcortical=False,
                                    parcellation=parc,
                                    gene_list='0.2',
                                    hemisphere=current_hemisphere,
                                    search_method=('wandb', 'mse', inner_cv_runs),
                                    save_sim=False,
                                    track_wandb=True,
                                    skip_cv=skip_cv
                                )
                            
                            # Clear GPU memory
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            # Clear CPU memory
                            gc.collect()