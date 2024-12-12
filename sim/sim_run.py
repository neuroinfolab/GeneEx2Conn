# Gene2Conn/sim/sim_run.py

# imports
from imports import * 

# data load
from data.data_load import load_transcriptome, load_connectome
import data.data_load
importlib.reload(data.data_load)

# data utils
from data.data_utils import (
    reconstruct_connectome,
    reconstruct_upper_triangle,
    make_symmetric,
    expand_X_symmetric,
    expand_Y_symmetric,
    expand_X_symmetric_shared,
    expand_X_Y_symmetric_conn_only,
    expand_shared_matrices,
    expand_X_symmetric_w_conn, 
    process_cv_splits, 
    process_cv_splits_conn_only_model, 
    expanded_inner_folds_combined_plus_indices,
    expanded_inner_folds_combined_plus_indices_connectome
)
import data.data_utils
importlib.reload(data.data_utils)

# cross-validation classes
from data.cv_split import (
    RandomCVSplit, 
    SchaeferCVSplit, 
    CommunityCVSplit, 
    SubnetworkCVSplit
)

# sim utility functions
import sim.sim_utils
from sim.sim_utils import grid_search_init, drop_test_network, find_best_params
importlib.reload(sim.sim_utils)

import sim.sim
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


def single_sim_run(feature_type, cv_type, model_type, use_gpu, connectome_target='FC', feature_interactions=None, use_shared_regions=False, test_shared_regions=False, omit_subcortical=False, parcellation='S100', gene_list='0.2', resolution=1.0, random_seed=42, save_sim=False, search_method=('random', 'mse'), save_model_json=False, track_wandb=False):
    """
    Runs a single simulation for a given feature type and model configuration.

    This function initializes and runs a simulation for various types of features (e.g., transcriptome, functional, 
    structural data) and model configurations (e.g., ridge regression, XGBoost, neural networks). It handles different 
    scenarios based on input parameters such as cross-validation type, GPU acceleration, feature types, and shared region 
    handling for connectome prediction tasks. 

    Parameters:
    ----------
    feature_type : str
        The type of feature used in the simulation. Options include: 
        'transcriptome', 'transcriptomePCA', 'functional', 'structural', 'structural_spectralL', 'structural_spectralA', 'euclidean'.
    
    cv_type : str
        The type of cross-validation method used (e.g., 'random', 'community').
    
    model_type : str
        The machine learning model type used in the simulation. Options include: 
        'ridge', 'random_forest', 'xgboost', 'dynamic neural net' etc.
    
    use_gpu : bool
        If True, the simulation will use GPU acceleration for models and computations where applicable.
    
    connectome_target : str, optional
        Target connectome type to predict. Options are 'FC' (functional) or 'SC' (structural).
    
    summary_measure : str, optional
        Summary measure used in the simulation.
        For example:
        - 'kronecker' for Kronecker product measure 
        - 'strength_and_corr' for structural summary measures
        - An integer value (e.g. '5', '10') specifying number of components to use from spectral embeddings
        - A negative integer (e.g. '-5', '-10') to use last N components from spectral embeddings

    use_shared_regions : bool, optional
        Whether to include shared brain regions in the analysis. 
    
    test_shared_regions : bool, optional
        Whether to test on shared regions. 

    include_conn_feats : bool, optional
        Whether to include functional connectivity features in the analysis 
        Note - will append connectivity features to the end of the feature vector. will not include connectivity to test regions. 

    resolution : float, optional
        Resolution parameter for the community splits. 
    
    random_seed : int, optional
        Seed for community splits. 
    
    save_sim : bool, optional
        If True, the simulation results will be saved to disk. 
    
    search_method : str, optional
        The hyperparameter search method to use. Options include: 'random', 'grid', 'bayes'. 

    save_model_json : bool, optional
        If True, the model JSON will be saved to disk. This is only applicable for XGBoost models.

    Returns:
    -------
    single_model_results : list
        A list containing the results of the simulation for the specified model type.

    Example:
    -------
    single_sim_run(
        feature_type=['transcriptomePCA'], 
        cv_type='random', 
        model_type='mlp', 
        use_gpu=True, 
        summary_measure='kronecker', 
        resolution=1.0, 
        random_seed=42, 
        search_method='bayes'
    )
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
                    predict_connectome_from_connectome=False,
                    include_conn_feats=False,
                    test_shared_regions=test_shared_regions,
                    # added in new parameters
                    omit_subcortical=omit_subcortical,
                    parcellation=parcellation,
                    gene_list=gene_list,
                    connectome_target=connectome_target,
                    save_model_json=save_model_json
                )
    
    sim.run_sim(search_method, track_wandb)
    single_model_results.append(sim.results)
    
    # Save sim data
    if save_sim:
        save_sims(single_model_results, feature_type, cv_type, model_type, use_shared_regions, test_shared_regions, search_method, resolution, random_seed, connectome_target)
    
    return single_model_results

