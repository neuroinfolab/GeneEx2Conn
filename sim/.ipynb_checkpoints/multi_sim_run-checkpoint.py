# Gene2Conn/sim/multi_sim_run.py

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
import data.cv_split
importlib.reload(data.cv_split)

# prebuilt model classes
from models.prebuilt_models import ModelBuild
import models.prebuilt_models
importlib.reload(models.prebuilt_models)

# metric classes
from metrics.eval import (
    ModelEvaluator,
    pearson_numpy,
    pearson_cupy,
    mse_numpy,
    mse_cupy,
    r2_numpy,
    r2_cupy
)
import metrics.eval
importlib.reload(metrics.eval)

# sim utility functions
import sim.sim_utils
from sim.sim_utils import grid_search_init, drop_test_network, find_best_params
importlib.reload(sim.sim_utils)

import sim.sim
from sim.sim import Simulation


def open_pickled_results(file):# Specify the path to your pickle file
    """
    Function to open any pickle file from sim_results directory
    """
    pickle_file_path = "./sim/sim_results/" + file
    
    # Open the pickle file in read mode
    with open(pickle_file_path, "rb") as file:
        # Load the data from the pickle file
        pickle_results = pickle.load(file)
    
    return pickle_results


def combine_results(conn_file, trans_file, transconn_file, output_file):
    """
    Function to combine single sim results
    """
    # Load individual pickle files
    conn_results = open_pickled_results(conn_file)
    trans_results = open_pickled_results(trans_file)
    transconn_results = open_pickled_results(transconn_file)

    # Remove the unnecessary outer list
    conn_results = conn_results[0] if len(conn_results) == 1 else conn_results
    trans_results = trans_results[0] if len(trans_results) == 1 else trans_results
    transconn_results = transconn_results[0] if len(transconn_results) == 1 else transconn_results

    # Combine the results
    combined_results = [conn_results, trans_results, transconn_results]

    # Save the combined results to a new pickle file
    output_pickle_path = "./sim/sim_results/" + output_file
    with open(output_pickle_path, "wb") as file:
        pickle.dump(combined_results, file)

    print(f"Combined results saved to {output_pickle_path}")
    

def save_sims(multi_model_results, feature_type, cv_type, model_type, use_shared_regions, test_shared_regions, resolution, random_seed): 
    """
    Function to save all sim results to a pickle file
    """

    sim_results_file_path = os.getcwd() + '/sim/sim_results/'
    if feature_type == "all":
        results_file_str = "multi_sim_" + cv_type + "_" + model_type
    elif feature_type == "conn only":
        results_file_str = "single_sim_conn_" + cv_type + "_" + model_type
    elif feature_type == "trans only":
        results_file_str = "single_sim_trans_" + cv_type + "_" + model_type
    elif feature_type == "trans plus conn":
        results_file_str = "single_sim_transplusconn_" + cv_type + "_" + model_type

    
    if cv_type == "community": 
        results_file_str += str(resolution)
        results_file_str += '_' + str(random_seed)
        # take in or out manually 
        # results_file_str += '_PCA' 

        
    if use_shared_regions: 
        results_file_str += "_useshared"
        if test_shared_regions: 
            results_file_str += "_testshared"
        else: 
            results_file_str += "_trainshared"        

    results_file_path = os.path.join(sim_results_file_path, results_file_str)
    results_file_path_pickle = results_file_path + '.pickle'
    
    # Save the list to a file using pickle
    with open(results_file_path_pickle, 'wb') as f:
        pickle.dump(multi_model_results, f)
    
    print("Multi simulation results have been saved.")
    return


def multi_sim_run(cv_type, model_type, use_gpu, use_shared_regions=False, test_shared_regions=False, resolution=1.0, random_seed=42, save_sim=False, search_method='random'):
    """
    Function to run simulations for all possible feature types: connectome only, transcriptome only, connectome+transcriptome
    """
    
    # List to store each model types results
    multi_model_results = []
    feature_type="all"
    
    # Connectome only 
    conn_sim = Simulation(
            cv_type=cv_type, model_type=model_type, gpu_acceleration=use_gpu,
                    predict_connectome_from_connectome=True, resolution=resolution, 
                    random_seed=random_seed, use_shared_regions=use_shared_regions,
                    include_conn_feats=False, test_shared_regions=test_shared_regions
        )

    # Execute sim
    conn_sim.run_sim(search_method)
    multi_model_results.append(conn_sim.results)

    # Transcriptome only 
    trans_sim = Simulation(
            cv_type=cv_type, model_type=model_type, gpu_acceleration=use_gpu,
                    predict_connectome_from_connectome=False,
                    resolution=resolution,random_seed=random_seed,
                    use_shared_regions=use_shared_regions, include_conn_feats=False,
                    test_shared_regions=test_shared_regions
        )

    # Execute sim
    trans_sim.run_sim(search_method)
    multi_model_results.append(trans_sim.results)

    # Connectome and transcriptome
    trans_conn_sim = Simulation(
        cv_type=cv_type, model_type=model_type, gpu_acceleration=use_gpu,
                predict_connectome_from_connectome=False, resolution=resolution,
                random_seed=random_seed, use_shared_regions=use_shared_regions, include_conn_feats=True,
                test_shared_regions=test_shared_regions
    )

    # Execute sim
    trans_conn_sim.run_sim(search_method)
    multi_model_results.append(trans_conn_sim.results)

    # Save sim data
    if save_sim:
        save_sims(multi_model_results, feature_type, cv_type, model_type, use_shared_regions, test_shared_regions, resolution, random_seed)
    
    return multi_model_results


def single_sim_run(feature_type, cv_type, model_type, use_gpu, summary_measure=None, use_shared_regions=False, test_shared_regions=False, resolution=1.0, random_seed=42, save_sim=False, search_method='random'):
    """
    Function to run a simulations for single feature type: connectome only, transcriptome only, connectome+transcriptome
    """
    
    # List to store each model types results
    single_model_results = []

    if feature_type == "conn only":
        # Connectome only 
        conn_sim = Simulation(
                        cv_type=cv_type, model_type=model_type, gpu_acceleration=use_gpu,
                        predict_connectome_from_connectome=True, resolution=resolution,
                        random_seed=random_seed,
                        use_shared_regions=use_shared_regions, include_conn_feats=False,
                        test_shared_regions=test_shared_regions
            )
        conn_sim.run_sim(search_method)
        single_model_results.append(conn_sim.results)

    elif feature_type == "trans only":
        # Transcriptome only 
        trans_sim = Simulation(
                        cv_type=cv_type, model_type=model_type, gpu_acceleration=use_gpu,
                        predict_connectome_from_connectome=False, resolution=resolution, 
                        random_seed=random_seed, use_shared_regions=use_shared_regions,
                        include_conn_feats=False, test_shared_regions=test_shared_regions
            )
        trans_sim.run_sim(search_method)
        single_model_results.append(trans_sim.results)
        
    elif feature_type == "trans plus conn":
        # Connectome and transcriptome
        trans_conn_sim = Simulation(
                    cv_type=cv_type, model_type=model_type, gpu_acceleration=use_gpu,
                    predict_connectome_from_connectome=False, resolution=resolution,
                    random_seed=random_seed, use_shared_regions=use_shared_regions,
                    include_conn_feats=True, test_shared_regions=test_shared_regions
        )
        trans_conn_sim.run_sim(search_method)
        single_model_results.append(trans_conn_sim.results)

    # Save sim data
    if save_sim: 
        save_sims(single_model_results, feature_type, cv_type, model_type, use_shared_regions, test_shared_regions, resolution, random_seed)
    
    return single_model_results

