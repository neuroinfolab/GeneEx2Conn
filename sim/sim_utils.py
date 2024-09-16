# Gene2Conn/sim/sim_utils.py

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
    mse_cupy,
    mse_numpy,
    r2_cupy, 
    r2_numpy
)
import metrics.eval
importlib.reload(metrics.eval)

def bytes2human(n):
    """
    Convert bytes to a human-readable format.
    """
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if abs(n) >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)
    return "%sB" % n

def print_system_usage():
    """
    Print the current system CPU and RAM usage.
    """
    print(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")
    memory_info = psutil.virtual_memory()
    print(f"RAM Usage: {memory_info.percent}%")
    print(f"Available RAM: {bytes2human(memory_info.available)}")
    print(f"Total RAM: {bytes2human(memory_info.total)}")


def drop_test_network(cv_type, network_dict, value, idx):
        """
        Drop an entry from the dictionary based on the given value and return a new dictionary.
        Helper function for removing schaefer network from dict
        """

        # Create a copy of the original dictionary
        new_dict = network_dict.copy()
    
        if cv_type == 'schaefer': 
            # Identify the keys to remove
            keys_to_remove = [key for key, val in new_dict.items() if val == value]
            
            # Remove the identified keys from the new dictionary
            for key in keys_to_remove:
                del new_dict[key]
        else: 
            new_dict.pop(str(idx))

        return new_dict


def grid_search_init(gpu_acceleration, model, X_combined, Y_combined, param_grid, train_test_indices):
    """
    Helper function to initialize gridsearch object based on if GPU acceleration is being used
    """

    if gpu_acceleration:
        X_combined = cp.array(X_combined)
        Y_combined = cp.array(Y_combined)
        cupy_scorer = make_scorer(pearson_cupy, greater_is_better=True)
        grid_search = GridSearchCV(model.get_model(), 
                                   param_grid, 
                                   cv=train_test_indices, 
                                   scoring=cupy_scorer, 
                                   verbose=2,
                                   refit=False,
                                   #n_jobs=1,
                                   #random_state=42
                                   )
    else:
        grid_search = GridSearchCV(model.get_model(), 
                                   param_grid, 
                                   cv=train_test_indices, 
                                   scoring='neg_mean_squared_error', 
                                   verbose=2, 
                                   refit=False, 
                                   n_jobs=-1, 
                                   #random_state=42
                                   )
        
    return grid_search, X_combined, Y_combined



def random_search_init(gpu_acceleration, model, X_combined, Y_combined, param_distributions, train_test_indices, n_iter=100):
    """
    Helper function to initialize RandomizedSearchCV object based on if GPU acceleration is being used.
    """
    if gpu_acceleration:
        X_combined = cp.array(X_combined)
        Y_combined = cp.array(Y_combined)
        cupy_scorer = make_scorer(pearson_cupy, greater_is_better=True)
        random_search = RandomizedSearchCV(model.get_model(), 
                                           param_distributions, 
                                           n_iter=n_iter, 
                                           cv=train_test_indices, 
                                           scoring=cupy_scorer, 
                                           verbose=2, 
                                           refit=False,
                                           #n_jobs=2,
                                           random_state=42)
    else:
        random_search = RandomizedSearchCV(model.get_model(), 
                                           param_distributions, 
                                           n_iter=n_iter, 
                                           cv=train_test_indices, 
                                           scoring='neg_mean_squared_error', 
                                           verbose=2, 
                                           refit=False, 
                                           n_jobs=-1,
                                           random_state=42)
        
    return random_search, X_combined, Y_combined


def bayes_search_init(gpu_acceleration, model, X_combined, Y_combined, search_space, train_test_indices, n_iter=10):
    """
    Helper function to initialize BayesSearchCV object based on if GPU acceleration is being used
    """

    if gpu_acceleration:
        print('ACCELERATING')
        X_combined = cp.array(X_combined)
        Y_combined = cp.array(Y_combined)
        cupy_scorer = make_scorer(pearson_cupy, greater_is_better=True) # mse directionality needs to be debugged
        error_score = 0.0
        bayes_search = BayesSearchCV(
            model.get_model(),
            search_space,
            n_iter=50, # n_iter
            n_points=5,
            cv=train_test_indices,
            scoring=cupy_scorer,
            verbose=3,
            random_state=42,
            refit=False, 
            return_train_score=False, # should optimize on test score 
            error_score=error_score,
            optimizer_kwargs={'base_estimator': 'GP', 'acq_func': 'PI'}  # Use Expected Improvement for more exploitation
        )
        print(bayes_search.optimizer_kwargs)
    else:
        bayes_search = BayesSearchCV(
            model.get_model(),
            search_space,
            n_iter=n_iter,
            cv=train_test_indices,
            scoring='neg_mean_squared_error',
            verbose=2,
            refit=False,
            n_jobs=-1,
            random_state=42
        )
        
    return bayes_search, X_combined, Y_combined


def find_best_params(grid_search_cv_results):
    """
    Helper function to score custom gridsearch
    """
    
    parameter_combos = grid_search_cv_results[0]['params']
    print('PARAMETER COMBOS', parameter_combos)
    
    gridsearch_rank = []
    for idx, result in enumerate(grid_search_cv_results):
        rank = result['rank_test_score']
        gridsearch_rank.append(rank)

    print('GRIDSEARCH RANK', gridsearch_rank)
    
    rank_sum = np.sum(gridsearch_rank, 0)
    best_params = parameter_combos[np.argmin(rank_sum)]

    print('BEST INNNER PARAMS', best_params)
    
    return best_params
    