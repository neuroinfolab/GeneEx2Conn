# imports
from env.imports import *
import inspect

# metric classes
from models.metrics.eval import (
    pearson_numpy,
    pearson_cupy,
    mse_cupy,
    mse_numpy,
    r2_cupy, 
    r2_numpy
)


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


def validate_inputs(
    features: list = None,
    feature_dict: dict = None,
    cv_type: str = None,
    model_type: str = None,
    connectome_target: str = None,
    **kwargs
) -> None:
    """
    Flexible input validation for Gene2Conn parameters.
    Only validates parameters that are passed in.
    
    Args:
        features: List of feature names to validate
        feature_dict: Dictionary mapping feature names to their data arrays
        cv_type: Type of cross-validation ('random', 'community', 'schaefer')
        model_type: Type of model 
        connectome_target: Target connectome type ('FC', 'SC')
        **kwargs: Additional parameters to validate as needed
        
    Raises:
        ValueError: If any parameters are invalid
    """
    # Validate features if provided
    if features is not None and feature_dict is not None:
        for feature in features:
            if 'spectral' in feature:
                # Extract base feature type and number of components
                feature_parts = feature.split('_')
                feature_type = '_'.join(feature_parts[:-1])
                try:
                    num_latents = int(feature_parts[-1])
                except ValueError:
                    raise ValueError(f"Invalid spectral feature format: {feature}. Must end with integer number of components.")
                
                if feature_type not in feature_dict:
                    raise ValueError(f"Unknown feature type: {feature_type}")
                
                # Check if requested components are within bounds
                feature_dim = feature_dict[feature_type].shape[1]
                if abs(num_latents) > feature_dim:
                    raise ValueError(f"Requested {abs(num_latents)} components for {feature_type} but only {feature_dim} components available")
            else:
                if feature not in feature_dict:
                    raise ValueError(f"Unknown feature type: {feature}")
    
    # Validate CV type if provided
    if cv_type is not None:
        valid_cv_types = {'random', 'community', 'schaefer', 'spatial'}
        if cv_type not in valid_cv_types:
            raise ValueError(f"Invalid cv_type: {cv_type}. Must be one of {valid_cv_types}")
    
    # Validate model type if provided
    if model_type is not None:
        valid_model_types = {'linear', 'ridge', 'pls', 'xgboost', 'dynamic_mlp', 'random_forest', 
                             'bilinear_lowrank', 'bilinear_SCM', 
                             'shared_mlp_encoder', 'shared_linear_encoder', 'shared_transformer'}
        if model_type not in valid_model_types:
            raise ValueError(f"Invalid model_type: {model_type}. Must be one of {valid_model_types}")
    
    # Validate connectome target if provided
    if connectome_target is not None:
        valid_targets = {'FC', 'SC'}
        if connectome_target.upper() not in valid_targets:
            raise ValueError(f"Invalid connectome target: {connectome_target}. Must be one of {valid_targets}")


def load_sweep_config(file_path, input_dim, include_coords=None):
    """
    Load a sweep config file and update the input_dim parameter.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    config['parameters']['input_dim']['value'] = input_dim

    ### Need to update this include coords logic
    if include_coords:
        config['parameters']['include_coords']['value'] = True
    
    return config


def load_best_parameters(yaml_file_path, input_dim, include_coords=None):
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract the best_parameters section
    best_parameters = config.get('best_parameters', {})
    
    # Convert the nested structure to a flat dictionary
    best_config = {key: value['values'][0] if isinstance(value, dict) and 'values' in value else value
                   for key, value in best_parameters.items()}
    
    # override dynamic vars
    best_config['input_dim'] = input_dim

    if include_coords is True:
        best_config = None
        best_parameters = config.get('best_parameters_euclid', {})
        best_config = {key: value['values'][0] if isinstance(value, dict) and 'values' in value else value
                   for key, value in best_parameters.items()}
        best_config['input_dim'] = input_dim
        print('include_coords', include_coords)
    elif include_coords is False:
        best_config['include_coords'] = include_coords
    
    return best_config

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


def get_scorer(metric, gpu_acceleration):
    """
    Helper function to return the appropriate scorer based on the metric and GPU acceleration.
    
    Args:
    metric (str): The metric to use for scoring. Options: 'mse', 'pearson', 'r2'
    gpu_acceleration (bool): Whether GPU acceleration is being used
    
    Returns:
    scorer: A sklearn-compatible scorer object
    """
    if gpu_acceleration:
        if metric == 'mse':
            return make_scorer(mse_cupy, greater_is_better=False)
        elif metric == 'pearson':
            return make_scorer(pearson_cupy, greater_is_better=True)
        elif metric == 'r2':
            return make_scorer(r2_cupy, greater_is_better=True)
    else:
        if metric == 'mse':
            return 'neg_mean_squared_error'
        elif metric == 'pearson':
            return make_scorer(pearson_numpy, greater_is_better=True)
        elif metric == 'r2':
            return 'r2'
    
    raise ValueError(f"Unsupported metric: {metric}")


def grid_search_init(gpu_acceleration, model, X_combined, Y_combined, param_grid, train_test_indices, metric='mse'):
    """
    Helper function to initialize gridsearch object based on if GPU acceleration is being used
    """
    scorer = get_scorer(metric, gpu_acceleration)

    if gpu_acceleration:
        X_combined = cp.array(X_combined)
        Y_combined = cp.array(Y_combined)
        grid_search = GridSearchCV(model.get_model(), 
                                   param_grid, 
                                   cv=train_test_indices, 
                                   scoring=scorer, 
                                   verbose=3,
                                   return_train_score=True,
                                   error_score='raise',
                                   refit=False)
    else:
        grid_search = GridSearchCV(model.get_model(), 
                                   param_grid, 
                                   cv=train_test_indices, 
                                   scoring=scorer, 
                                   verbose=3, 
                                   refit=False)
        
    return grid_search, X_combined, Y_combined


def random_search_init(gpu_acceleration, model, X_combined, Y_combined, param_distributions, train_test_indices, n_iter=100, metric='mse'):
    """
    Helper function to initialize RandomizedSearchCV object based on if GPU acceleration is being used.
    """
    scorer = get_scorer(metric, gpu_acceleration)

    if gpu_acceleration:
        X_combined = cp.array(X_combined)
        Y_combined = cp.array(Y_combined)
        random_search = RandomizedSearchCV(model.get_model(), 
                                           param_distributions, 
                                           n_iter=n_iter, 
                                           cv=train_test_indices, 
                                           scoring=scorer, 
                                           verbose=3, 
                                           refit=False,
                                           n_jobs=1,
                                           random_state=42)
    else:
        random_search = RandomizedSearchCV(model.get_model(), 
                                           param_distributions, 
                                           n_iter=n_iter, 
                                           cv=train_test_indices, 
                                           scoring=scorer, 
                                           verbose=3, 
                                           refit=False, 
                                           n_jobs=-1,
                                           random_state=42)
        
    return random_search, X_combined, Y_combined


def bayes_search_init(gpu_acceleration, model, X_combined, Y_combined, search_space, train_test_indices, n_iter=10, metric='mse'):
    """
    Helper function to initialize BayesSearchCV object based on if GPU acceleration is being used
    """
    scorer = get_scorer(metric, gpu_acceleration)

    if gpu_acceleration:
        print('ACCELERATING')
        X_combined = cp.array(X_combined)
        Y_combined = cp.array(Y_combined)
        error_score = 0.0
        bayes_search = BayesSearchCV(
            model.get_model(),
            search_space,
            n_iter=n_iter, 
            n_points=10,
            cv=train_test_indices,
            scoring=scorer,
            verbose=3,
            random_state=42,
            refit=False, 
            return_train_score=True, # should optimize on test score 
            error_score=error_score,
            optimizer_kwargs={'base_estimator': 'GP', 'acq_func': 'PI'}  # Use Expected Improvement for more exploitation
        )
        #print(bayes_search.optimizer_kwargs)
    else:
        bayes_search = BayesSearchCV(
            model.get_model(),
            search_space,
            n_iter=n_iter,
            cv=train_test_indices,
            scoring=scorer,
            verbose=3,
            refit=False,
            n_jobs=-1,
            random_state=42
        )
        
    return bayes_search, X_combined, Y_combined


def find_best_params(grid_search_cv_results):
    """
    Helper function to score custom gridsearch, works for grid and random with random seed.
    Computes parameter combo that ranked the highest over all possible folds for a given cv split.
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


def extract_model_params(model):
    """
    Dynamically extracts hyperparameters from a PyTorch model by analyzing
    the __init__() method of its class.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        dict: A dictionary of hyperparameters used to initialize the model.
    """
    params = {}

    # Get the class of the model
    model_class = model.__class__

    # Retrieve the __init__ method signature
    init_signature = inspect.signature(model_class.__init__)
    
    # Get parameter names from the __init__ function (excluding self)
    param_names = list(init_signature.parameters.keys())[1:]

    # Extract stored hyperparameter values from the model
    for param in param_names:
        if hasattr(model, param):  # Ensure the attribute exists in self
            params[param] = getattr(model, param)

    return params


def train_sweep(config, model_type, feature_type, connectome_target, cv_type, outer_fold_idx, inner_fold_splits, device, sweep_id, model_classes, parcellation, hemisphere, omit_subcortical, gene_list, seed):
    """
    Training function for W&B sweeps for deep learning models.
    
    Args:
        config: W&B sweep configuration
        model_type: Type of deep learning model
        feature_type: List of feature dictionaries
        connectome_target: Target connectome type
        cv_type: Type of cross-validation
        outer_fold_idx: Current outer fold index
        inner_fold_splits: List of inner fold data splits
        device: torch device (cuda/cpu)
        sweep_id: Current W&B sweep ID
    
    Returns:
        float: Mean validation loss across inner folds
    """
    feature_str = "+".join(str(k) if v is None else f"{k}_{v}"
                         for feat in feature_type 
                         for k,v in feat.items())
    run_name = f"{model_type}_{feature_str}_{connectome_target}_{cv_type}_fold{outer_fold_idx}_innerCV" 

    run = wandb.init(
        project="gx2conn",
        name=run_name,
        group=f"sweep_{sweep_id}",
        tags=["inner cross validation", f'cv_type_{cv_type}', f"fold{outer_fold_idx}", f"model_{model_type}", f"split_{cv_type}{seed}", f'feature_type_{feature_str}', f'target_{connectome_target}', f"parcellation_{parcellation}",  f"hemisphere_{hemisphere}", f"omit_subcortical_{omit_subcortical}", f"gene_list_{gene_list}"],
        reinit=True
    )

    sweep_config = wandb.config
    inner_fold_metrics = {
        'train_losses': [], 'val_losses': [], 
        'train_pearsons': [], 'val_pearsons': []
    }

    # Get the appropriate model class
    if model_type not in model_classes:
        raise ValueError(f"Model type {model_type} not supported for W&B sweeps")
    
    ModelClass = model_classes[model_type]

    # Process each inner fold
    for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(inner_fold_splits):
        print(f'Processing inner fold {fold_idx}')
        
        # Initialize model dynamically based on sweep config
        model = ModelClass(**sweep_config).to(device)

        # Train model
        history = model.fit(X_train, y_train, X_val, y_val)
        
        # Log epoch-wise metrics
        for epoch, metrics in enumerate(zip(history['train_loss'], history['val_loss'], 
                                          history['train_pearson'], history['val_pearson'])):
            wandb.log({
                'inner_fold': fold_idx,
                f'fold{fold_idx}_epoch': epoch,
                f'fold{fold_idx}_train_loss': metrics[0],
                f'fold{fold_idx}_val_loss': metrics[1],
                f'fold{fold_idx}_train_pearson': metrics[2],
                f'fold{fold_idx}_val_pearson': metrics[3]
            })
        
        # Store final metrics
        inner_fold_metrics['train_losses'].append(history['train_loss'][-1])
        inner_fold_metrics['val_losses'].append(history['val_loss'][-1])
        inner_fold_metrics['train_pearsons'].append(history['train_pearson'][-1])
        inner_fold_metrics['val_pearsons'].append(history['val_pearson'][-1])

    # Log mean metrics across folds
    mean_metrics = {
        'mean_train_loss': np.mean(inner_fold_metrics['train_losses']),
        'mean_val_loss': np.mean(inner_fold_metrics['val_losses']),
        'mean_train_pearson': np.mean(inner_fold_metrics['train_pearsons']),
        'mean_val_pearson': np.mean(inner_fold_metrics['val_pearsons'])
    }
    wandb.log(mean_metrics)
    
    run.finish()
    return mean_metrics['mean_val_loss']


def log_wandb_metrics(feature_type, model_type, connectome_target, cv_type, fold_idx, train_metrics, test_metrics, best_val_score, best_model, train_history, model_classes, parcellation, hemisphere, omit_subcortical, gene_list, seed):
    """
    Log metrics to Weights & Biases for tracking experiments.
    
    Args:
        feature_type (list): List of feature dictionaries
        model_type (str): Type of model being used
        connectome_target (str): Target connectome type
        cv_type (str): Type of cross validation
        fold_idx (int): Current fold index
        train_history (dict): Training history for epoch-based models
        train_metrics (dict): Final training metrics
        test_metrics (dict): Final test metrics 
        best_val_score (float): Best validation score
        best_model: Trained model object
    """
    # Create feature string for run name
    feature_str = "+".join(str(k) if v is None else f"{k}_{v}" 
                        for feat in feature_type 
                        for k,v in feat.items())
    run_name = f"{model_type}_{feature_str}_{connectome_target}_{cv_type}_fold{fold_idx}_final_eval" # add in parcellation, hemisphere, subcortex, gene list

    # Initialize a new, standalone W&B run for final evaluation results
    final_eval_run = wandb.init(
        project="gx2conn",
        name=run_name,
        tags=["final_eval", f'cv_type_{cv_type}', f'outerfold_{fold_idx}',  f'model_{model_type}',  f"split_{cv_type}{seed}", f'feature_type_{feature_str}', f'target_{connectome_target}', f'parcellation_{parcellation}', f'hemisphere_{hemisphere}', f'omit_subcortical_{omit_subcortical}', f'gene_list_{gene_list}'],
        reinit=True
    )

    if model_type in model_classes:
        wandb.watch(best_model, log='all')
        for epoch, (train_loss, val_loss, train_pearson, val_pearson) in enumerate(zip(train_history['train_loss'], train_history['val_loss'], train_history['train_pearson'], train_history['val_pearson'])):
            wandb.log({'train_mse_loss': train_loss, 'train_pearson': train_pearson, 'test_mse_loss': val_loss, 'test_pearson': val_pearson})

    wandb.log({
        'final_train_metrics': train_metrics, 'final_test_metrics': test_metrics, 'best_val_loss': best_val_score, 'config': best_model.get_params() if hasattr(best_model, 'get_params') else extract_model_params(best_model)
    })
    
    final_eval_run.finish()
    wandb.finish()
    print("Final evaluation metrics logged successfully.")


def extract_feature_importances(model_type, best_model, save_model_json=False):
    """
    Extract feature importances and model JSON from trained models.
    
    Args:
        model_type (str): Type of model ('pls', 'xgboost', 'ridge', etc.)
        best_model: Trained model object
        save_model_json (bool): Whether to save XGBoost model as JSON
        
    Returns:
        tuple: (feature_importances, model_json)
            - feature_importances: Array of feature importance values or None
            - model_json: Model JSON string for XGBoost or None
    """
    model_json = None
    feature_importances = None
    
    if model_type == 'pls':
        feature_importances = best_model.x_weights_[:, 0]  # Weights for the first component
    elif model_type == 'xgboost':
        feature_importances = best_model.feature_importances_
        if save_model_json:
            booster = best_model.get_booster()
            model_json = booster.save_raw("json").decode("utf-8")
    elif model_type == 'ridge':
        feature_importances = best_model.coef_
        
    return feature_importances, model_json



