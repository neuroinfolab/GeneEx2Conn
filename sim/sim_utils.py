from env.imports import *

from models.metrics.eval import (
    pearson_numpy,
    pearson_cupy,
    mse_cupy,
    mse_numpy,
    r2_cupy, 
    r2_numpy, 
    accuracy_numpy,
    accuracy_cupy,
    logloss_cupy,
    logloss_numpy
)

from data.data_utils import expand_X_symmetric, expand_Y_symmetric

# HELPERS 
def set_seed(seed=42):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Forces deterministic behavior
    torch.backends.cudnn.benchmark = False     # Disables auto-tuning (non-deterministic)

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


# CONFIG LOADING
def load_sweep_config(file_path, input_dim, binarize):
    """
    Load a sweep config file and update the input_dim parameter.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    config['parameters']['input_dim']['value'] = input_dim
    
    if binarize is not None:
        config['parameters']['binarize']['value'] = binarize

    return config

def load_best_parameters(yaml_file_path, input_dim, binarize):
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract the best_parameters section
    best_parameters = config.get('best_parameters', {})
    
    # Convert the nested structure to a flat dictionary
    best_config = {key: value['values'][0] if isinstance(value, dict) and 'values' in value else value
                   for key, value in best_parameters.items()}
    
    best_config['input_dim'] = input_dim    
    
    if binarize is not None:
        best_config['binarize'] = binarize
    
    return best_config


# CROSS-VALIDATION
def drop_test_network(cv_type, network_dict, value, test_fold_idx):
        """
        Drop an entry from the dictionary based on the given value and return a new dictionary.
        Helper function for removing schaefer network from dict
        """
        # Create a copy of the original dictionary
        new_dict = network_dict.copy()
    
        if cv_type == 'schaefer': # REVISIT THIS LOGIC
            # Identify the keys to remove
            keys_to_remove = [key for key, val in new_dict.items() if val == value]
            
            # Remove the identified keys from the new dictionary
            for key in keys_to_remove:
                del new_dict[key]
        else: 
            new_dict.pop(str(test_fold_idx))

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
        elif metric == 'acc':
            return make_scorer(accuracy_cupy, greater_is_better=True)
        elif metric == 'logloss':
            return make_scorer(logloss_cupy, greater_is_better=False)
        else: 
            return None
    else:
        if metric == 'mse':
            return 'neg_mean_squared_error'
        elif metric == 'pearson':
            return make_scorer(pearson_numpy, greater_is_better=True)
        elif metric == 'r2':
            return 'r2'
        elif metric == 'acc':
            return make_scorer(accuracy_numpy, greater_is_better=True)
        elif metric == 'logloss':
            return make_scorer(logloss_numpy, greater_is_better=False)
        else: 
            return None

    raise ValueError(f"Metric {metric} not supported")

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
                             return_train_score=True if gpu_acceleration else False,
                             error_score='raise' if gpu_acceleration else 'raise',
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
                                        n_jobs=1 if gpu_acceleration else -1,
                                        random_state=42)
        
    return random_search, X_combined, Y_combined

def bayes_search_init(gpu_acceleration, model, X_combined, Y_combined, search_space, train_test_indices, n_iter=10, metric='mse'):
    """
    Helper function to initialize BayesSearchCV object based on if GPU acceleration is being used
    """
    scorer = get_scorer(metric, gpu_acceleration)

    if gpu_acceleration:
        X_combined = cp.array(X_combined)
        Y_combined = cp.array(Y_combined)

    bayes_search = BayesSearchCV(
        model.get_model(),
        search_space,
        n_iter=n_iter,
        n_points=10 if gpu_acceleration else None,
        cv=train_test_indices,
        scoring=scorer,
        verbose=3,
        random_state=42,
        refit=False,
        return_train_score=True if gpu_acceleration else False,
        error_score=0.0 if gpu_acceleration else 'raise',
        n_jobs=-1 if not gpu_acceleration else None,
        optimizer_kwargs={'base_estimator': 'GP', 'acq_func': 'PI'} if gpu_acceleration else None
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


# EXTRACT MODEL ATTRIBUTES
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


# WANDB
def train_sweep(config, model_type, feature_type, connectome_target, cv_type, outer_fold_idx, inner_fold_splits, device, sweep_id, model_classes, parcellation, hemisphere, omit_subcortical, gene_list, seed, binarize, null_model):
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
        tags=["inner cross validation", f'cv_type_{cv_type}', f"fold{outer_fold_idx}", f"model_{model_type}", f"split_{cv_type}{seed}", f'feature_type_{feature_str}', f'target_{connectome_target}', f"parcellation_{parcellation}",  f"hemisphere_{hemisphere}", f"omit_subcortical_{omit_subcortical}", f"gene_list_{gene_list}", f"binarize_{binarize}", f"null_model_{null_model}"],
        reinit=True
    )

    sweep_config = wandb.config
    inner_fold_metrics = {
        'train_losses': [], 'val_losses': []
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
        for epoch, metrics in enumerate(zip(history['train_loss'], history['val_loss'])):
            wandb.log({
                'inner_fold': fold_idx,
                f'fold{fold_idx}_epoch': epoch,
                f'fold{fold_idx}_train_loss': metrics[0],
                f'fold{fold_idx}_val_loss': metrics[1]
            })
        
        # Store final metrics
        inner_fold_metrics['train_losses'].append(history['train_loss'][-1])
        inner_fold_metrics['val_losses'].append(history['val_loss'][-1])

    # Log mean metrics across folds
    mean_metrics = {
        'mean_train_loss': np.mean(inner_fold_metrics['train_losses']),
        'mean_val_loss': np.mean(inner_fold_metrics['val_losses'])
    }
    wandb.log(mean_metrics)
    
    run.finish()
    return mean_metrics['mean_val_loss']

def train_sweep_torch(config, model_type, train_indices, feature_type, connectome_target, dataset, cv_type, cv_obj, outer_fold_idx, device, sweep_id, model_classes, parcellation, hemisphere, omit_subcortical, gene_list, seed, binarize, impute_strategy, sort_genes, null_model):
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
    run_name = f"{model_type}_{feature_str}_{connectome_target}_{cv_type}{seed}_fold{outer_fold_idx}_innerCV" 
    run = wandb.init(
        project="gx2conn",
        name=run_name,
        group=f"sweep_{sweep_id}",
        tags=[
            "inner cross validation",
            f'cv_type_{cv_type}',
            f"fold{outer_fold_idx}",
            f"model_{model_type}",
            f"split_{cv_type}{seed}",
            f'feature_type_{feature_str}',
            f'target_{connectome_target}',
            f"parcellation_{parcellation}",
            f"hemisphere_{hemisphere}",
            f"omit_subcortical_{omit_subcortical}",
            f"gene_list_{gene_list}",
            f"binarize_{binarize}",
            f"impute_strategy_{impute_strategy}",
            f"sort_genes_{sort_genes}",
            f"null_model_{null_model}"
        ],
        reinit=True
    )
    sweep_config = wandb.config
    
    inner_fold_metrics = {'final_train_loss': [], 'final_val_loss': [], 
                          'final_train_pearson': [], 'final_val_pearson': []}

    # Get the appropriate model class
    if model_type not in model_classes:
        raise ValueError(f"Model type {model_type} not supported for W&B sweeps")
    
    ModelClass = model_classes[model_type]

    # Process each inner fold
    for fold_idx, (train_indices, test_indices) in enumerate(cv_obj.split()):
        print(f'Processing inner fold {fold_idx}')
        if fold_idx == 0:  # Only run CV on the first inner fold to test more parameters
            train_region_pairs = expand_X_symmetric(np.array(train_indices).reshape(-1, 1)).astype(int)
            test_region_pairs = expand_X_symmetric(np.array(test_indices).reshape(-1, 1)).astype(int)
            train_indices_expanded = np.array([dataset.valid_pair_to_expanded_idx[tuple(pair)] for pair in train_region_pairs])
            test_indices_expanded = np.array([dataset.valid_pair_to_expanded_idx[tuple(pair)] for pair in test_region_pairs])    
    
            # Initialize model dynamically based on sweep config and fit
            if 'pls' in model_type:
                model = ModelClass(**sweep_config, train_indices=train_indices, test_indices=test_indices, region_pair_dataset=dataset).to(device)
            else:
                model = ModelClass(**sweep_config).to(device)
            
            if model_type == 'pls_twostep':
                history = model.fit(dataset, train_indices, test_indices)
                # Store final metrics
                inner_fold_metrics['final_train_loss'].append(history['train_loss'])
                inner_fold_metrics['final_val_loss'].append(history['val_loss'])
                inner_fold_metrics['final_train_pearson'].append(history['train_pearson'])
                inner_fold_metrics['final_val_pearson'].append(history['val_pearson'])
            else: 
                history = model.fit(dataset, train_indices_expanded, test_indices_expanded)
                # Log epoch-wise metrics
                for epoch, metrics in enumerate(zip(history['train_loss'], history['val_loss'])):
                    wandb.log({
                        'inner_fold': fold_idx,
                        f'fold{fold_idx}_epoch': epoch,
                        f'fold{fold_idx}_train_loss': metrics[0],
                        f'fold{fold_idx}_val_loss': metrics[1]
                    })
                
                train_dataset = Subset(dataset, train_indices_expanded)
                train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, pin_memory=True)
                test_dataset = Subset(dataset, test_indices_expanded)
                val_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, pin_memory=True)
            
                predictions, targets = model.predict(train_loader)
                train_pearson = pearsonr(predictions, targets)[0]
                predictions, targets = model.predict(val_loader)
                val_pearson = pearsonr(predictions, targets)[0]
                # Store final metrics
                inner_fold_metrics['final_train_loss'].append(history['train_loss'][-1])
                inner_fold_metrics['final_val_loss'].append(history['val_loss'][-1])
                inner_fold_metrics['final_train_pearson'].append(train_pearson)
                inner_fold_metrics['final_val_pearson'].append(val_pearson)

    # Log mean metrics across folds
    mean_metrics = {
        'mean_train_loss': np.mean(inner_fold_metrics['final_train_loss']),
        'mean_val_loss': np.mean(inner_fold_metrics['final_val_loss']),
        'mean_train_pearson': np.mean(inner_fold_metrics['final_train_pearson']),
        'mean_val_pearson': np.mean(inner_fold_metrics['final_val_pearson'])
    }
    wandb.log(mean_metrics)
    run.finish()
    
    return mean_metrics['mean_val_loss']