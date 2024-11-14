# Gene2Conn/sim/plot.py

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
from models.base_models import ModelBuild
import models.base_models
importlib.reload(models.base_models)

# metric classes
from models.metrics.eval import (
    ModelEvaluator,
    pearson_numpy,
    pearson_cupy,
    mse_cupy,
    r2_cupy
)
import models.metrics.eval
importlib.reload(models.metrics.eval)


import sim.sim_run
from sim.sim_run import single_sim_run, open_pickled_results



def plot_single_model_predictions_with_metrics(single_model_results):
    """
    Function to plot ground truth and predictions of a single model type with key test metrics for each fold.
    
    Parameters:
    single_model_results (list): List of results from a single simulation run.
    """
    for fold_idx, fold_results in enumerate(single_model_results[0]):
        # Ground truth for the fold
        y_true_fold = reconstruct_connectome(fold_results['y_true'])
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(y_true_fold, cmap='viridis') # vmin=0, vmax=1,
        plt.title(f'Fold {fold_idx + 1} Ground Truth')
        plt.colorbar()
        
        # Predicted results
        y_pred_fold = reconstruct_connectome(fold_results['y_pred'])
        plt.subplot(1, 2, 2)
        plt.imshow(y_pred_fold, cmap='viridis', vmax=np.max(y_true_fold)) # vmin=0, vmax=1,
        plt.title(f'Fold {fold_idx + 1} Prediction')
        plt.colorbar()
        
        # Extract key test metrics
        test_metrics = fold_results['test_metrics']
        metrics_text = f"Pearson Corr: {test_metrics['pearson_corr']:.3f}\n"
        metrics_text += f"Conn Corr: {test_metrics.get('connectome_corr', 'N/A'):.3f}\n"
        metrics_text += f"R²: {test_metrics.get('connectome_r2', 'N/A'):.3f}\n"
        metrics_text += f"Geodesic: {test_metrics.get('geodesic_distance', 'N/A'):.3f}\n"
        metrics_text += f"MSE: {test_metrics['mse']:.3f}"
        
        # Display the metrics below the plot
        plt.gca().text(0.5, -0.2, metrics_text, ha='center', va='top', transform=plt.gca().transAxes, fontsize=14)
        
        plt.tight_layout()
        plt.show()


def plot_fold_performance(feature_type, cv_type, model_type, metric='pearson_corr', summary_measure=None, 
                         connectome_target='FC', search_method=('bayes', 'mse'), random_seed=42, 
                         use_shared_regions=False, test_shared_regions=False, backup=False, resolution=None):
    """
    Creates a bar plot comparing train and test performance across folds for a single simulation run.
    
    Parameters:
    -----------
    feature_type : str or list
        Type of features used in the simulation (e.g., ['structural', 'euclidean'])
    cv_type : str
        Type of cross-validation used (e.g., 'random', 'community')
    model_type : str
        Type of model used (e.g., 'xgboost', 'ridge')
    metric : str, optional
        Metric to plot ('pearson_corr', 'mse', 'r2', etc.)
    summary_measure : str, optional
        Summary measure used in simulation
    connectome_target : str, optional
        Target connectome type ('FC' or 'SC')
    search_method : tuple, optional
        Search method used ('bayes', 'grid', etc.) and optimization metric
    random_seed : int, optional
        Random seed used in simulation
    use_shared_regions : bool, optional
        Whether shared regions were used
    test_shared_regions : bool, optional
        Whether shared regions were tested
    backup : bool, optional
        Whether to load from backup directory
    resolution : float, optional
        Resolution parameter for community cross-validation
    """
    # Build filename based on parameters
    results_file_str = f"{str(feature_type)}"
    if summary_measure:
        results_file_str += f"_{summary_measure}"
    results_file_str += f"_{connectome_target}_{model_type}_{cv_type}"
    if resolution is not None and cv_type == 'community':
        results_file_str += str(resolution)
    results_file_str += '_' + str(random_seed)
    results_file_str += "_" + search_method[0] + "_" + search_method[1] + "_search"
    
    if use_shared_regions: 
        results_file_str += "_useshared"
        if test_shared_regions: 
            results_file_str += "_testshared"
        else: 
            results_file_str += "_trainshared"        
    
    results_file_str = re.sub(r'[^\w\s_]', '', str(results_file_str))
    
    # Load results using existing function
    results = open_pickled_results(results_file_str + '.pickle', backup=backup)
    
    # Extract metrics for each fold
    train_scores = []
    test_scores = []
    fold_labels = []
    
    for fold_idx, fold_results in enumerate(results[0]):
        train_scores.append(fold_results['train_metrics'][metric])
        test_scores.append(fold_results['test_metrics'][metric])
        fold_labels.append(f'Fold {fold_idx + 1}')
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(fold_labels))
    width = 0.35
    
    plt.bar(x - width/2, train_scores, width, label='Train', color='skyblue')
    plt.bar(x + width/2, test_scores, width, label='Test', color='lightcoral')
    
    plt.xlabel('Folds')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'{metric.replace("_", " ").title()} by Fold')
    plt.xticks(x, fold_labels)
    plt.legend()
    
    # Add value labels on top of bars
    for i, score in enumerate(train_scores):
        plt.text(i - width/2, score, f'{score:.3f}', ha='center', va='bottom')
    for i, score in enumerate(test_scores):
        plt.text(i + width/2, score, f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average Train {metric}: {np.mean(train_scores):.3f} ± {np.std(train_scores):.3f}")
    print(f"Average Test {metric}: {np.mean(test_scores):.3f} ± {np.std(test_scores):.3f}")


def plot_summary_measure_comparison(feature_type, cv_type, model_type, summary_measures, metric='pearson_corr',
                                  connectome_target='FC', search_method=('bayes', 'mse'), random_seed=42, 
                                  use_shared_regions=False, test_shared_regions=False, backup=False, resolution=None):
    """
    Creates a plot comparing performance across different summary measures.
    
    Parameters:
    -----------
    feature_type : str or list
        Type of features used in simulations (e.g., 'structural_spectralL')
    cv_type : str
        Type of cross-validation used (e.g., 'random', 'community')
    model_type : str
        Type of model used (e.g., 'xgboost', 'ridge')
    summary_measures : list
        List of summary measures to compare (e.g., ['3', '5', '10'])
    metric : str, optional
        Metric to plot ('pearson_corr', 'mse', 'r2', etc.)
    connectome_target : str, optional
        Target connectome type ('FC' or 'SC')
    search_method : tuple, optional
        Search method used ('bayes', 'grid', etc.) and optimization metric
    random_seed : int, optional
        Random seed used in simulations
    use_shared_regions : bool, optional
        Whether shared regions were used
    test_shared_regions : bool, optional
        Whether shared regions were tested
    backup : bool, optional
        Whether to load from backup directory
    resolution : float, optional
        Resolution parameter for community cross-validation
    """
    # Store results for each summary measure
    train_means = []
    train_stds = []
    test_means = []
    test_stds = []
    
    for summary_measure in summary_measures:
        # Build filename based on parameters
        results_file_str = f"{str(feature_type)}"
        if summary_measure:
            results_file_str += f"_{summary_measure}"
        results_file_str += f"_{connectome_target}_{model_type}_{cv_type}"
        if resolution is not None and cv_type == 'community':
            results_file_str += str(resolution)
        results_file_str += '_' + str(random_seed)
        results_file_str += "_" + search_method[0] + "_" + search_method[1] + "_search"
        
        if use_shared_regions: 
            results_file_str += "_useshared"
            if test_shared_regions: 
                results_file_str += "_testshared"
            else: 
                results_file_str += "_trainshared"        
        
        results_file_str = re.sub(r'[^\w\s_\-]', '', str(results_file_str))
        
        try:
            # Load results using existing function
            results = open_pickled_results(results_file_str + '.pickle', backup=backup)
            
            # Extract metrics for each fold
            train_scores = []
            test_scores = []
            
            for fold_results in results[0]:
                train_scores.append(fold_results['train_metrics'][metric])
                test_scores.append(fold_results['test_metrics'][metric])
            
            # Calculate mean and std
            train_means.append(np.mean(train_scores))
            train_stds.append(np.std(train_scores))
            test_means.append(np.mean(test_scores))
            test_stds.append(np.std(test_scores))
            
        except FileNotFoundError:
            print(f"Results file not found for summary measure: {summary_measure}")
            continue
    
    # Create plot
    plt.figure(figsize=(10, 6))
    x = np.arange(len(summary_measures))
    width = 0.35
    
    plt.bar(x - width/2, train_means, width, label='Train', color='skyblue', 
            yerr=train_stds, capsize=5)
    plt.bar(x + width/2, test_means, width, label='Test', color='lightcoral',
            yerr=test_stds, capsize=5)
    
    plt.xlabel('Number of Components')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Performance Comparison Across Different Numbers of Components\n{feature_type}')
    plt.xticks(x, summary_measures)
    plt.legend()
    
    # Add value labels on top of bars
    for i, (train_mean, test_mean) in enumerate(zip(train_means, test_means)):
        plt.text(i - width/2, train_mean, f'{train_mean:.3f}', 
                ha='center', va='bottom')
        plt.text(i + width/2, test_mean, f'{test_mean:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for i, summary_measure in enumerate(summary_measures):
        print(f"\nComponents: {summary_measure}")
        print(f"Train {metric}: {train_means[i]:.3f} ± {train_stds[i]:.3f}")
        print(f"Test {metric}: {test_means[i]:.3f} ± {test_stds[i]:.3f}")

