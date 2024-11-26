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


def extract_simulation_metrics(base_path, subfolder, model_types, cv_types, 
                             feature_types, summary_measures=None, resolutions=None, 
                             random_seeds=None, connectome_target='FC', metric='pearson_corr',
                             use_shared_regions=False, test_shared_regions=False):
    """
    Helper function to extract metrics from simulation pickle files.
    
    Parameters:
    -----------
    base_path : str
        Base path to simulation results directory
    subfolder : str
        Subfolder within sim_results if any
    model_types : str or list
        Model type(s) to extract results for
    cv_types : str or list
        Cross-validation type(s) to extract results for
    feature_types : list
        List of feature types to compare
    summary_measures : list, optional
        List of summary measures to compare
    resolutions : list, optional
        List of resolutions for community CV
    random_seeds : list, optional
        List of random seeds used
    connectome_target : str, optional
        Target connectome type ('FC' or 'SC')
    metric : str, optional
        Metric to extract ('pearson_corr' or 'mse')
    use_shared_regions : bool, optional
        Whether shared regions were used
    test_shared_regions : bool, optional
        Whether shared regions were tested
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing extracted metrics
    """
    # Convert single values to lists for consistent processing
    model_types = [model_types] if isinstance(model_types, str) else model_types
    cv_types = [cv_types] if isinstance(cv_types, str) else cv_types
    
    # Initialize lists to store results
    results_data = []
    
    # Default values if not provided
    if random_seeds is None:
        random_seeds = [42]
    if resolutions is None:
        resolutions = [None]
    if summary_measures is None:
        summary_measures = [None]
        
    # Iterate through all combinations
    for model_type in model_types:
        for cv_type in cv_types:
            for feature_type in feature_types:
                for summary_measure in summary_measures:
                    for resolution in resolutions:
                        for seed in random_seeds:
                            # Build base filename pattern
                            base_pattern = f"{str(feature_type)}"
                            if summary_measure:
                                base_pattern += f"_{summary_measure}"
                            base_pattern += f"_{connectome_target}_{model_type}_{cv_type}"
                            if resolution is not None and cv_type == 'community':
                                base_pattern += str(resolution)
                            base_pattern += f'_{seed}'
                            
                            # Clean the base pattern for regex
                            base_pattern = re.sub(r'[^\w\s_\-]', '', str(base_pattern))
                            
                            # List all files in directory
                            sim_dir = os.path.join(base_path, 'sim/sim_results/', subfolder)
                            matching_files = [f for f in os.listdir(sim_dir) 
                                           if f.startswith(base_pattern) and f.endswith('.pickle')]
                            
                            if not matching_files:
                                print(f"No matching files found for pattern: {base_pattern}")
                                continue
                                
                            # Use the first matching file
                            results_file = matching_files[0]
                            
                            try:
                                # Load results
                                results = open_pickled_results(
                                    results_file,
                                    added_dir=subfolder,
                                    backup=False
                                )
                                
                                # Extract metrics for each fold
                                test_scores = []
                                for fold_results in results[0]:
                                    test_scores.append(fold_results['test_metrics'][metric])
                                
                                # Store results
                                results_data.append({
                                    'Model Type': model_type,
                                    'CV Type': cv_type,
                                    'Feature Type': feature_type,
                                    'Summary Measure': summary_measure,
                                    'Resolution': resolution,
                                    'Seed': seed,
                                    'Mean Score': np.mean(test_scores),
                                    'Std Error': np.std(test_scores) / np.sqrt(len(test_scores))
                                })
                                
                            except FileNotFoundError:
                                print(f"File not found: {results_file}")
                                continue
                            except Exception as e:
                                print(f"Error processing {results_file}: {str(e)}")
                                continue
    
    return pd.DataFrame(results_data)

def plot_performance_comparison(results_df, metric='pearson_corr', group_by='Feature Type',
                              hue='Model Type', style='academic', figsize=(12, 6)):
    """
    Creates a bar plot comparing performance across different configurations.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame containing simulation results from extract_simulation_metrics
    metric : str, optional
        Metric to plot ('pearson_corr' or 'mse')
    group_by : str, optional
        Column to group results by on x-axis
    hue : str, optional
        Column to use for different colored bars
    style : str, optional
        Plot style ('academic' or 'default')
    figsize : tuple, optional
        Figure size
    """
    if style == 'academic':
        plt.style.use('seaborn-whitegrid')
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12
        })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group data and calculate statistics
    grouped_data = results_df.groupby([group_by, hue]).agg({
        'Mean Score': ['mean', 'std', 'count']
    }).reset_index()
    
    # Calculate standard error
    grouped_data['std_err'] = grouped_data[('Mean Score', 'std')] / np.sqrt(grouped_data[('Mean Score', 'count')])
    
    # Set up plot parameters
    unique_hues = grouped_data[hue].unique()
    n_groups = len(grouped_data[group_by].unique())
    n_hues = len(unique_hues)
    width = 0.8 / n_hues
    
    # Create bars
    for i, hue_val in enumerate(unique_hues):
        mask = grouped_data[hue] == hue_val
        x = np.arange(n_groups) + i * width - (n_hues-1) * width/2
        
        bars = ax.bar(x, 
                     grouped_data[mask][('Mean Score', 'mean')],
                     width,
                     yerr=grouped_data[mask]['std_err'],
                     label=hue_val,
                     capsize=5)
        
        # Add value labels on top of bars
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{height:.3f}',
                   ha='center', va='bottom')
    
    # Customize plot
    ax.set_xlabel(group_by)
    ax.set_ylabel('Pearson Correlation' if metric == 'pearson_corr' else 'Mean Squared Error')
    ax.set_title(f'Model Performance Comparison\nGrouped by {group_by}')
    ax.set_xticks(np.arange(n_groups))
    ax.set_xticklabels(grouped_data[group_by].unique(), rotation=45, ha='right')
    
    # Add legend
    ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    summary_stats = grouped_data.pivot_table(
        values=[('Mean Score', 'mean'), 'std_err'],
        index=group_by,
        columns=hue
    )
    print(summary_stats.round(3))

def compare_simulation_results(model_types, cv_types, feature_types, summary_measures=None,
                             resolutions=None, random_seeds=None, connectome_target='FC',
                             metric='pearson_corr', group_by='Feature Type', 
                             style='academic', figsize=(12, 6)):
    """
    High-level function to extract and plot simulation comparison results.
    
    Parameters:
    -----------
    model_types : str or list
        Model type(s) to compare
    cv_types : str or list
        Cross-validation type(s) to compare
    feature_types : list
        List of feature types to compare
    summary_measures : list, optional
        List of summary measures to compare
    resolutions : list, optional
        List of resolutions for community CV
    random_seeds : list, optional
        List of random seeds used
    connectome_target : str, optional
        Target connectome type ('FC' or 'SC')
    metric : str, optional
        Metric to plot ('pearson_corr' or 'mse')
    group_by : str, optional
        Column to group results by on x-axis
    style : str, optional
        Plot style ('academic' or 'default')
    figsize : tuple, optional
        Figure size
    """
    # Extract metrics
    results_df = extract_simulation_metrics(
        base_path=os.getcwd(),
        subfolder='',
        model_types=model_types,
        cv_types=cv_types,
        feature_types=feature_types,
        summary_measures=summary_measures,
        resolutions=resolutions,
        random_seeds=random_seeds,
        connectome_target=connectome_target,
        metric=metric
    )
    
    # Create plot
    if not results_df.empty:
        plot_performance_comparison(
            results_df,
            metric=metric,
            group_by=group_by,
            style=style,
            figsize=figsize
        )
    else:
        print("No results found for the specified parameters.")

def get_sim_performance(feature_type, cv_type, model_type, metric='pearson_corr',
                       connectome_target='FC', summary_measure=None, resolution=None, 
                       random_seed=42, use_shared_regions=False, test_shared_regions=False):
    """
    Get mean train and test performance metrics for a single simulation configuration.
    
    Parameters:
    -----------
    feature_type : str or list
        Type of features used in simulation (e.g., ['transcriptome', 'structural_spectralA'])
    cv_type : str
        Type of cross-validation used (e.g., 'random', 'community')
    model_type : str
        Type of model used (e.g., 'xgboost', 'mlp')
    metric : str, optional
        Metric to extract ('pearson_corr' or 'mse')
    connectome_target : str, optional
        Target connectome type ('FC' or 'SC')
    summary_measure : str, optional
        Summary measure used in simulation (e.g., '10' for spectral embeddings)
    resolution : float, optional
        Resolution parameter for community CV
    random_seed : int, optional
        Random seed used in simulation
    use_shared_regions : bool, optional
        Whether shared regions were used
    test_shared_regions : bool, optional
        Whether shared regions were tested
    
    Returns:
    --------
    dict
        Dictionary containing mean and std error for both train and test performance
        Keys: 'train_mean', 'train_stderr', 'test_mean', 'test_stderr'
    """
    # Build base filename pattern
    base_pattern = f"{str(feature_type)}"
    if summary_measure:
        base_pattern += f"_{summary_measure}"
    base_pattern += f"_{connectome_target}_{model_type}_{cv_type}"
    if resolution is not None and cv_type == 'community':
        base_pattern += str(resolution)
    base_pattern += f'_{random_seed}'
    
    # Clean the pattern
    base_pattern = re.sub(r'[^\w\s_\-]', '', str(base_pattern))
    
    # List all files in directory
    sim_dir = os.path.join(os.getcwd(), 'sim/sim_results/')
    matching_files = [f for f in os.listdir(sim_dir) 
                     if f.startswith(base_pattern) and f.endswith('.pickle')]
    
    if not matching_files:
        raise FileNotFoundError(f"No matching files found for pattern: {base_pattern}")
    
    # Use the first matching file
    try:
        results = open_pickled_results(matching_files[0])
        
        # Extract metrics for each fold
        train_scores = []
        test_scores = []
        
        for fold_results in results[0]:
            train_scores.append(fold_results['train_metrics'][metric])
            test_scores.append(fold_results['test_metrics'][metric])
        
        # Calculate statistics
        performance_stats = {
            'train_mean': np.mean(train_scores),
            'train_stderr': np.std(train_scores) / np.sqrt(len(train_scores)),
            'test_mean': np.mean(test_scores),
            'test_stderr': np.std(test_scores) / np.sqrt(len(test_scores))
        }
        
        return performance_stats
        
    except Exception as e:
        print(f"Error processing {matching_files[0]}: {str(e)}")
        return None

def get_aggregate_performance(feature_type, cv_type, model_type, resolutions, random_seeds,
                            metric='pearson_corr', connectome_target='FC', summary_measure=None,
                            use_shared_regions=False, test_shared_regions=False):
    """
    Get aggregate performance metrics across multiple resolutions and seeds.
    
    Parameters:
    -----------
    feature_type : str or list
        Type of features used in simulation (e.g., ['transcriptome', 'structural_spectralA'])
    cv_type : str
        Type of cross-validation used (e.g., 'random', 'community')
    model_type : str
        Type of model used (e.g., 'xgboost', 'mlp')
    resolutions : list
        List of resolutions to aggregate over (e.g., [1.01, 1.02])
    random_seeds : list
        List of random seeds to aggregate over (e.g., [1, 2, 42])
    metric : str, optional
        Metric to extract ('pearson_corr' or 'mse')
    connectome_target : str, optional
        Target connectome type ('FC' or 'SC')
    summary_measure : str, optional
        Summary measure used in simulation (e.g., '10' for spectral embeddings)
    use_shared_regions : bool, optional
        Whether shared regions were used
    test_shared_regions : bool, optional
        Whether shared regions were tested
    
    Returns:
    --------
    dict
        Dictionary containing aggregated statistics:
        - 'train_mean': Mean training performance across all runs
        - 'train_stderr': Standard error of training performance
        - 'test_mean': Mean test performance across all runs
        - 'test_stderr': Standard error of test performance
        - 'n_runs': Number of successful runs
        - 'failed_runs': List of (resolution, seed) pairs that failed
    """
    # Lists to store all scores
    all_train_means = []
    all_test_means = []
    failed_runs = []
    
    # Iterate through all combinations
    for resolution in resolutions:
        for seed in random_seeds:
            try:
                stats = get_sim_performance(
                    feature_type=feature_type,
                    cv_type=cv_type,
                    model_type=model_type,
                    metric=metric,
                    connectome_target=connectome_target,
                    summary_measure=summary_measure,
                    resolution=resolution,
                    random_seed=seed,
                    use_shared_regions=use_shared_regions,
                    test_shared_regions=test_shared_regions
                )
                
                if stats:
                    all_train_means.append(stats['train_mean'])
                    all_test_means.append(stats['test_mean'])
                else:
                    failed_runs.append((resolution, seed))
                    
            except FileNotFoundError:
                failed_runs.append((resolution, seed))
                continue
            except Exception as e:
                print(f"Error processing resolution {resolution}, seed {seed}: {str(e)}")
                failed_runs.append((resolution, seed))
                continue
    
    # Calculate aggregate statistics
    n_successful = len(all_train_means)
    
    if n_successful == 0:
        print("No successful runs found")
        return None
    
    aggregate_stats = {
        'train_mean': np.mean(all_train_means),
        'train_stderr': np.std(all_train_means) / np.sqrt(n_successful),
        'test_mean': np.mean(all_test_means),
        'test_stderr': np.std(all_test_means) / np.sqrt(n_successful),
        'n_runs': n_successful,
        'failed_runs': failed_runs
    }
    
    # Print summary
    print(f"\nSummary for {model_type} model with {feature_type}:")
    print(f"Successfully processed {n_successful} out of {len(resolutions) * len(random_seeds)} runs")
    if failed_runs:
        print(f"Failed runs (resolution, seed): {failed_runs}")
    
    return aggregate_stats

def plot_model_feature_comparison(feature_types, model_types, cv_type, resolutions, random_seeds,
                                metric='pearson_corr', connectome_target='FC', use_grayscale=True):
    """
    Create a bar plot comparing model performance across different feature types.
    """
    # Store results for plotting
    results = []
    
    # Feature type mapping for display
    feature_display_names = {
        'transcriptome': 'Gene Expression',
        'structural_spectralA_10': 'Structural',
        'transcriptome structural_spectralA_10': 'Gene Expression\n+Structural'
    }
    
    # Get performance for each feature-model combination
    for feature in feature_types:
        for model in model_types:
            stats = get_aggregate_performance(
                feature_type=feature,
                cv_type=cv_type,
                model_type=model,
                resolutions=resolutions,
                random_seeds=random_seeds,
                metric=metric,
                connectome_target=connectome_target
            )
            
            if stats:
                results.append({
                    'Feature Type': feature_display_names.get(feature, feature),
                    'Model': model.upper(),
                    'Mean': stats['test_mean'],
                    'Std Err': stats['test_stderr']
                })
    
    if not results:
        print("No results found for the specified parameters")
        return
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(results)
    
    # Set up plot with narrower width
    plt.figure(figsize=(10, 5))
    
    # Calculate bar positions
    n_features = len(feature_types)
    n_models = len(model_types)
    bar_width = 0.2
    x = np.arange(n_features)
    
    # Set colors
    if use_grayscale:
        colors = ['#737373', '#bdbdbd', '#d9d9d9', '#525252'][:n_models]
    else:
        colors = sns.color_palette('pastel', n_models)
    
    # Create bars for each model
    for i, (model, color) in enumerate(zip(model_types, colors)):
        model_data = df[df['Model'] == model.upper()]
        
        plt.bar(x + i*bar_width - (n_models-1)*bar_width/2, 
               model_data['Mean'],
               bar_width,
               yerr=model_data['Std Err'],
               label=model.upper(),
               color=color,
               capsize=5,
               edgecolor='black')
    
    # Customize plot
    plt.xticks(x, [feature_display_names.get(f, f) for f in feature_types], 
               fontsize=18)
    plt.yticks(fontsize=16)
    
    # Set y-axis label and limits, and determine metric name for title
    if metric == 'pearson_corr':
        plt.ylabel('Pearson r', fontsize=20)
        plt.ylim(0.0, 0.7)
        metric_name = 'Pearson correlation'
    else:
        plt.ylabel('MSE', fontsize=20)
        plt.ylim(0.0, 0.15)
        metric_name = 'MSE'
    
    # Add dynamic title
    plt.title(f'Test performance ({metric_name}) over 5 community splits', 
             fontsize=20, 
             pad=20)
    
    # Remove gridlines
    plt.grid(False)
    
    # Move legend outside the plot
    plt.legend(title='Model Type', 
              fontsize=16, 
              title_fontsize=16, 
              loc='upper left', 
              bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    summary_df = df.pivot_table(
        values=['Mean', 'Std Err'],
        index='Feature Type',
        columns='Model',
        aggfunc='first'
    )
    print(summary_df.round(3))

