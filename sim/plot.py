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



def plot_predictions(multi_model_results):
    """
    Function to plot ground truth and predictions of each model.
    
    Parameters:
    multi_model_results (list): List of results from the multi_sim_run function.
    """
    for fold_idx, fold_results in enumerate(multi_model_results[0]):
        # Ground truth for the fold
        y_true_fold = reconstruct_connectome(fold_results['y_true'])
        
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 4, 1)
        plt.imshow(y_true_fold, vmin=0, vmax=1)
        plt.title(f'Fold {fold_idx + 1} Ground Truth')
        plt.colorbar()
        
        for model_idx, model_results in enumerate(multi_model_results):
            y_pred_fold = reconstruct_connectome(model_results[fold_idx]['y_pred'])
            plt.subplot(1, 4, model_idx + 2)
            plt.imshow(y_pred_fold, vmin=0, vmax=1)
            model_type = ['Connectome Only', 'Transcriptome Only', 'Combined'][model_idx]
            plt.title(f'{model_type} Prediction')
            plt.colorbar()
        
        plt.show()


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


def plot_predictions_with_metrics(multi_model_results):
    """
    Function to plot ground truth and predictions of each model with key test metrics.
    
    Parameters:
    multi_model_results (list): List of results from the multi_sim_run function.
    """
    model_labels = ['Connectome Only', 'Transcriptome Only', 'Combined']
    
    for fold_idx, fold_results in enumerate(multi_model_results[0]):
        # Ground truth for the fold
        y_true_fold = reconstruct_connectome(fold_results['y_true'])
        
        plt.figure(figsize=(24, 6))
        plt.subplot(1, 4, 1)
        plt.imshow(y_true_fold, vmin=0, vmax=1, cmap='viridis')
        plt.title(f'Fold {fold_idx + 1} Ground Truth')
        plt.colorbar()
        
        for model_idx, model_results in enumerate(multi_model_results):
            y_pred_fold = reconstruct_connectome(model_results[fold_idx]['y_pred'])
            plt.subplot(1, 4, model_idx + 2)
            plt.imshow(y_pred_fold, vmin=0, vmax=1, cmap='viridis')
            plt.title(f'{model_labels[model_idx]} Prediction')
            plt.colorbar()
            
            # Extract key test metrics
            test_metrics = model_results[fold_idx]['test_metrics']
            metrics_text = f"Pearson Corr: {test_metrics['pearson_corr']:.4f}\n"
            metrics_text += f"Conn Corr: {test_metrics['connectome_corr']:.4f}\n"
            metrics_text += f"R²: {test_metrics['connectome_r2']:.4f}\n"
            metrics_text += f"Geodesic: {test_metrics['geodesic_distance']:.4f}\n"
            metrics_text += f"MSE: {test_metrics['mse']:.4f}"
            
            # Display the metrics below the plot
            plt.gca().text(0.5, -0.15, metrics_text, ha='center', va='top', transform=plt.gca().transAxes, fontsize=14)
        
        plt.tight_layout()
        plt.show()

def plot_best_params_table(multi_model_results):
    """
    Function to plot a table of best model parameters and their associated correlations.
    
    Parameters:
    multi_model_results (list): List of results from the multi_sim_run function.
    """
    # Define the columns
    columns = ['Held-out Fold', 'Model Type', 'n_estimators', 'max_depth', 'learning_rate', 
               'reg_lambda', 'reg_alpha', 'subsample', 'colsample_bytree', 'gamma', 'min_child_weight', 
               'train_corr', 'test_corr']
    
    # Initialize the data list
    data = []
    
    # Model labels
    model_labels = ['conn-conn', 'trans-conn', 'trans+conn-conn']
    
    # Iterate over folds and models
    for fold_idx, fold_results in enumerate(multi_model_results[0]):
        for model_idx, model_results in enumerate(multi_model_results):
            # Extract model parameters
            model_params = model_results[fold_idx]['model_parameters']
            # Extract training and testing correlations
            train_corr = model_results[fold_idx]['train_metrics']['pearson_corr']
            test_corr = model_results[fold_idx]['test_metrics']['pearson_corr']
            
            # Append the data
            data.append([
                fold_idx + 1, 
                model_labels[model_idx], 
                model_params.get('n_estimators'), 
                model_params.get('max_depth'), 
                model_params.get('learning_rate'), 
                model_params.get('reg_lambda'), 
                model_params.get('reg_alpha'), 
                model_params.get('subsample'), 
                model_params.get('colsample_bytree'), 
                model_params.get('min_child_weight'), 
                train_corr, 
                test_corr
            ])
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Plot the table
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(columns))))
    
    # Set the title
    plt.title('Best Model Parameters and Correlations for Each Fold and Model Type', fontsize=14)
    plt.show()


def plot_best_parameters_table_colored(multi_model_results):
    """
    Function to plot a table of best parameters and their associated correlation for each model and fold.
    
    Parameters:
    multi_model_results (list): List of results from the multi_sim_run function.
    """
    # Initialize lists to store data
    folds = []
    model_types = []
    n_estimators = []
    max_depth = []
    learning_rate = []
    reg_lambda = []
    reg_alpha = []
    subsample = []
    colsample_bytree = []
    min_child_weight = []
    train_corr = []
    test_corr = []

    # Extract the relevant data
    for fold_idx, fold_results in enumerate(multi_model_results[0]):
        for model_idx, model_results in enumerate(multi_model_results):
            params = model_results[fold_idx]['model_parameters']
            folds.append(fold_idx + 1)
            model_types.append(['conn-conn', 'trans-conn', 'trans+conn-conn'][model_idx])
            n_estimators.append(params['n_estimators'])
            max_depth.append(params['max_depth'])
            learning_rate.append(params['learning_rate'])
            reg_lambda.append(params['reg_lambda'])
            reg_alpha.append(params['reg_alpha'])
            subsample.append(params['subsample'])
            colsample_bytree.append(params['colsample_bytree'])
            min_child_weight.append(params['min_child_weight'])
            train_corr.append(round(model_results[fold_idx]['train_metrics']['pearson_corr'], 3))
            test_corr.append(round(model_results[fold_idx]['test_metrics']['pearson_corr'], 3))

    # Create DataFrame
    data = {
        'Held-out Fold': folds,
        'Model Type': model_types,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'reg_lambda': reg_lambda,
        'reg_alpha': reg_alpha,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'min_child_weight': min_child_weight,
        'train_corr': train_corr,
        'test_corr': test_corr
    }
    df = pd.DataFrame(data)

    # Define number of unique folds and model types
    unique_folds = df['Held-out Fold'].nunique()
    unique_model_types = df['Model Type'].nunique()

    # Create color palettes
    fold_colors = sns.color_palette("husl", unique_folds)
    model_colors = sns.color_palette("Greys", unique_model_types)

    # Map folds and model types to colors
    fold_color_map = {fold: fold_colors[i] for i, fold in enumerate(sorted(df['Held-out Fold'].unique()))}
    model_color_map = {model: model_colors[i] for i, model in enumerate(sorted(df['Model Type'].unique()))}

    # Create the plot
    fig, ax = plt.subplots(figsize=(20, 10))  # Adjust the figure size to make columns wider
    ax.axis('tight')
    ax.axis('off')

    # Add a table at the bottom of the axes
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    
    # Set the font size and the background color for the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)  # Increase the font size
    table.scale(1.5, 1.5)  # Scale the table to make columns wider
    
    # Apply color coding
    for i, key in enumerate(df.columns):
        for j in range(len(df)):
            if i == 0:
                table[(j + 1, i)].set_facecolor(fold_color_map[folds[j]])
            elif i == 1:
                table[(j + 1, i)].set_facecolor(model_color_map[model_types[j]])
    
    plt.show()


def barplot_model_performance(multi_model_results):
    """
    Function to plot performance metrics of each model for each fold.
    
    Parameters:
    multi_model_results (list): List of results from the multi_sim_run function.
    """
    metrics = ['mse', 'pearson_corr']  # Currently set for MSE and Pearson Correlation

    num_folds = len(multi_model_results[0])
    model_types = ['Connectome Only', 'Transcriptome Only', 'Combined']

    for metric in metrics:
        plt.figure(figsize=(15, 5 * num_folds))

        for fold_idx in range(num_folds):
            train_values = []
            test_values = []

            for model_results in multi_model_results:
                train_values.append(model_results[fold_idx]['train_metrics'].get(metric, np.nan))
                test_values.append(model_results[fold_idx]['test_metrics'].get(metric, np.nan))

            # Plotting
            x = np.arange(len(model_types))  # the label locations
            width = 0.35  # the width of the bars

            fig, ax = plt.subplots(figsize=(10, 5))
            rects1 = ax.bar(x - width/2, train_values, width, label='Train', color='b', alpha=0.6)
            rects2 = ax.bar(x + width/2, test_values, width, label='Test', color='orange', alpha=0.6)

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} by model type and fold {fold_idx + 1}')
            ax.set_xticks(x)
            ax.set_xticklabels(model_types)
            ax.legend()

            # Attach a text label above each bar in rects, displaying its height.
            def autolabel(rects):
                """Attach a text label above each bar in *rects*, displaying its height."""
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.3f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')

            autolabel(rects1)
            autolabel(rects2)

            fig.tight_layout()
            plt.show()


def boxplot_model_performance(results):
    """
    Function to generate box plots comparing model performance across multiple metrics.
    
    Parameters:
    results (list): List of results from the multi_sim_run function.
    """
    all_metrics = ['mse', 'mae', 'r2', 'pearson_corr', 'connectome_corr', 'connectome_r2', 'geodesic_distance']

    # Loop through each metric and generate a box plot
    for metric in all_metrics:
        metric_present = all(metric in fold['test_metrics'] for model_results in results for fold in model_results)
        
        if metric_present:
            plt.figure(figsize=(12, 8))
            data = [
                [fold['test_metrics'][metric] for fold in model_results] for model_results in results
            ]
            plt.boxplot(data, labels=['Connectome Only', 'Transcriptome Only', 'Combined'])
            plt.title(f'Model Performance Comparison ({metric})')
            plt.ylabel(metric)
            plt.xlabel('Model')
            plt.grid(True)
            plt.show()


def violin_plot_model_performance(results):
    """
    Function to create violin plots for train and test performance of each model type, color-coded by held-out fold.
    
    Parameters:
    results (list): List of results from the multi_sim_run function.
    """
    # Metrics to plot
    metrics = ['mse', 'r2', 'pearson_corr', 'connectome_corr', 'connectome_r2', 'geodesic_distance']
    
    num_folds = len(results[0])
    model_types = ['Connectome Only', 'Transcriptome Only', 'Combined']
    fold_colors = sns.color_palette("husl", num_folds)
    
    for metric in metrics:
        metric_present = all(metric in fold['test_metrics'] for model_results in results for fold in model_results)
        
        if metric_present:
            plt.figure(figsize=(18, 6))
            plt.title(f'Model Performance Comparison ({metric})')
            
            data = []
            for model_idx, model_results in enumerate(results):
                for fold_idx, fold_results in enumerate(model_results):
                    train_value = fold_results['train_metrics'].get(metric, np.nan)
                    test_value = fold_results['test_metrics'].get(metric, np.nan)
                    
                    data.append({
                        'Model': model_types[model_idx],
                        'Type': 'Train',
                        'Value': train_value,
                        'Fold': fold_idx
                    })
                    data.append({
                        'Model': model_types[model_idx],
                        'Type': 'Test',
                        'Value': test_value,
                        'Fold': fold_idx
                    })

            df = pd.DataFrame(data)
            sns.violinplot(x='Model', y='Value', hue='Type', data=df, split=True, inner=None)
            
            for model_idx, model_results in enumerate(results):
                for fold_idx, fold_results in enumerate(model_results):
                    train_value = fold_results['train_metrics'].get(metric, np.nan)
                    test_value = fold_results['test_metrics'].get(metric, np.nan)
                    
                    # Adding train points
                    plt.scatter(model_idx - 0.15, train_value, color=fold_colors[fold_idx], edgecolor='black')
                    # Adding test points
                    plt.scatter(model_idx + 0.15, test_value, color=fold_colors[fold_idx], edgecolor='black')
                    
            # Add legend for folds
            for i in range(num_folds):
                plt.scatter([], [], color=fold_colors[i], label=f'Fold {i+1}')
            plt.legend(title='Held-out Folds', loc='upper right')
            
            plt.ylabel(metric)
            plt.xlabel('Model Type')
            plt.grid(True)
            plt.show()

def plot_transcriptome_performance_bar(multi_sim_results, network_list):
    """
    Function to plot test performance (Pearson correlation) of the transcriptome-based model for each held-out fold.
    
    Parameters:
    multi_sim_results (list): List of results from the multi_sim_run function for Schafer simulations.
    """
    networks = network_list
    
    # Extract test performance (Pearson correlation) for transcriptome-based model
    transcriptome_model_idx = 1  # Assuming the transcriptome-based model is at index 1
    test_performance = [fold_results['test_metrics']['pearson_corr'] for fold_results in multi_sim_results[transcriptome_model_idx]]
    
    # Create a bar plot
    plt.figure(figsize=(14, 8))
    bars = plt.bar(networks, test_performance, color=plt.cm.tab20.colors[:len(networks)])
    
    # Increase font sizes
    plt.xlabel('Held-out Network', fontsize=16)
    plt.ylabel('Pearson Correlation', fontsize=16)
    plt.ylim(0, 1)  # Set the y-axis range to be from 0 to 1
    plt.title('Transcriptome-based Model Test Performance (Pearson Correlation)', fontsize=16)
    
    # Annotate bars with the correlation values
    for bar, corr in zip(bars, test_performance):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(corr, 3), ha='center', va='bottom', fontsize=18)

    # Set the tick parameters
    plt.xticks(fontsize=16)
    #plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_connectome_performance_bar(multi_sim_results, network_list):
    """
    Function to plot test performance (Pearson correlation) of the connectome-based model for each held-out fold.
    
    Parameters:
    multi_sim_results (list): List of results from the multi_sim_run function for Schafer simulations.
    """
    networks = network_list
    
    # Extract test performance (Pearson correlation) for connectome-based model
    connectome_model_idx = 0  # Assuming the connectome-based model is at index 0
    test_performance = [fold_results['test_metrics']['pearson_corr'] for fold_results in multi_sim_results[connectome_model_idx]]
    
    # Create a bar plot
    plt.figure(figsize=(14, 8))
    bars = plt.bar(networks, test_performance, color=plt.cm.tab20.colors[:len(networks)])
    
    # Increase font sizes
    plt.xlabel('Held-out Network', fontsize=16)
    plt.ylabel('Pearson Correlation', fontsize=16)
    plt.ylim(0, 1)  # Set the y-axis range to be from 0 to 1
    plt.title('Connectome-based Model Test Performance (Pearson Correlation)', fontsize=16)
    
    # Annotate bars with the correlation values
    for bar, corr in zip(bars, test_performance):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(corr, 3), ha='center', va='bottom', fontsize=18)

    # Set the tick parameters
    plt.xticks(fontsize=16)
    #plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_combined_performance_bar(multi_sim_results, network_list):
    """
    Function to plot the combined transcriptome, connectome, and combined model test performance (pearson correlation) as a bar plot.
    
    Parameters:
    multi_sim_results (list): List of results from the multi_sim_run function.
    network_list (list): List of networks in the order they were held out.
    """
    # Extract test performance (Pearson correlation) for all three models
    connectome_performance = [fold_results['test_metrics']['pearson_corr'] for fold_results in multi_sim_results[0]]
    transcriptome_performance = [fold_results['test_metrics']['pearson_corr'] for fold_results in multi_sim_results[1]]
    combined_performance = [fold_results['test_metrics']['pearson_corr'] for fold_results in multi_sim_results[2]]

    # Create a bar plot
    x = range(len(network_list))
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 8))

    bars1 = ax.bar([p - width for p in x], connectome_performance, width, label='Connectome', color='orange')
    bars2 = ax.bar(x, transcriptome_performance, width, label='Transcriptome', color='blue')
    bars3 = ax.bar([p + width for p in x], combined_performance, width, label='Combined', color='green')

    # Add labels and title
    ax.set_xlabel('Networks', fontsize=14)
    ax.set_ylabel('Pearson Correlation', fontsize=14)
    ax.set_title('Model Test Performance Comparison', fontsize=16)
    ax.set_ylim(0, 1)  # Set the y-axis range to be from 0 to 1
    ax.set_xticks(x)
    ax.set_xticklabels(network_list, rotation=45, fontsize=14)
    ax.legend(fontsize=12)

    # Add value labels on top of bars
    def add_value_labels(bars):
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom', fontsize=12)

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    plt.tight_layout()
    plt.show()
