from env.imports import *

import importlib
import data
import data.data_utils
from data.data_load import load_transcriptome, load_connectome, load_coords, load_network_labels
from models import *
from data import * 
from sim import *
import models
import models.metrics
from models.metrics import *
import wandb
from datetime import datetime, timedelta
from collections import defaultdict
from matplotlib.patches import Patch
import re

api = wandb.Api()
project_path = "alexander-ratzan-new-york-university/gx2conn"

models = ['cge', 'gaussian_kernel', 'exponential_decay',
          'bilinear_CM', 'pls_bilineardecoder', 'bilinear_lowrank',
          'dynamic_mlp', 'shared_transformer', 'shared_transformer_cls']

model_feature_types = {
    'cge': ['transcriptome_PCA'],
    'gaussian_kernel': ['euclidean'],
    'exponential_decay': ['euclidean'],
    'bilinear_CM': ['transcriptome', 'transcriptome_PCA'],
    'pls_bilineardecoder': ['transcriptome'],
    'bilinear_lowrank': ['transcriptome'],
    'dynamic_mlp': ['transcriptome', 'transcriptome+euclidean'],
    'shared_transformer': ['transcriptome'],
    'shared_transformer_cls': ['transcriptome']
}

model_groups = {
    'rules based': ['cge', 'gaussian_kernel', 'exponential_decay'],
    'bilinear': ['bilinear_CM', 'bilinear_CM_PCA', 'pls_bilineardecoder', 'bilinear_lowrank'],
    'deep learning': ['dynamic_mlp', 'dynamic_mlp_coords', 'shared_transformer', 'shared_transformer_cls']
}

# Weight definitions for weighted averaging by test set size
CV_WEIGHTS = {
    'schaefer': {
        'Vis': 61, 
        'SomMot': 77,
        'DorsAttn': 46, 
        'SalVentAttn': 47,
        'Limbic': 26,
        'Cont': 52,
        'Default': 91,
        'Subcortical': 46,
        'Cerebellum': 9
    },
    'lobe': {
        'Frontal': 132,
        'Parietal': 121,
        'Temporal': 95,
        'Occipital': 52,
        'Subcortex': 46,
        'Cerebellum': 10
    }
}

def date_days_ago(days):
    """Returns the date (YYYY-MM-DD) that was `days` ago from today."""
    from datetime import datetime, timedelta
    return (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

def days_ago_from_date(date_str):
    """Returns how many days ago the given date (YYYY-MM-DD) was from today."""
    from datetime import datetime
    today = datetime.now().date()
    date = datetime.strptime(date_str, '%Y-%m-%d').date()
    return (today - date).days

def weighted_mean_and_se(values, weights):
    """
    Compute weighted mean and standard deviation for a set of values and weights.
    
    Args:
        values (np.ndarray): Array of values
        weights (list): List of weights corresponding to each value
        
    Returns:
        tuple: (weighted_mean, weighted_standard_deviation)
    """
    # Convert to numpy arrays and handle NaN values
    values = np.array(values)
    weights = np.array(weights)[:len(values)]  # Truncate weights to match values length
    
    # Remove NaN values and corresponding weights
    mask = ~np.isnan(values)
    values = values[mask]
    weights = weights[mask]
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Calculate weighted mean
    weighted_mean = np.sum(values * weights)
    
    # Calculate weighted variance
    weighted_var = np.sum(weights * (values - weighted_mean)**2) / (1 - np.sum(weights**2))
    
    # Calculate weighted standard deviation
    weighted_std = np.sqrt(weighted_var)
    
    return weighted_mean, weighted_std

def fetch_and_summarize_wandb_runs(model, cv_type, null_model, feature_type='transcriptome', target='FC', gene_list='0.2', within_last=60, before_last=0, use_weighted=False, exclude='HCP', only_include='UKBB', return_history=False):
    """
    Fetches wandb runs matching specific tags and summarizes their final train/test metrics.
    Handles different CV types with their expected number of runs:
    - random/spatial: 40 runs
    - schaefer: 9 runs  
    - lobe: 6 runs
    
    Args:
        model (str): Model name, e.g., 'bilinear_CM'
        cv_type (str): CV type, one of: 'random', 'spatial', 'schaefer', 'lobe'
        null_model (str): Null model label, e.g., 'none'
        feature_type (str): Feature type, e.g., 'transcriptome_PCA'
        target (str): Target connectome type, e.g., 'FC'
        within_last (int): Search for runs within this many days ago (default: 60)
        before_last (int): Exclude runs from this many days ago (default: 0)
        use_weighted (bool): Whether to compute weighted statistics for schaefer/lobe CV
        exclude (str): Dataset to exclude from search (default: 'HCP')
        return_history (bool): If True, return (summary_df, history_df) tuple
    
    Returns:
        summary_df (pd.DataFrame): DataFrame with mean, std, std of all train/test metrics
                                  If use_weighted=True and cv_type in ['schaefer', 'lobe'], 
                                  includes weighted_mean and weighted_std rows
        history_df (pd.DataFrame): Individual run data (only returned if return_history=True)
    """
    # Set time filters
    end_time = datetime.now() - timedelta(days=before_last)
    start_time = datetime.now() - timedelta(days=within_last)
    
    # Set expected number of runs based on cv_type
    if cv_type == "schaefer":
        expected_runs = 9
    elif cv_type == "lobe":
        expected_runs = 6
    else:  # random or spatial
        expected_runs = 40
    
    filters = {
        "tags": {
            "$all": [
                "final_eval",
                f"model_{model}",
                f"target_{target}",
                f"cv_type_{cv_type}",
                f"gene_list_{gene_list}",
                f"null_model_{null_model}",
                f"feature_type_{feature_type}"
            ],
        },
        "created_at": {
            "$gte": start_time.isoformat(), 
            "$lte": end_time.isoformat()
        },
        "state": "finished"
    }
    
    # Add exclusion filter if specified
    if exclude != "":
        filters["tags"]["$nin"] = [f"dataset_{exclude}"]

    if only_include != "":
        filters["tags"]["$in"] = [f"dataset_{only_include}"]

    print(f"🔍 Fetching runs for: model={model}, cv_type={cv_type}, null_model={null_model}, feature_type={feature_type}")
    runs = api.runs(project_path, filters=filters, order="-created_at")
    
    run_data = []
    for run in runs:
        metrics = {}
        summary = run.summary

        # Attempt to extract the pearson_r value (or set to NaN if missing)
        pearson = summary.get("final_test_metrics", {}).get("pearson_r", np.nan)

        # Only consider runs with meaningful final_test_metrics
        if "final_test_metrics" not in summary:
            continue
        
        for k, v in summary.get('final_train_metrics', {}).items():
            if isinstance(v, (int, float)):
                metrics[f'train_{k}'] = v

        for k, v in summary.get('final_test_metrics', {}).items():
            if isinstance(v, (int, float)):
                metrics[f'test_{k}'] = v

        metrics['run_name'] = run.name
        metrics['run_id'] = run.id
        metrics['final_test_pearson_r'] = pearson
        
        # Extract fold number for weighted calculations
        if use_weighted and cv_type in ['schaefer', 'lobe']:
            fold_match = re.search(r'fold(\d+)', run.name)
            if fold_match:
                metrics['fold'] = int(fold_match.group(1))
        
        run_data.append(metrics)
    
    df = pd.DataFrame(run_data)

    if len(df) < expected_runs:
        raise ValueError(f"❌ Expected {expected_runs} runs, but found {len(df)}.")

    # Handle deduplication based on CV type
    if use_weighted and cv_type in ['schaefer', 'lobe']:
        # Sort by fold number for proper weight assignment
        df = df.sort_values('fold')
        # Deduplicate by fold keeping highest test pearson_r within each fold
        df_unique = (
            df.sort_values(["fold", "final_test_pearson_r"], ascending=[True, False])
              .drop_duplicates("fold", keep="first")
        )
        if len(df_unique) != expected_runs:
            raise ValueError(f"❌ Expected {expected_runs} unique folds, found {len(df_unique)} after deduplication.")
    else:
        # Deduplicate by run_name using highest test pearson_r
        df_unique = (
            df.sort_values("final_test_pearson_r", ascending=False)
              .drop_duplicates("run_name", keep="first")
        )
        if len(df_unique) != expected_runs:
            raise ValueError(f"❌ Expected {expected_runs} unique run names, found {len(df_unique)} after deduplication (from {len(df)} total runs).")

    # Store history before cleaning for aggregation
    history_df = df_unique.copy()
    
    # Clean and summarize
    columns_to_drop = ["run_name", "run_id", "final_test_pearson_r"]
    if 'fold' in df_unique.columns:
        columns_to_drop.append("fold")
    df_clean = df_unique.drop(columns=columns_to_drop, errors="ignore")

    summary_df = pd.DataFrame({
        "mean": df_clean.mean(),
        "std": df_clean.std()
    }).T
    
    # Add weighted statistics if requested and applicable
    if use_weighted and cv_type in ['schaefer', 'lobe']:
        weights = list(CV_WEIGHTS[cv_type].values())
        
        # Calculate weighted statistics for final_test_pearson_r
        weighted_mean, weighted_std = weighted_mean_and_se(df_unique['final_test_pearson_r'].values, weights)
        summary_df.loc['weighted_mean', 'final_test_pearson_r'] = weighted_mean
        summary_df.loc['weighted_std', 'final_test_pearson_r'] = weighted_std
        
        # Calculate weighted statistics for test_pearson_r if it exists
        if 'test_pearson_r' in df_clean.columns:
            test_pearson_values = df_unique['test_pearson_r'].values if 'test_pearson_r' in df_unique.columns else df_clean['test_pearson_r'].values
            weighted_mean_test, weighted_std_test = weighted_mean_and_se(test_pearson_values, weights)
            summary_df.loc['weighted_mean', 'test_pearson_r'] = weighted_mean_test
            summary_df.loc['weighted_std', 'test_pearson_r'] = weighted_std_test

    if return_history:
        return summary_df, history_df
    else:
        return summary_df

def process_model_feature_combinations(cv_type, null_model, models, model_feature_types, summary_dict, use_weighted=False, exclude='HCP', only_include='', within_last=None, before_last=None, time_ranges=None):
    """
    Helper function to process model/feature type combinations and populate summary dictionary.
    
    Args:
        cv_type (str): Cross-validation type
        null_model (str): Null model type
        models (list): List of model names to process
        model_feature_types (dict): Mapping of models to their feature types
        summary_dict (dict): Dictionary to populate with results
        use_weighted (bool): Whether to use weighted statistics
        exclude (str): Dataset to exclude
        only_include (str): Dataset to include only
        within_last (int): Default within_last parameter (fallback)
        before_last (int): Default before_last parameter (fallback)
        time_ranges (dict): Custom time ranges with structure:
            {
                'default': {'within_last': int, 'before_last': int},
                'model_specific': {
                    'model_name': {'within_last': int, 'before_last': int}
                },
                'null_model_specific': {
                    'null_model_name': {'within_last': int, 'before_last': int}
                },
                'model_null_specific': {
                    ('model_name', 'null_model_name'): {'within_last': int, 'before_last': int}
                }
            }
    """
    def get_time_range(model, null_model):
        """Get time range for specific model/null_model combination with fallback hierarchy"""
        if time_ranges is None:
            return within_last, before_last
            
        # Priority 1: model_null_specific (most specific)
        if 'model_null_specific' in time_ranges:
            key = (model, null_model)
            if key in time_ranges['model_null_specific']:
                range_dict = time_ranges['model_null_specific'][key]
                return range_dict.get('within_last'), range_dict.get('before_last')
        
        # Priority 2: model_specific
        if 'model_specific' in time_ranges and model in time_ranges['model_specific']:
            range_dict = time_ranges['model_specific'][model]
            return range_dict.get('within_last'), range_dict.get('before_last')
            
        # Priority 3: null_model_specific  
        if 'null_model_specific' in time_ranges and null_model in time_ranges['null_model_specific']:
            range_dict = time_ranges['null_model_specific'][null_model]
            return range_dict.get('within_last'), range_dict.get('before_last')
            
        # Priority 4: default from time_ranges
        if 'default' in time_ranges:
            range_dict = time_ranges['default']
            return range_dict.get('within_last'), range_dict.get('before_last')
            
        # Priority 5: function parameters (original fallback)
        return within_last, before_last
    
    # Set expected number of runs based on cv_type
    if cv_type == "schaefer":
        expected_runs = 9
    elif cv_type == "lobe":
        expected_runs = 6
    else:  # random or spatial
        expected_runs = 40
        
    weighted_str = " (weighted)" if use_weighted and cv_type in ['schaefer', 'lobe'] else ""
    print(f"Checking which model/feature type combinations return {expected_runs} runs for null_model={null_model}{weighted_str}:\n")
    
    for model in models:
        feature_types = model_feature_types[model]
        for feature_type in feature_types:
            try:
                # Get custom time range for this model/null_model combination
                model_within_last, model_before_last = get_time_range(model, null_model)
                
                df = fetch_and_summarize_wandb_runs(
                    model, cv_type, null_model, feature_type, 
                    use_weighted=use_weighted, exclude=exclude,
                    only_include=only_include,
                    within_last=model_within_last, before_last=model_before_last
                )
                
                # Handle bilinear_CM and dynamic_mlp_coords splits explicitly
                if model == "bilinear_CM" and feature_type == "transcriptome_PCA":
                    summary_dict["bilinear_CM_PCA"] = df
                    print(f"✓ bilinear_CM_PCA: Successfully found {expected_runs} runs")
                elif model == "bilinear_CM" and feature_type == "transcriptome":
                    summary_dict["bilinear_CM"] = df
                    print(f"✓ bilinear_CM: Successfully found {expected_runs} runs")
                elif model == "dynamic_mlp" and feature_type == "transcriptome+euclidean":
                    summary_dict["dynamic_mlp_coords"] = df
                    print(f"✓ dynamic_mlp_coords: Successfully found {expected_runs} runs")
                elif model == "dynamic_mlp" and feature_type == "transcriptome":
                    summary_dict["dynamic_mlp"] = df
                    print(f"✓ dynamic_mlp: Successfully found {expected_runs} runs")
                else:
                    summary_dict[model] = df
                    print(f"✓ {model} with {feature_type}: Successfully found {expected_runs} runs")

            except ValueError as e:
                print(f"✗ {model} with {feature_type}: {str(e)}")
            except Exception as e:
                print(f"! {model} with {feature_type}: Unexpected error: {str(e)}")
                
def plot_model_barchart(summary_dict, metric="test_pearson_r", xlim=(0.1, 0.9), highlight_models=None, highlight_label=None, ascending=False):
    """
    Create a horizontal bar plot of model performance with error bars, ordered by performance.
    
    Args:
        summary_dict: dict of result DataFrames from fetch_and_summarize_wandb_runs
        metric: str, metric column to visualize (e.g., 'test_pearson_r')
        xlim: tuple, x-axis limits
        highlight_models: list of model names to highlight with a different color
        highlight_label: str, label to add next to highlighted models (e.g. "uses pooling")
        ascending: bool, if True sort from worst to best, if False sort from best to worst (default)
    """
    # Set global font size and derived sizes
    base_fontsize = 20
    plt.rcParams.update({'font.size': base_fontsize})
    label_fontsize = base_fontsize * 0.67
    legend_fontsize = base_fontsize * 0.78

    # Flatten model info into DataFrame
    plot_data = []
    for model_key in summary_dict:
        df = summary_dict[model_key]
        if metric in df.columns:
            # Format model name
            display_name = model_key
            if 'shared_transformer' in display_name:
                display_name = display_name.replace('shared_transformer', 'smt')
            if 'dynamic_mlp' in display_name:
                display_name = display_name.replace('dynamic_mlp', 'mlp')
                
            plot_data.append({
                "Model": display_name,
                "Mean": df.loc["mean", metric],
                "Std": df.loc["std", metric],
                "Highlighted": model_key in (highlight_models or [])
            })

    plot_df = pd.DataFrame(plot_data)

    # Sort by performance (ascending=True for worst to best, ascending=False for best to worst)
    plot_df = plot_df.sort_values("Mean", ascending=ascending)

    # Plot
    plt.figure(figsize=(8, 7), dpi=300)
    
    # Create two color palettes
    default_color = sns.color_palette("viridis", as_cmap=True)(0.3)
    highlight_color = sns.color_palette("husl", as_cmap=True)(0.7)
    
    # Plot bars with different colors based on highlight status
    for i, (_, row) in enumerate(plot_df.iterrows()):
        color = highlight_color if row["Highlighted"] else default_color
        plt.barh(i, row["Mean"], height=0.6, color=color)
        
        # Add error bars
        plt.errorbar(
            x=row["Mean"],
            y=i,
            xerr=row["Std"],
            fmt='none',
            ecolor='black',
            capsize=3,
            linewidth=2
        )
        
        # Add mean value text
        plt.text(
            row["Mean"] + 0.01,  # Small offset from bar end
            i,
            f'{row["Mean"]:.3f}',
            va='center',
            fontsize=label_fontsize
        )

    ax = plt.gca()
    ax.set_xlim(xlim)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_xlabel("Pearson-r", fontsize=label_fontsize)
    ax.set_ylabel("")
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["Model"])
    ax.invert_yaxis()  # best at top, worst at bottom

    # Set tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=label_fontsize)

    # Add legend if there are highlighted models
    if highlight_models and highlight_label:
        legend_elements = [
            Patch(facecolor=highlight_color, label=highlight_label)
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=legend_fontsize-2)

    sns.despine()
    plt.tight_layout()
    plt.show()

def plot_true_vs_null_model_barchart_w_legend(
    summary_true_dict,
    summary_null_dict,
    metric="test_pearson_r",
    model_groups=None,
    xlim=(0.1, 0.9),
    overlay_style="alpha"  # or "hatch"
):
    """
    Plot a horizontal bar chart comparing true vs null model performance (e.g., Pearson-r).
    Bars for true performance are solid, and null are translucent or hatched overlays.

    Args:
        summary_true_dict: dict of true result DataFrames (from fetch_and_summarize_wandb_runs)
        summary_null_dict: dict of null result DataFrames
        metric: str, metric column to visualize (e.g., 'test_pearson_r')
        model_groups: dict of grouped models with display names
        xlim: tuple, x-axis limits
        overlay_style: 'alpha' or 'hatch' for null model overlay style
    """
    if model_groups is None:
        model_groups = {
            'Deep Learning': {
                'dynamic_mlp': 'MLP',
                'dynamic_mlp_coords': 'MLP w/ coords',
                'shared_transformer': 'SMT',
                'shared_transformer_cls': 'SMT w/ [CLS]'
            },
            'Bilinear': {
                'bilinear_CM': 'Connectome Model',
                'bilinear_CM_PCA': 'Connectome Model (PCA)',
                'pls_bilineardecoder': 'Connectome Model (PLS)',
                'bilinear_lowrank': 'Connectome Model (LR)'
            },
            'Rules-Based': {
                'cge': 'CGE',
                'gaussian_kernel': 'Gaussian Kernel',
                'exponential_decay': 'Exponential Decay'
            }
        }

    # Set fonts
    base_fontsize = 22
    label_fontsize = base_fontsize * 0.67
    legend_fontsize = base_fontsize * 0.67
    plt.rcParams.update({'font.size': base_fontsize})

    # Flatten model info into DataFrame
    plot_data = []
    for group_name, model_dict in model_groups.items():
        for model_key, display_name in model_dict.items():
            if model_key in summary_true_dict and model_key in summary_null_dict:
                df_true = summary_true_dict[model_key]
                df_null = summary_null_dict[model_key]
                if metric in df_true.columns and metric in df_null.columns:
                    plot_data.append({
                        "Model": display_name,
                        "Group": group_name,
                        "TrueMean": df_true.loc["mean", metric],
                        "TrueStd": df_true.loc["std", metric],
                        "NullMean": df_null.loc["mean", metric],
                        "NullStd": df_null.loc["std", metric]
                    })

    plot_df = pd.DataFrame(plot_data)

    # Sort by descending true performance
    plot_df = plot_df.sort_values("TrueMean", ascending=False)
    plot_df.reset_index(drop=True, inplace=True)

    # Group color map
    unique_groups = list(model_groups.keys())
    palette = sns.color_palette("viridis", n_colors=12, desat=1.0)[2::4]  # Starting from index 3 to get lighter/bluer colors
    group_color_map = {group: color for group, color in zip(unique_groups, palette)}

    # Plot
    plt.figure(figsize=(9, 7), dpi=300)
    ax = plt.gca()

    for i in range(len(plot_df)):
        row = plot_df.iloc[i]
        y = i
        # True bar
        ax.barh(
            y=y,
            width=row["TrueMean"],
            height=0.6,
            color=group_color_map[row["Group"]],
            edgecolor="black",
            zorder=1
        )
        ax.errorbar(
            x=row["TrueMean"],
            y=y,
            xerr=row["TrueStd"],
            fmt='none',
            ecolor='black',
            capsize=1,
            linewidth=1,
            zorder=2
        )
        
        # For top model, add the value as a label to the right of the true bar
        if i == 0:
            ax.text(
                row["TrueMean"],
                y - 0.4,  # shift upward (adjust spacing if needed)
                f"{row['TrueMean']:.2f}",
                va="bottom",
                ha="center",
                fontsize=label_fontsize,
                color="black"
            )

        # Skip null bars for Gaussian Kernel and Exponential Decay
        if row["Model"] not in ["Gaussian Kernel", "Exponential Decay"]:
            # Null bar overlay
            if overlay_style == 'hatch':
                ax.barh(
                    y=y,
                    width=row["NullMean"],
                    height=0.6,
                    left=0,
                    edgecolor="black",
                    facecolor='none',
                    hatch="////",
                    linewidth=1,
                    zorder=3
                )
            elif overlay_style == 'alpha':
                ax.barh(
                    y=y,
                    width=row["NullMean"],
                    height=0.6,
                    left=0,
                    color="lightgray",
                    edgecolor="black",
                    alpha=0.3,
                    zorder=3
                )

            # Null error bars
            ax.errorbar(
                x=row["NullMean"],
                y=y,
                xerr=row["NullStd"],
                fmt='none',
                ecolor='black',
                capsize=1,
                linewidth=1,
                linestyle='--',
                zorder=4
            )

    # Final plot adjustments
    ax.set_xlim(xlim[0], xlim[1])
    xticks = np.arange(xlim[0], xlim[1] + 0.2, 0.2)  
    ax.set_xticks(xticks)
    ax.set_xlabel("Pearson-r", fontsize=label_fontsize)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["Model"], fontsize=label_fontsize)
    ax.invert_yaxis()  # Higher scores on top
    ax.tick_params(axis='x', labelsize=label_fontsize)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=color, edgecolor='black', label=group) for group, color in group_color_map.items()]
    ax.legend(handles=legend_patches, fontsize=legend_fontsize, bbox_to_anchor=(1.05, .5), loc='upper left')

    sns.despine()
    plt.tight_layout()
    plt.show()

def plot_true_vs_null_model_barchart(
    summary_true_dict,
    summary_null_dict,
    metric="test_pearson_r",
    model_groups=None,
    xlim=(0.1, 0.9),
    overlay_style="alpha",  # or "hatch"
    display_metric=False,
    horizontal=False
):
    if model_groups is None:
        model_groups = {
            'Non-Linear': {
                'dynamic_mlp': 'MLP',
                'dynamic_mlp_coords': 'MLP w/ coords',
                'shared_transformer': 'SMT',
                'shared_transformer_cls': 'SMT w/ [CLS]'
            },
            'Bilinear': {
                'bilinear_CM': 'CM',
                'bilinear_CM_PCA': 'CM (PCA)',
                'pls_bilineardecoder': 'CM (PLS)',
                'bilinear_lowrank': 'CM (Lowrank)'
            },
            'Feature Based': {
                'cge': 'CGE',
                'gaussian_kernel': 'Gauss. Kernel',
                'exponential_decay': 'Exp. Decay'
            }
        }

    base_fontsize = 22
    label_fontsize = base_fontsize * 0.67
    plt.rcParams.update({'font.size': base_fontsize})

    # Prepare plot data
    plot_data = []
    for group_name, model_dict in model_groups.items():
        for model_key, display_name in model_dict.items():
            if model_key in summary_true_dict and model_key in summary_null_dict:
                df_true = summary_true_dict[model_key]
                df_null = summary_null_dict[model_key]
                if metric in df_true.columns and metric in df_null.columns:
                    plot_data.append({
                        "Model": display_name,
                        "Group": group_name,
                        "TrueMean": df_true.loc["mean", metric],
                        "TrueStd": df_true.loc["std", metric],
                        "NullMean": df_null.loc["mean", metric],
                        "NullStd": df_null.loc["std", metric]
                    })

    plot_df = pd.DataFrame(plot_data)
    
    # Sort by performance - for horizontal, ascending=True (worst to best left to right)
    # For vertical, ascending=False (best to worst top to bottom)
    plot_df = plot_df.sort_values("TrueMean", ascending=horizontal).reset_index(drop=True)

    # Color map
    unique_groups = list(model_groups.keys())
    palette = sns.color_palette("viridis", n_colors=12, desat=1.0)[2::4]
    group_color_map = {group: color for group, color in zip(unique_groups, palette)}

    # Plotting - adjust figure size based on orientation
    if horizontal:
        plt.figure(figsize=(7, 4), dpi=300)
    else:
        plt.figure(figsize=(6, 7), dpi=300)
    ax = plt.gca()

    for i, row in plot_df.iterrows():
        if horizontal:
            # Horizontal bars (vertical chart)
            x = i
            ax.bar(
                x=x,
                height=row["TrueMean"],
                width=0.6,
                color=group_color_map[row["Group"]],
                edgecolor="black",
                zorder=1
            )
            ax.errorbar(
                x=x,
                y=row["TrueMean"],
                yerr=row["TrueStd"],
                fmt='none',
                ecolor='black',
                capsize=1,
                linewidth=1,
                zorder=2
            )

            if display_metric:
                ax.text(
                    x,
                    row["TrueMean"] + 0.02,  # slight offset from bar end
                    f"{row['TrueMean']:.3f} ± {row['TrueStd']:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=label_fontsize * 0.8,  # Smaller font since showing both values
                    color="black",
                    rotation=45
                )
            elif i == len(plot_df) - 1:  # Show metric for best model (rightmost)
                ax.text(
                    x,
                    row["TrueMean"] + 0.02,
                    f"{row['TrueMean']:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=label_fontsize*0.8,
                    color="black"
                )

            if row["Model"] not in ["Gaussian Kernel", "Exponential Decay"]:
                if overlay_style == 'hatch':
                    ax.bar(
                        x=x,
                        height=row["NullMean"],
                        width=0.6,
                        bottom=0,
                        facecolor='none',
                        edgecolor='black',
                        hatch="////",
                        linewidth=1,
                        zorder=3
                    )
                elif overlay_style == 'alpha':
                    ax.bar(
                        x=x,
                        height=row["NullMean"],
                        width=0.6,
                        bottom=0,
                        color="lightgray",
                        edgecolor="black",
                        alpha=0.3,
                        zorder=3
                    )
                ax.errorbar(
                    x=x,
                    y=row["NullMean"],
                    yerr=row["NullStd"],
                    fmt='none',
                    ecolor='black',
                    capsize=1,
                    linewidth=1,
                    linestyle='--',
                    zorder=4
                )
        else:
            # Vertical bars (horizontal chart) - original behavior
            y = i
            ax.barh(
                y=y,
                width=row["TrueMean"],
                height=0.6,
                color=group_color_map[row["Group"]],
                edgecolor="black",
                zorder=1
            )
            ax.errorbar(
                x=row["TrueMean"],
                y=y,
                xerr=row["TrueStd"],
                fmt='none',
                ecolor='black',
                capsize=1,
                linewidth=1,
                zorder=2
            )

            if display_metric:
                ax.text(
                    row["TrueMean"] + 0.02,  # slight offset from bar end
                    y,
                    f"{row['TrueMean']:.3f} ± {row['TrueStd']:.3f}",
                    va="center",
                    ha="left",
                    fontsize=label_fontsize * 0.8,  # Smaller font since showing both values
                    color="black"
                )
            elif i == 0:  # Only show metric for top model if display_metric is False
                ax.text(
                    row["TrueMean"],
                    y - 0.4,  # shift upward
                    f"{row['TrueMean']:.2f}",
                    va="bottom",
                    ha="center",
                    fontsize=label_fontsize,
                    color="black"
                )

            if row["Model"] not in ["Gaussian Kernel", "Exponential Decay"]:
                if overlay_style == 'hatch':
                    ax.barh(
                        y=y,
                        width=row["NullMean"],
                        height=0.6,
                        facecolor='none',
                        edgecolor='black',
                        hatch="////",
                        linewidth=1,
                        zorder=3
                    )
                elif overlay_style == 'alpha':
                    ax.barh(
                        y=y,
                        width=row["NullMean"],
                        height=0.6,
                        color="lightgray",
                        edgecolor="black",
                        alpha=0.3,
                        zorder=3
                    )
                ax.errorbar(
                    x=row["NullMean"],
                    y=y,
                    xerr=row["NullStd"],
                    fmt='none',
                    ecolor='black',
                    capsize=1,
                    linewidth=1,
                    linestyle='--',
                    zorder=4
                )

    # Set axis properties based on orientation
    if horizontal:
        ax.set_ylim(*xlim)
        ax.set_yticks(np.arange(xlim[0], xlim[1] + 0.2, 0.2))
        ax.set_ylabel("Pearson-r", fontsize=label_fontsize)
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(plot_df["Model"], fontsize=label_fontsize, rotation=45, ha='right')
        ax.tick_params(axis='y', labelsize=label_fontsize)
        ax.tick_params(axis='x', labelsize=label_fontsize, length=0)  # Remove tick marks but keep labels
    else:
        ax.set_xlim(*xlim)
        ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.2, 0.2))
        ax.set_xlabel("Pearson-r", fontsize=label_fontsize)
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df["Model"], fontsize=label_fontsize)
        ax.invert_yaxis()
        ax.tick_params(axis='x', labelsize=label_fontsize)

    sns.despine()
    plt.tight_layout()
    plt.show()

    return group_color_map  # return for external legend


def plot_true_vs_null_model_barchart_weighted(
    summary_true_dict,
    summary_null_dict,
    metric="final_test_pearson_r",
    model_groups=None,
    xlim=(0.1, 0.9),
    overlay_style="alpha",  # or "hatch"
    horizontal=False
):
    """
    Plot a horizontal bar chart comparing true vs null model performance using weighted metrics.

    Args:
        summary_true_dict (dict): Model → summary DataFrame (with weighted metrics).
        summary_null_dict (dict): Model → summary DataFrame for null (e.g. spin).
        metric (str): Metric column name (default = "final_test_pearson_r").
        model_groups (dict): Optional custom model groupings.
        xlim (tuple): X-axis limits for the bar chart.
        overlay_style (str): Style for null bars ('alpha' or 'hatch').

    Returns:
        dict: Group-to-color mapping for external legend construction.
    """
    if model_groups is None:
        model_groups = {
            'Non-Linear': {
                'dynamic_mlp': 'MLP',
                'dynamic_mlp_coords': 'MLP w/ coords',
                'shared_transformer': 'SMT',
                'shared_transformer_cls': 'SMT w/ [CLS]'
            },
            'Bilinear': {
                'bilinear_CM': 'Connectome Model',
                'bilinear_CM_PCA': 'Connectome Model (PCA)',
                'pls_bilineardecoder': 'Connectome Model (PLS)',
                'bilinear_lowrank': 'Connectome Model (LR)'
            },
            'Feature Based': {
                'cge': 'CGE',
                'gaussian_kernel': 'Gaussian Kernel',
                'exponential_decay': 'Exponential Decay'
            }
        }

    base_fontsize = 22
    label_fontsize = base_fontsize * 0.67
    plt.rcParams.update({'font.size': base_fontsize})

    # Collect data into plot dataframe
    plot_data = []
    for group_name, model_dict in model_groups.items():
        for model_key, display_name in model_dict.items():
            if model_key in summary_true_dict and model_key in summary_null_dict:
                df_true = summary_true_dict[model_key]
                df_null = summary_null_dict[model_key]
                if metric in df_true.columns and metric in df_null.columns:
                    plot_data.append({
                        "Model": display_name,
                        "Group": group_name,
                        "WeightedTrueMean": df_true.loc["weighted_mean", metric],
                        "WeightedTrueStd": df_true.loc["weighted_std", metric],
                        "WeightedNullMean": df_null.loc["weighted_mean", metric],
                        "WeightedNullStd": df_null.loc["weighted_std", metric]
                    })

    plot_df = pd.DataFrame(plot_data)
    
    # Sort by performance - for horizontal, ascending=True (worst to best left to right)
    # For vertical, ascending=False (best to worst top to bottom)
    plot_df = plot_df.sort_values("WeightedTrueMean", ascending=horizontal).reset_index(drop=True)

    # Color mapping
    unique_groups = list(model_groups.keys())
    palette = sns.color_palette("viridis", n_colors=12)[2::4]
    group_color_map = {group: color for group, color in zip(unique_groups, palette)}

    # Plot - adjust figure size based on orientation
    if horizontal:
        plt.figure(figsize=(10, 4), dpi=300)
    else:
        plt.figure(figsize=(6, 7), dpi=300)
    ax = plt.gca()

    for i, row in plot_df.iterrows():
        if horizontal:
            # Horizontal bars (vertical chart)
            x = i
            ax.bar(
                x=x,
                height=row["WeightedTrueMean"],
                width=0.6,
                color=group_color_map[row["Group"]],
                edgecolor="black",
                zorder=1
            )
            ax.errorbar(
                x=x,
                y=row["WeightedTrueMean"],
                yerr=row["WeightedTrueStd"],
                fmt='none',
                ecolor='black',
                capsize=1,
                linewidth=1,
                zorder=2
            )

            # Annotate best model with Pearson-r (rightmost)
            if i == len(plot_df) - 1:
                ax.text(
                    x,
                    row["WeightedTrueMean"] + 0.02,
                    f"{row['WeightedTrueMean']:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=label_fontsize,
                    color="black"
                )

            # Overlay null bar (except for kernel baselines)
            if row["Model"] not in ["Gaussian Kernel", "Exponential Decay"]:
                if overlay_style == "hatch":
                    ax.bar(
                        x=x,
                        height=row["WeightedNullMean"],
                        width=0.6,
                        bottom=0,
                        facecolor='none',
                        edgecolor='black',
                        hatch="////",
                        linewidth=1,
                        zorder=3
                    )
                elif overlay_style == "alpha":
                    ax.bar(
                        x=x,
                        height=row["WeightedNullMean"],
                        width=0.6,
                        bottom=0,
                        color="lightgray",
                        edgecolor="black",
                        alpha=0.3,
                        zorder=3
                    )
                ax.errorbar(
                    x=x,
                    y=row["WeightedNullMean"],
                    yerr=row["WeightedNullStd"],
                    fmt='none',
                    ecolor='black',
                    capsize=1,
                    linewidth=1,
                    linestyle='--',
                    zorder=4
                )
        else:
            # Vertical bars (horizontal chart) - original behavior
            y = i
            ax.barh(
                y=y,
                width=row["WeightedTrueMean"],
                height=0.6,
                color=group_color_map[row["Group"]],
                edgecolor="black",
                zorder=1
            )
            ax.errorbar(
                x=row["WeightedTrueMean"],
                y=y,
                xerr=row["WeightedTrueStd"],
                fmt='none',
                ecolor='black',
                capsize=1,
                linewidth=1,
                zorder=2
            )

            # Annotate best model with Pearson-r
            if i == 0:
                ax.text(
                    row["WeightedTrueMean"],
                    y - 0.4,  # shift upward (adjust spacing if needed)
                    f"{row['WeightedTrueMean']:.2f}",
                    va="bottom",
                    ha="center",
                    fontsize=label_fontsize,
                    color="black"
                )

            # Overlay null bar (except for kernel baselines)
            if row["Model"] not in ["Gaussian Kernel", "Exponential Decay"]:
                if overlay_style == "hatch":
                    ax.barh(
                        y=y,
                        width=row["WeightedNullMean"],
                        height=0.6,
                        facecolor='none',
                        edgecolor='black',
                        hatch="////",
                        linewidth=1,
                        zorder=3
                    )
                elif overlay_style == "alpha":
                    ax.barh(
                        y=y,
                        width=row["WeightedNullMean"],
                        height=0.6,
                        color="lightgray",
                        edgecolor="black",
                        alpha=0.3,
                        zorder=3
                    )
                ax.errorbar(
                    x=row["WeightedNullMean"],
                    y=y,
                    xerr=row["WeightedNullStd"],
                    fmt='none',
                    ecolor='black',
                    capsize=1,
                    linewidth=1,
                    linestyle='--',
                    zorder=4
                )

    # Set axis properties based on orientation
    if horizontal:
        ax.set_ylim(*xlim)
        ax.set_yticks(np.arange(xlim[0], xlim[1] + 0.2, 0.2))
        ax.set_ylabel("Pearson-r", fontsize=label_fontsize)
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(plot_df["Model"], fontsize=label_fontsize, rotation=45, ha='right')
        ax.tick_params(axis='y', labelsize=label_fontsize)
        ax.tick_params(axis='x', labelsize=label_fontsize, length=0)  # Remove tick marks but keep labels
    else:
        ax.set_xlim(*xlim)
        ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.2, 0.2))
        ax.set_xlabel("Pearson-r", fontsize=label_fontsize)
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df["Model"], fontsize=label_fontsize)
        ax.invert_yaxis()
        ax.tick_params(axis='x', labelsize=label_fontsize)

    sns.despine()
    plt.tight_layout()
    plt.show()

    return group_color_map

def plot_final_subsetted_barchart(
    summary_true_dict,
    summary_null_dict,
    metric="test_pearson_r",
    ylim=(0.1, 0.9),
    overlay_style="ratio",  # "alpha", "hatch", or "ratio"
    horizontal=False,
    title=None
):
    """
    Create a final subsetted bar chart with customizable model groups and performance axis limits.
    
    Args:
        summary_true_dict (dict): Model → summary DataFrame for true performance
        summary_null_dict (dict): Model → summary DataFrame for null performance  
        metric (str): Metric column name (default = "test_pearson_r")
        ylim (tuple): Exact y-axis limits for performance values
        overlay_style (str): Style for null comparison - "alpha", "hatch", or "ratio"
        horizontal (bool): If True, create horizontal layout (worst to best, left to right)
        title (str): Optional title for the plot (e.g., 'Random Split UKBB Performance')
    
    Returns:
        dict: Group-to-color mapping for external legend construction
    """
    
    # Manually defined model groups - can be easily modified here
    model_groups = {
        'Non-Linear': {
            'dynamic_mlp': 'MLP',
            'shared_transformer': 'SMT',
            'dynamic_mlp_coords': 'MLP w/ coords',
            'shared_transformer_cls': 'SMT w/ [CLS]'
        },
        'Bilinear': {
            'pls_bilineardecoder': 'Bilinear PLS',
            'bilinear_lowrank': 'Bilinear Low-Rank',
        }
    }

    base_fontsize = 22
    label_fontsize = base_fontsize * 0.7
    plt.rcParams.update({'font.size': base_fontsize})

    # Prepare plot data
    plot_data = []
    for group_name, model_dict in model_groups.items():
        for model_key, display_name in model_dict.items():
            if model_key in summary_true_dict and model_key in summary_null_dict:
                df_true = summary_true_dict[model_key]
                df_null = summary_null_dict[model_key]
                if metric in df_true.columns and metric in df_null.columns:
                    plot_data.append({
                        "Model": display_name,
                        "Group": group_name,
                        "TrueMean": df_true.loc["mean", metric],
                        "TrueStd": df_true.loc["std", metric],
                        "NullMean": df_null.loc["mean", metric],
                        "NullStd": df_null.loc["std", metric]
                    })

    plot_df = pd.DataFrame(plot_data)
    
    # Sort by performance - for horizontal, ascending=True (worst to best left to right)
    # For vertical, ascending=False (best to worst top to bottom)
    plot_df = plot_df.sort_values("TrueMean", ascending=horizontal).reset_index(drop=True)

    # Color mapping
    unique_groups = list(model_groups.keys())
    palette = sns.color_palette("viridis", n_colors=12, desat=1.0)[2::4]
    group_color_map = {group: color for group, color in zip(unique_groups, palette)}

    # Plotting - adjust figure size based on orientation
    if horizontal:
        plt.figure(figsize=(7, 5), dpi=300)
    else:
        plt.figure(figsize=(6, 7), dpi=300)
    ax = plt.gca()

    for i, row in plot_df.iterrows():
        if horizontal:
            # Horizontal bars (vertical chart)
            x = i
            ax.bar(
                x=x,
                height=row["TrueMean"],
                width=0.6,
                color=group_color_map[row["Group"]],
                edgecolor="black",
                zorder=1
            )
            ax.errorbar(
                x=x,
                y=row["TrueMean"],
                yerr=row["TrueStd"],
                fmt='none',
                ecolor='black',
                capsize=2,
                linewidth=2,
                zorder=2
            )

            # Add metric text based on overlay style
            if overlay_style == "ratio":
                # Show true/null ratio without leading zeros
                ratio_text = f"{row['TrueMean']:.2f}/{row['NullMean']:.2f}".replace('0.', '.')
                ax.text(
                    x,
                    row["TrueMean"] + 0.025,  # Increased spacing from bar
                    ratio_text,
                    ha="center",
                    va="bottom",
                    fontsize=label_fontsize,  # Increased font size
                    color="black"
                )
            else:
                # Show true performance for all models if alpha style, otherwise only best model
                if overlay_style == "alpha":
                    # Show true performance above all bars for alpha overlay
                    ax.text(
                        x,
                        row["TrueMean"] + 0.025,  # Increased spacing from bar
                        f"{row['TrueMean']:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=label_fontsize,  # Increased font size
                        color="black"
                    )
                elif i == len(plot_df) - 1:
                    # Show only true performance for best model (rightmost) for other styles
                    ax.text(
                        x,
                        row["TrueMean"] + 0.025,  # Increased spacing from bar
                        f"{row['TrueMean']:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=label_fontsize,  # Increased font size
                        color="black"
                    )

            # Add null overlay if not using ratio style
            if overlay_style != "ratio":
                if overlay_style == 'hatch':
                    ax.bar(
                        x=x,
                        height=row["NullMean"],
                        width=0.6,
                        bottom=0,
                        facecolor='none',
                        edgecolor='black',
                        hatch="////",
                        linewidth=1,
                        zorder=3
                    )
                elif overlay_style == 'alpha':
                    ax.bar(
                        x=x,
                        height=row["NullMean"],
                        width=0.6,
                        bottom=0,
                        color="lightgray",
                        edgecolor="black",
                        alpha=0.3,
                        zorder=3
                    )
                ax.errorbar(
                    x=x,
                    y=row["NullMean"],
                    yerr=row["NullStd"],
                    fmt='none',
                    ecolor='black',
                    capsize=1,
                    linewidth=1,
                    linestyle='--',
                    zorder=4
                )
        else:
            # Vertical bars (horizontal chart) - original behavior
            y = i
            ax.barh(
                y=y,
                width=row["TrueMean"],
                height=0.6,
                color=group_color_map[row["Group"]],
                edgecolor="black",
                zorder=1
            )
            ax.errorbar(
                x=row["TrueMean"],
                y=y,
                xerr=row["TrueStd"],
                fmt='none',
                ecolor='black',
                capsize=2,
                linewidth=2,
                zorder=2
            )

            # Add metric text based on overlay style
            if overlay_style == "ratio":
                # Show true/null ratio without leading zeros
                ratio_text = f"{row['TrueMean']:.2f}/{row['NullMean']:.2f}".replace('0.', '.')
                ax.text(
                    row["TrueMean"] + 0.02,  # Increased spacing from bar
                    y,
                    ratio_text,
                    va="center",
                    ha="left",
                    fontsize=label_fontsize,  # Increased font size
                    color="black"
                )
            else:
                # Show only true performance for best model (topmost)
                if i == 0:
                    ax.text(
                        row["TrueMean"],
                        y - 0.4,
                        f"{row['TrueMean']:.2f}",
                        va="bottom",
                        ha="center",
                        fontsize=label_fontsize,
                        color="black"
                    )

            # Add null overlay if not using ratio style
            if overlay_style != "ratio":
                if overlay_style == 'hatch':
                    ax.barh(
                        y=y,
                        width=row["NullMean"],
                        height=0.6,
                        facecolor='none',
                        edgecolor='black',
                        hatch="////",
                        linewidth=1,
                        zorder=3
                    )
                elif overlay_style == 'alpha':
                    ax.barh(
                        y=y,
                        width=row["NullMean"],
                        height=0.6,
                        color="lightgray",
                        edgecolor="black",
                        alpha=0.3,
                        zorder=3
                    )
                ax.errorbar(
                    x=row["NullMean"],
                    y=y,
                    xerr=row["NullStd"],
                    fmt='none',
                    ecolor='black',
                    capsize=1,
                    linewidth=1,
                    linestyle='--',
                    zorder=4
                )

    # Set axis properties based on orientation with hard-coded limits
    range_size = ylim[1] - ylim[0]
    if range_size % 0.2 == 0:
        tick_interval = 0.2
    else:
        tick_interval = 0.1
    
    if horizontal:
        ax.set_ylim(*ylim)
        ax.set_yticks(np.arange(ylim[0], ylim[1] + tick_interval/2, tick_interval))
        ax.set_ylabel("Pearson-r", fontsize=label_fontsize)
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(plot_df["Model"], fontsize=label_fontsize, rotation=45, ha='right')
        ax.tick_params(axis='y', labelsize=label_fontsize)
        ax.tick_params(axis='x', labelsize=label_fontsize, length=0)  # Remove tick marks but keep labels
    else:
        ax.set_xlim(*ylim)
        ax.set_xticks(np.arange(ylim[0], ylim[1] + tick_interval/2, tick_interval))
        ax.set_xlabel("Pearson-r", fontsize=label_fontsize)
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df["Model"], fontsize=label_fontsize)
        ax.invert_yaxis()
        ax.tick_params(axis='x', labelsize=label_fontsize)

    # Add title if provided
    if title:
        plt.title(title, fontsize=base_fontsize-4, pad=30)

    sns.despine()
    plt.tight_layout()
    plt.show()

    return group_color_map
