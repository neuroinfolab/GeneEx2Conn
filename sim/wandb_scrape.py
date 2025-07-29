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

def weighted_mean_and_se(values, weights):
    """
    Compute weighted mean and standard error for a set of values and weights.
    
    Args:
        values (np.ndarray): Array of values
        weights (list): List of weights corresponding to each value
        
    Returns:
        tuple: (weighted_mean, weighted_standard_error)
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
    
    # Calculate weighted standard error
    weighted_se = np.sqrt(weighted_var / len(values))
    
    return weighted_mean, weighted_se

def fetch_and_summarize_wandb_runs(model, cv_type, null_model, feature_type='transcriptome', days=7, use_weighted=False, exclude='HCP', return_history=False):
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
        days (int): Number of past days to search within
        use_weighted (bool): Whether to compute weighted statistics for schaefer/lobe CV
        exclude (str): Dataset to exclude from search (default: 'HCP')
        return_history (bool): If True, return (summary_df, history_df) tuple
    
    
    Returns:
        summary_df (pd.DataFrame): DataFrame with mean, std, stderr of all train/test metrics
                                  If use_weighted=True and cv_type in ['schaefer', 'lobe'], 
                                  includes weighted_mean and weighted_stderr rows
        history_df (pd.DataFrame): Individual run data (only returned if return_history=True)
    """
    time_filter = datetime.now() - timedelta(days=days)
    
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
                f"cv_type_{cv_type}",
                f"null_model_{null_model}",
                f"feature_type_{feature_type}"
            ],
            "$nin": ["dataset_HCP"]
        },
        "created_at": {"$gte": time_filter.isoformat()},
        "state": "finished"
    }
    
    # Add exclusion filter if specified
    if exclude:
        filters["tags"]["$nin"] = [f"dataset_{exclude}"]
    
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
            raise ValueError(f"❌ Expected {expected_runs} unique run names, found {len(df_unique)} after deduplication.")

    # Store history before cleaning for aggregation
    history_df = df_unique.copy()
    
    # Clean and summarize
    columns_to_drop = ["run_name", "run_id", "final_test_pearson_r"]
    if 'fold' in df_unique.columns:
        columns_to_drop.append("fold")
    df_clean = df_unique.drop(columns=columns_to_drop, errors="ignore")

    summary_df = pd.DataFrame({
        "mean": df_clean.mean(),
        "std": df_clean.std(),
        "stderr": df_clean.sem()
    }).T
    
    # Add weighted statistics if requested and applicable
    if use_weighted and cv_type in ['schaefer', 'lobe']:
        weights = list(CV_WEIGHTS[cv_type].values())
        
        # Calculate weighted statistics for final_test_pearson_r
        weighted_mean, weighted_se = weighted_mean_and_se(df_unique['final_test_pearson_r'].values, weights)
        summary_df.loc['weighted_mean', 'final_test_pearson_r'] = weighted_mean
        summary_df.loc['weighted_stderr', 'final_test_pearson_r'] = weighted_se
        
        # Calculate weighted statistics for test_pearson_r if it exists
        if 'test_pearson_r' in df_clean.columns:
            test_pearson_values = df_unique['test_pearson_r'].values if 'test_pearson_r' in df_unique.columns else df_clean['test_pearson_r'].values
            weighted_mean_test, weighted_se_test = weighted_mean_and_se(test_pearson_values, weights)
            summary_df.loc['weighted_mean', 'test_pearson_r'] = weighted_mean_test
            summary_df.loc['weighted_stderr', 'test_pearson_r'] = weighted_se_test

    if return_history:
        return summary_df, history_df
    else:
        return summary_df


def plot_model_barchart(summary_dict, metric="test_pearson_r", model_groups=None, xlim=(0.1, 0.9)):
    """
    Create a horizontal bar plot of model performance with error bars, ordered by descending mean.
    """
    if model_groups is None:
        model_groups = {
            'Deep Learning': {
                'dynamic_mlp': 'Deep Neural Net',
                'dynamic_mlp_coords': 'Deep Neural Net w/ coords',
                'shared_transformer': 'SMT',
                'shared_transformer_cls': 'SMT w/ [CLS]'
            },
            'Bilinear': {
                'bilinear_CM': 'CM',
                'bilinear_CM_PCA': 'CM (PCA)',
                'pls_bilineardecoder': 'CM (PLS)',
                'bilinear_lowrank': 'CM (Low-Rank)'},
            'Rules-Based': {
                'cge': 'CGE',
                'gaussian_kernel': 'Gaussian Kernel', 
                'exponential_decay': 'Exponential Decay'
            }
        }
    else:
        # Convert list-based model_groups to dict format
        formatted_groups = {}
        for group_name, models_list in model_groups.items():
            formatted_groups[group_name] = {
                model: model.replace('_', ' ').title() 
                for model in models_list
            }
        model_groups = formatted_groups

    # Set global font size and derived sizes
    base_fontsize = 20
    plt.rcParams.update({'font.size': base_fontsize})
    label_fontsize = base_fontsize * 0.67  # ~12
    legend_fontsize = base_fontsize * 0.78  # ~14

    # Flatten model info into DataFrame
    plot_data = []
    for group_name, model_dict in model_groups.items():
        for model_key, model_display in model_dict.items():
            if model_key in summary_dict:
                df = summary_dict[model_key]
                if metric in df.columns:
                    plot_data.append({
                        "Model": model_display,
                        "Group": group_name,
                        "Mean": df.loc["mean", metric],
                        "StdErr": df.loc["stderr", metric]
                    })

    plot_df = pd.DataFrame(plot_data)

    # Sort by descending mean performance
    plot_df = plot_df.sort_values("Mean", ascending=True)

    # Plot
    plt.figure(figsize=(8, 7), dpi=300)
    ax = sns.barplot(
        data=plot_df,
        y="Model",
        x="Mean",
        hue="Group",
        dodge=False,
        palette=sns.color_palette("viridis", n_colors=8, desat=1.0)[::3],  # Using every 3rd color from 8 viridis colors for more contrast
        errorbar=None
    )

    # Add error bars manually
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        ax.errorbar(
            x=row["Mean"],
            y=idx,
            xerr=row["StdErr"],
            fmt='none',
            ecolor='black',
            capsize=3,
            linewidth=2
        )

    ax.set_xlim(0.1, 1.0)
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_xlabel("Pearson-r", fontsize=label_fontsize)
    ax.set_ylabel("")
    ax.invert_yaxis()  # best at top, worst at bottom

    # Set tick label sizes relative to base font size
    ax.tick_params(axis='both', which='major', labelsize=label_fontsize)

    # Adjust legend with relative font size
    plt.legend(fontsize=legend_fontsize, bbox_to_anchor=(1.05, 1), loc='upper left')

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
                'dynamic_mlp': 'Deep Neural Net',
                'dynamic_mlp_coords': 'Deep Neural Net w/ coords',
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
                        "TrueStdErr": df_true.loc["stderr", metric],
                        "NullMean": df_null.loc["mean", metric],
                        "NullStdErr": df_null.loc["stderr", metric]
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
            xerr=row["TrueStdErr"],
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
                xerr=row["NullStdErr"],
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
    display_metric=False
):
    if model_groups is None:
        model_groups = {
            'Non-Linear': {
                'dynamic_mlp': 'Deep Neural Net',
                'dynamic_mlp_coords': 'Deep Neural Net w/ coords',
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
                        "TrueStdErr": df_true.loc["stderr", metric],
                        "NullMean": df_null.loc["mean", metric],
                        "NullStdErr": df_null.loc["stderr", metric]
                    })

    plot_df = pd.DataFrame(plot_data)
    plot_df = plot_df.sort_values("TrueMean", ascending=False).reset_index(drop=True)

    # Color map
    unique_groups = list(model_groups.keys())
    palette = sns.color_palette("viridis", n_colors=12, desat=1.0)[2::4]
    group_color_map = {group: color for group, color in zip(unique_groups, palette)}

    # Plotting
    plt.figure(figsize=(6, 7), dpi=300)
    ax = plt.gca()

    for i, row in plot_df.iterrows():
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
            xerr=row["TrueStdErr"],
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
                f"{row['TrueMean']:.3f}",
                va="center",
                ha="left",
                fontsize=label_fontsize,
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
                xerr=row["NullStdErr"],
                fmt='none',
                ecolor='black',
                capsize=1,
                linewidth=1,
                linestyle='--',
                zorder=4
            )

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
    overlay_style="alpha"  # or "hatch"
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
                'dynamic_mlp': 'Deep Neural Net',
                'dynamic_mlp_coords': 'Deep Neural Net w/ coords',
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
                        "WeightedTrueStdErr": df_true.loc["weighted_stderr", metric],
                        "WeightedNullMean": df_null.loc["weighted_mean", metric],
                        "WeightedNullStdErr": df_null.loc["weighted_stderr", metric]
                    })

    plot_df = pd.DataFrame(plot_data)
    plot_df = plot_df.sort_values("WeightedTrueMean", ascending=False).reset_index(drop=True)

    # Color mapping
    unique_groups = list(model_groups.keys())
    palette = sns.color_palette("viridis", n_colors=12)[2::4]
    group_color_map = {group: color for group, color in zip(unique_groups, palette)}

    # Plot
    plt.figure(figsize=(6, 7), dpi=300)
    ax = plt.gca()

    for i, row in plot_df.iterrows():
        y = i

        # True bar
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
            xerr=row["WeightedTrueStdErr"],
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
                xerr=row["WeightedNullStdErr"],
                fmt='none',
                ecolor='black',
                capsize=1,
                linewidth=1,
                linestyle='--',
                zorder=4
            )

    # Axes and styling
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


def process_model_feature_combinations(cv_type, null_model, days, models, model_feature_types, summary_dict, use_weighted=False, exclude='HCP'):
    """Helper function to process model/feature type combinations and populate summary dictionary"""
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
                df = fetch_and_summarize_wandb_runs(
                    model, cv_type, null_model, feature_type, days, 
                    use_weighted=use_weighted, exclude=exclude
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

