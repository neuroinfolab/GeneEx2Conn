from env.imports import *
from itertools import combinations
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from IPython import get_ipython
from data.data_utils import reconstruct_connectome
import models.metrics.distance_FC
from models.metrics.distance_FC import distance_FC


def in_jupyter_notebook():
    """Check if code is running in a Jupyter notebook"""
    try:
        if 'IPKernelApp' in get_ipython().config:
            return True
    except:
        pass
    return False

class PlotConfig:
    """Configuration for evaluation and plotting"""
    
    def __init__(self, plot_mode='basic', **plot_params):
        # Auto-detect jupyter for plot display
        self.show_plots = True if in_jupyter_notebook() else False
        
        # Plot modes: 'metrics' (no plots), 'basic' (key plots), 'verbose' (all plots)
        self.plot_mode = plot_mode
        self.mode_title = '' # dynamically set to Train or Test

        # Global plot parameters with defaults
        self.plot_params = {
            'figsize': (10, 7),
            'alpha': 0.5,
            'point_size': 5,
            'font_size': 26,
            **plot_params
        }

# Custom numpy scorers for inner-CV
def pearson_numpy(y_true, y_pred):
    """Numpy-based Pearson correlation"""
    return pearsonr(y_true, y_pred)[0]

def mse_numpy(y_true, y_pred):
    """Numpy-based mean squared error"""
    return np.mean(np.square(y_true - y_pred))

def r2_numpy(y_true, y_pred):
    """Numpy-based R-squared"""
    y_true_mean = np.mean(y_true)
    total_ss = np.sum(np.square(y_true - y_true_mean))
    residual_ss = np.sum(np.square(y_true - y_pred))
    return 1 - (residual_ss / total_ss)

def accuracy_numpy(y_true, y_pred):
    """Numpy-based accuracy"""
    y_pred_labels = np.round(y_pred)
    return accuracy_score(y_true, y_pred_labels)

def logloss_numpy(y_true, y_pred):
    """Numpy-based log loss"""
    return log_loss(y_true, y_pred)

# Custom cupy scorers for inner-CV (GPU acceleration)
def pearson_cupy(y_true, y_pred):
    """Cupy-based Pearson correlation"""
    y_pred = cp.asarray(y_pred)
    y_true = cp.asarray(y_true).ravel()
    y_pred = y_pred.ravel()
    corr_matrix = cp.corrcoef(y_true, y_pred)
    cp.cuda.Stream.null.synchronize()
    return corr_matrix[0, 1]

def mse_cupy(y_true, y_pred):
    """Cupy-based mean squared error"""
    y_pred = cp.asarray(y_pred)
    mse = cp.mean(cp.square(y_pred - y_true))
    cp.cuda.Stream.null.synchronize()
    return mse

def r2_cupy(y_true, y_pred):
    """Cupy-based R-squared"""
    y_pred = cp.asarray(y_pred)
    y_true_mean = cp.mean(y_true)
    total_ss = cp.sum(cp.square(y_true - y_true_mean))
    residual_ss = cp.sum(cp.square(y_true - y_pred))
    return 1 - (residual_ss / total_ss)

def accuracy_cupy(y_true, y_pred):
    """Cupy-based accuracy"""
    y_pred = cp.asarray(y_pred)
    y_true = cp.asarray(y_true)

    y_pred_labels = cp.round(y_pred)
    accuracy = cp.mean(y_pred_labels == y_true)
    
    cp.cuda.Stream.null.synchronize()
    return accuracy.item()

def logloss_cupy(y_true, y_pred):
    """Cupy-based log loss"""
    # Convert inputs to cupy arrays
    y_pred = cp.asarray(y_pred)
    y_true = cp.asarray(y_true)
    
    # Clip predictions to avoid log(0) 
    eps = 1e-15
    y_pred = cp.clip(y_pred, eps, 1 - eps)
    
    # Calculate log loss
    losses = -(y_true * cp.log(y_pred) + (1 - y_true) * cp.log(1 - y_pred))
    loss = cp.mean(losses)
    
    cp.cuda.Stream.null.synchronize()
    return loss.item()


def compute_basic_metrics(y_true, y_pred, binarize=False):
    """Compute basic metrics for regression or classification"""
    if binarize:
        y_pred_binary = np.round(y_pred) # threshold at 0.5
        return {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary),
            'recall': recall_score(y_true, y_pred_binary),
            'f1': f1_score(y_true, y_pred_binary),
            'auc_roc': roc_auc_score(y_true, y_pred)
        }
    else:
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'pearson_r': pearsonr(y_true, y_pred)[0]
        }


def compute_distance_metrics(y_true, y_pred, distances):
    """Compute distance-based correlation metrics"""
    # Distance range thresholds
    dist_33, dist_67 = 175/3, 175*2/3
    
    # Create masks
    short_mask = distances <= dist_33
    mid_mask = (distances > dist_33) & (distances <= dist_67)
    long_mask = distances > dist_67
    
    # Calculate correlations
    metrics = {'overall_r': pearsonr(y_true, y_pred)[0]}
    
    if np.sum(short_mask) > 1:
        metrics['short_r'] = pearsonr(y_true[short_mask], y_pred[short_mask])[0]
    else:
        metrics['short_r'] = np.nan
        
    if np.sum(mid_mask) > 1:
        metrics['mid_r'] = pearsonr(y_true[mid_mask], y_pred[mid_mask])[0]
    else:
        metrics['mid_r'] = np.nan
        
    if np.sum(long_mask) > 1:
        metrics['long_r'] = pearsonr(y_true[long_mask], y_pred[long_mask])[0]
    else:
        metrics['long_r'] = np.nan
        
    return metrics


def compute_hemispheric_metrics(y_true, y_pred, indices, coords):
    """Compute hemisphere-based correlation metrics"""
    left_true, left_pred = [], []
    right_true, right_pred = [], []
    inter_true, inter_pred = [], []
    
    for idx, (i, j) in enumerate(combinations(indices, 2)):
        # Get x coordinates for both regions
        x_i = coords[i][0]
        x_j = coords[j][0]
        
        # Determine hemisphere based on x coordinates
        if x_i < 0 and x_j < 0:  # Left-left
            left_true.append(y_true[2*idx])
            left_pred.append(y_pred[2*idx])
        elif x_i > 0 and x_j > 0:  # Right-right
            right_true.append(y_true[2*idx])
            right_pred.append(y_pred[2*idx])
        else:  # Inter-hemispheric
            inter_true.append(y_true[2*idx])
            inter_pred.append(y_pred[2*idx])

    metrics = {}
    if len(left_true) > 1:
        metrics['left_hemi_r'] = pearsonr(left_true, left_pred)[0]
    if len(right_true) > 1:
        metrics['right_hemi_r'] = pearsonr(right_true, right_pred)[0]
    if len(inter_true) > 1:
        metrics['inter_hemi_r'] = pearsonr(inter_true, inter_pred)[0]
        
    return metrics


def compute_subnetwork_metrics(y_true, y_pred, indices, network_labels): 
    """
    Compute network-based correlation metrics for both intra and inter-network connections.
    
    Returns metrics dictionary with correlations for:
    - Intra-network connections within each network
    - Inter-network connections between Visual system and all other networks
    """
    networks = ['Cont', 'Default', 'SalVentAttn', 'Limbic', 
               'DorsAttn', 'SomMot', 'Vis', 'Subcortical', 'Cerebellum']
    
    # Initialize data structures for intra and inter-network connections
    intra_network_data = {net: {'true': [], 'pred': []} for net in networks}
    inter_network_data = {net: {'true': [], 'pred': []} for net in networks}

    for idx, (i, j) in enumerate(combinations(indices, 2)):
        net_i, net_j = network_labels[i], network_labels[j]
        
        # Skip if either network label is not in our list
        if net_i not in networks or net_j not in networks:
            continue
            
        true_val = y_true[2*idx]
        pred_val = y_pred[2*idx]
        
        if net_i == net_j: # Intra-network connections
            intra_network_data[net_i]['true'].append(true_val)
            intra_network_data[net_i]['pred'].append(pred_val)
        else: # Inter-network connections
            for net in [net_i, net_j]: # Add to both networks' inter-network data
                inter_network_data[net]['true'].append(true_val)
                inter_network_data[net]['pred'].append(pred_val)

    metrics = {}
    for network in networks:
        if len(intra_network_data[network]['true']) > 1:
            metrics[f'intra_network_{network}_r'] = pearsonr(
                intra_network_data[network]['true'],
                intra_network_data[network]['pred'])[0]
        else:
            metrics[f'intra_network_{network}_r'] = np.nan
    
    for network in networks:
        if len(inter_network_data[network]['true']) > 1:
            metrics[f'inter_network_{network}_r'] = pearsonr(
                inter_network_data[network]['true'],
                inter_network_data[network]['pred'])[0]
        else:
            metrics[f'inter_network_{network}_r'] = np.nan
            
    return metrics


# Simple plotting functions
def plot_distance_scatter(y_true, y_pred, distances, mode, config):
    """Generate distance-colored scatter plot"""
    plt.figure(figsize=config.plot_params['figsize'])
    
    scatter = plt.scatter(y_true, y_pred, c=distances, cmap='viridis',
                         alpha=config.plot_params['alpha'], 
                         s=config.plot_params['point_size'])
    
    # Add line of best fit and reference lines
    z = np.polyfit(y_true, y_pred, 1)
    plt.plot(y_true, np.poly1d(z)(y_true), "r:", alpha=0.5)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    
    # Formatting
    plt.xlabel('True Values', fontsize=config.plot_params['font_size'])
    plt.ylabel('Predicted Values', fontsize=config.plot_params['font_size'])
    plt.title(f'{config.mode_title} Distance-Based Predictions')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Distance (mm)', fontsize=config.plot_params['font_size'])
    
    plt.show()
    return plt.gcf()

def plot_subnetwork_scatter(y_true, y_pred, indices, network_labels, mode, config):
    """Generate network-based scatter plot colored by subnetwork"""
    plt.figure(figsize=config.plot_params['figsize'])
    
    # Define the standard network color scheme
    network_colors = {
        'Cont': '#D68E63',          # Darker Orange (Frontoparietal)
        'Default': '#D67A7A',       # Darker Red (Default Mode) 
        'SalVentAttn': '#55B755',   # Darker Green (Salience/Ventral Attention)
        'Limbic': '#D6CC7A',        # Darker Yellow (Limbic)
        'DorsAttn': '#D67AD6',      # Darker Magenta (Dorsal Attention)
        'SomMot': '#639CD6',        # Darker Light Blue (Somatomotor)
        'Vis': '#7B3B7B',           # Darker Purple (Visual)
        'Subcortical': '#808080',   # Gray (Subcortical)
        'Cerebellum': '#2F4F4F'     # Dark slate gray (Cerebellum)
    }

    # Initialize dictionaries to store predictions by network
    network_true = {net: [] for net in network_colors.keys()}
    network_pred = {net: [] for net in network_colors.keys()}

    # For each pair of regions in test set
    for idx, (i, j) in enumerate(combinations(indices, 2)):
        net_i = network_labels[i]
        net_j = network_labels[j]
        
        # Only store if regions are in same network
        if net_i == net_j and net_i in network_colors:
            network_true[net_i].append(y_true[2*idx])
            network_pred[net_i].append(y_pred[2*idx])

    # Plot each network's correlations
    for network in network_colors:
        if len(network_true[network]) > 0:
            true_vals = np.array(network_true[network])
            pred_vals = np.array(network_pred[network])
            
            # Calculate correlation
            corr = pearsonr(true_vals, pred_vals)[0]
            
            # Create scatter plot
            plt.scatter(true_vals, pred_vals,
                      c=network_colors[network], alpha=0.4, s=3,
                      label=f'{network} ({corr:.3f})')

    # Add line of best fit and reference lines
    z = np.polyfit(y_true, y_pred, 1)
    plt.plot(y_true, np.poly1d(z)(y_true), "r:", alpha=0.5)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    
    # Formatting
    plt.xlim(-0.4, 1.0)
    plt.ylim(-0.4, 1.0)
    plt.xlabel('True Values', fontsize=config.plot_params['font_size'])
    plt.ylabel('Predicted Values', fontsize=config.plot_params['font_size'])
    plt.title(f'{config.mode_title} Network-Based Predictions')
    
    plt.legend(fontsize=config.plot_params['font_size']-10, markerscale=3, 
              ncol=2, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    
    plt.show()
    return plt.gcf()


def plot_hemispheric_scatter(y_true, y_pred, indices, coords, mode, config):
    """Generate hemisphere-based scatter plot"""
    plt.figure(figsize=config.plot_params['figsize'])
    
    # Compute hemispheric data (same logic as metrics)
    left_true, left_pred = [], []
    right_true, right_pred = [], []
    inter_true, inter_pred = [], []
    
    for idx, (i, j) in enumerate(combinations(indices, 2)):
        # Get x coordinates for both regions
        x_i = coords[i][0]
        x_j = coords[j][0]
        
        # Determine hemisphere based on x coordinates
        if x_i < 0 and x_j < 0:  # Left-left
            left_true.append(y_true[2*idx])
            left_pred.append(y_pred[2*idx])
        elif x_i > 0 and x_j > 0:  # Right-right
            right_true.append(y_true[2*idx])
            right_pred.append(y_pred[2*idx])
        else:  # Inter-hemispheric
            inter_true.append(y_true[2*idx])
            inter_pred.append(y_pred[2*idx])

    # Plot each hemisphere type
    if left_true:
        left_r = pearsonr(left_true, left_pred)[0]
        plt.scatter(left_true, left_pred, c='#4040FF', alpha=0.2, s=1, 
                   label=f'Left Intra (r={left_r:.3f})')
    
    if right_true:
        right_r = pearsonr(right_true, right_pred)[0]
        plt.scatter(right_true, right_pred, c='#FF4040', alpha=0.2, s=1,
                   label=f'Right Intra (r={right_r:.3f})')
    
    if inter_true:
        inter_r = pearsonr(inter_true, inter_pred)[0]
        plt.scatter(inter_true, inter_pred, c='#40FF40', alpha=0.2, s=1,
                   label=f'Inter-hemi (r={inter_r:.3f})')

    # Add line of best fit and formatting
    z = np.polyfit(y_true, y_pred, 1)
    plt.plot(y_true, np.poly1d(z)(y_true), "r:", alpha=0.5)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    
    plt.xlim(-0.4, 1.0)
    plt.ylim(-0.4, 1.0)
    plt.xlabel('True Values', fontsize=config.plot_params['font_size'])
    plt.ylabel('Predicted Values', fontsize=config.plot_params['font_size'])
    plt.title(f'{config.mode_title} Hemispheric Predictions')
    plt.legend(fontsize=config.plot_params['font_size']-2, markerscale=5)
    plt.gca().set_aspect('equal')
    
    plt.show()
    return plt.gcf()


def plot_connectome_predictions_subset(Y_true, Y_pred, config):
    """Generate connectome comparison plots for subset predictions"""
    Y_pred_connectome = reconstruct_connectome(Y_pred, symmetric=True)
    Y_pred_connectome_asymmetric = reconstruct_connectome(Y_pred, symmetric=False)
    Y_true_connectome = reconstruct_connectome(Y_true)

    # Create high resolution figure
    plt.figure(figsize=(16, 4), dpi=300)
    
    plt.subplot(131)
    plt.imshow(Y_true_connectome, cmap='RdBu_r', vmin=-0.8, vmax=0.8, interpolation='nearest')
    plt.colorbar(shrink=0.5)
    plt.title(f'{config.mode_title} True Connectome', pad=10)
    
    plt.subplot(132) 
    plt.imshow(Y_pred_connectome_asymmetric, cmap='RdBu_r', vmin=-0.8, vmax=0.8, interpolation='nearest')
    plt.colorbar(shrink=0.5)
    plt.title(f'{config.mode_title} Predicted Connectome', pad=10)

    plt.subplot(133)
    plt.imshow(abs(Y_true_connectome - Y_pred_connectome_asymmetric), cmap='RdYlGn_r', interpolation='nearest')
    plt.colorbar(shrink=0.5)
    plt.title(f'{config.mode_title} Prediction Difference', pad=10)
    
    plt.tight_layout()
    plt.show()
    return plt.gcf()


def plot_connectome_predictions_full(Y, Y_true, Y_pred, indices, network_labels=None, binarize=False, config=None):
    """Generate full connectome visualization with prediction differences"""
    import matplotlib.colors as mcolors
    
    n = int(Y.shape[0])  # Get dimensions of square connectome
    if binarize: 
        split_mask = np.zeros((n, n)) + 0.1
    else: 
        split_mask = np.zeros((n, n))
        
    # Calculate prediction differences for region pairs
    diff = abs(Y_true - Y_pred)
    
    # For each pair of regions in indices
    for idx, (i, j) in enumerate(combinations(indices, 2)): # Each pair appears twice in diff - once in each direction
        split_mask[i,j] = diff[2*idx]   # First direction: i->j is at 2*idx
        split_mask[j,i] = diff[2*idx + 1]   # Second direction: j->i is at 2*idx + 1
            
    plt.figure(figsize=(16, 6))
    
    # Plot full connectome with network labels
    plt.subplot(121)
    plt.imshow(Y, cmap='RdBu_r', vmin=-0.8, vmax=0.8)
    plt.colorbar(shrink=0.5)
    mode_prefix = f"{config.mode_title} " if config and config.mode_title else ""
    plt.title(f'{mode_prefix}Full Connectome', fontsize=14)
    
    if network_labels is not None:
        # Create tick positions and labels
        tick_positions = []
        tick_labels = []
        start_idx = 0
        prev_label = network_labels[0]
        
        for i in range(1, len(network_labels)):
            if network_labels[i] != prev_label:
                tick_positions.append((start_idx + i - 1) / 2)
                tick_labels.append(prev_label)
                start_idx = i
                prev_label = network_labels[i]
        
        # Add the last group
        tick_positions.append((start_idx + len(network_labels) - 1) / 2)
        tick_labels.append(prev_label)

        plt.xticks(tick_positions, tick_labels, rotation=45, ha='right', fontsize=12)
        plt.yticks(tick_positions, tick_labels, fontsize=12)
    
    if binarize: 
        colors = [
            (0.0, "green"),  
            (0.1, "#2A0A4A"), 
            (1.0, "red"), 
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
    else: 
        colors = [
            (0.0, "#2A0A4A"),  # 0 -> dark purple
            (0.001, "green"),   # Near zero -> green
            (0.1, "green"),    # 0.1 -> still green
            (0.2, "yellow"),   # 0.2 -> yellow
            (1.0, "red"),      # 1.0 -> red
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
    
    # Plot prediction differences with network labels
    plt.subplot(122)
    plt.imshow(split_mask, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(shrink=0.5)
    plt.title(f'{mode_prefix}Prediction Differences', fontsize=14)
    
    if network_labels is not None:
        # Create tick positions and labels
        tick_positions = []
        tick_labels = []
        start_idx = 0
        prev_label = network_labels[0]
        
        for i in range(1, len(network_labels)):
            if network_labels[i] != prev_label:
                tick_positions.append((start_idx + i - 1) / 2)
                tick_labels.append(prev_label)
                start_idx = i
                prev_label = network_labels[i]
        
        # Add the last group
        tick_positions.append((start_idx + len(network_labels) - 1) / 2)
        tick_labels.append(prev_label)

        plt.xticks(tick_positions, tick_labels, rotation=45, ha='right', fontsize=12)
        plt.yticks(tick_positions, tick_labels, fontsize=12)
        
    plt.tight_layout()
    plt.show()
    return plt.gcf()


def format_metrics_output(metrics, mode):
    """Format metrics into organized categories for clean printing"""
    # Define metric categories
    global_metrics = ['mse', 'mae', 'r2', 'pearson_r', 'overall_r', 'accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    distance_metrics = ['short_r', 'mid_r', 'long_r']
    hemispheric_metrics = ['left_hemi_r', 'right_hemi_r', 'inter_hemi_r']
    graph_theory_metrics = ['geodesic_distance']
    
    # Network names for subnetwork metrics
    network_names = ['Cont', 'Default', 'SalVentAttn', 'Limbic', 'DorsAttn', 'SomMot', 'Vis', 'Subcortical', 'Cerebellum']
    
    print(f"\n{'='*50}")
    print(f"{mode.upper()} METRICS")
    print(f"{'='*50}")
    
    # Global Metrics
    global_found = {k: v for k, v in metrics.items() if k in global_metrics and not np.isnan(v)}
    if global_found:
        print(f"\nGLOBAL METRICS:")
        print("-" * 20)
        for metric, value in global_found.items():
            if metric in ['mse', 'mae']:
                print(f"  {metric.upper():<12}: {value:.6f}")
            elif metric in ['r2', 'pearson_r', 'overall_r']:
                print(f"  {metric.upper():<12}: {value:.4f}")
            else:
                print(f"  {metric.upper():<12}: {value:.4f}")
    
    # Distance-based Metrics
    distance_found = {k: v for k, v in metrics.items() if k in distance_metrics and not np.isnan(v)}
    if distance_found:
        print(f"\nDISTANCE-BASED CORRELATIONS:")
        print("-" * 35)
        for metric, value in distance_found.items():
            label = metric.replace('_r', '').replace('_', ' ').title()
            print(f"  {label:<12}: {value:.4f}")
    
    # Hemispheric Metrics
    hemispheric_found = {k: v for k, v in metrics.items() if k in hemispheric_metrics and not np.isnan(v)}
    if hemispheric_found:
        print(f"\nHEMISPHERIC CORRELATIONS:")
        print("-" * 30)
        for metric, value in hemispheric_found.items():
            if 'left' in metric:
                label = "Left Intra"
            elif 'right' in metric:
                label = "Right Intra"
            else:
                label = "Inter-hemi"
            print(f"  {label:<12}: {value:.4f}")
    
    # Subnetwork Metrics - Combined intra and inter
    intra_network_metrics = {k: v for k, v in metrics.items() 
                           if k.startswith('intra_network_') and k.endswith('_r') and not np.isnan(v)}
    inter_network_metrics = {k: v for k, v in metrics.items() 
                           if k.startswith('inter_network_') and k.endswith('_r') and not np.isnan(v)}
    
    if intra_network_metrics or inter_network_metrics:
        print(f"\nNETWORK CORRELATIONS:")
        print("-" * 25)
        for network in network_names:
            intra_key = f'intra_network_{network}_r'
            inter_key = f'inter_network_{network}_r'
            
            intra_val = metrics.get(intra_key)
            inter_val = metrics.get(inter_key)
            
            # Only show networks that have at least one valid metric
            if ((intra_val is not None and not np.isnan(intra_val)) or 
                (inter_val is not None and not np.isnan(inter_val))):
                
                intra_str = f"{intra_val:.4f}" if intra_val is not None and not np.isnan(intra_val) else "  N/A "
                inter_str = f"{inter_val:.4f}" if inter_val is not None and not np.isnan(inter_val) else "  N/A "
                
                print(f"  {network:<12}: Intra={intra_str} | Inter={inter_str}")
    
    # Graph Theory Metrics
    graph_found = {k: v for k, v in metrics.items() if k in graph_theory_metrics and not np.isnan(v)}
    if graph_found:
        print(f"\nGRAPH THEORY METRICS:")
        print("-" * 25)
        for metric, value in graph_found.items():
            label = metric.replace('_', ' ').title()
            print(f"  {label:<12}: {value:.4f}")
    
    # Other/Unknown Metrics (catch any we missed)
    all_categorized = (set(global_metrics) | set(distance_metrics) | set(hemispheric_metrics) | 
                      set(graph_theory_metrics) | 
                      {f'intra_network_{net}_r' for net in network_names} |
                      {f'inter_network_{net}_r' for net in network_names})
    
    other_metrics = {k: v for k, v in metrics.items() 
                    if k not in all_categorized and not np.isnan(v)}
    if other_metrics:
        print(f"\nOTHER METRICS:")
        print("-" * 20)
        for metric, value in other_metrics.items():
            print(f"  {metric:<12}: {value:.4f}")
    
    print(f"\n{'='*50}\n")