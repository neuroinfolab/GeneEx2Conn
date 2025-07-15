from env.imports import *
from itertools import combinations
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import matplotlib.colors as mcolors
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
            'figsize': (8, 6),
            'alpha': 0.5,
            'point_size': 3,
            'font_size': 18,
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


# Metrics functions
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
            'pearson_r': pearsonr(y_true, y_pred)[0],
            'spearman_r': spearmanr(y_true, y_pred)[0]
        }

def compute_distance_metrics(y_true, y_pred, distances):
    """Compute distance-based correlation metrics"""
    # Handle both tensor and numpy distances
    if torch.is_tensor(distances):
        distances = distances.cpu().numpy()
    else:
        distances = getattr(distances, 'get', lambda: distances)()

    # Distance range thresholds 
    dist_33, dist_67 = 175/3, 175*2/3
    
    # Create masks
    short_mask = distances <= dist_33
    mid_mask = (distances > dist_33) & (distances <= dist_67)
    long_mask = distances > dist_67

    # Calculate correlations
    metrics = {}
    
    # Convert masks to numpy if needed
    if torch.is_tensor(short_mask):
        short_mask = short_mask.cpu().numpy()
        mid_mask = mid_mask.cpu().numpy() 
        long_mask = long_mask.cpu().numpy()

    # Use numpy sum to count mask elements
    if np.sum(short_mask) > 100:
        metrics['short_r'] = pearsonr(y_true[short_mask], y_pred[short_mask])[0]
    else:
        metrics['short_r'] = np.nan
        
    if np.sum(mid_mask) > 100:
        metrics['mid_r'] = pearsonr(y_true[mid_mask], y_pred[mid_mask])[0]
    else:
        metrics['mid_r'] = np.nan
        
    if np.sum(long_mask) > 100:
        metrics['long_r'] = pearsonr(y_true[long_mask], y_pred[long_mask])[0]
    else:
        metrics['long_r'] = np.nan
        
    return metrics

def compute_strength_metrics(y_true, y_pred):
    """
    Compute correlation metrics based on connection strength ranges.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing correlations for strong negative (<-0.3), 
        weak (-0.3 to 0.3), and strong positive (>0.3) connections
    """
    # Create masks for different strength ranges
    strong_neg_mask = y_true < -0.3
    weak_mask = (y_true >= -0.3) & (y_true <= 0.3)
    strong_pos_mask = y_true > 0.3
    
    metrics = {}
    
    # Calculate correlations for each range if enough samples exist
    if np.sum(strong_neg_mask) > 100:
        metrics['strong_neg_r'] = pearsonr(y_true[strong_neg_mask], y_pred[strong_neg_mask])[0]
    else:
        metrics['strong_neg_r'] = np.nan
        
    if np.sum(weak_mask) > 100:
        metrics['weak_r'] = pearsonr(y_true[weak_mask], y_pred[weak_mask])[0]
    else:
        metrics['weak_r'] = np.nan
        
    if np.sum(strong_pos_mask) > 100:
        metrics['strong_pos_r'] = pearsonr(y_true[strong_pos_mask], y_pred[strong_pos_mask])[0]
    else:
        metrics['strong_pos_r'] = np.nan
        
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
    
def compute_subnetwork_metrics(y_true, y_pred, indices, network_labels, shared_indices=None):
    """
    Compute network-based correlation metrics for both intra and inter-network connections.
    
    Args:
        y_true: True connectivity values
        y_pred: Predicted connectivity values 
        indices: List of region indices
        network_labels: Network label for each region
        shared_indices: Optional list of shared region indices to include connections with
        
    Returns:
        Dictionary with correlations for:
        - Intra-network connections within each network
        - Inter-network connections between networks
    """
    networks = ['Cont', 'Default', 'SalVentAttn', 'Limbic', 
               'DorsAttn', 'SomMot', 'Vis', 'Subcortical', 'Cerebellum']
    
    # Initialize data structures for intra and inter-network connections
    intra_network_data = {net: {'true': [], 'pred': []} for net in networks}
    inter_network_data = {net: {'true': [], 'pred': []} for net in networks}

    # Helper function to process a pair of regions and their values
    def process_region_pair(net_i, net_j, true_val, pred_val):
        # Skip if either network label is not in our list
        if net_i not in networks or net_j not in networks:
            return
            
        # Add to appropriate network data
        if net_i == net_j: # Intra-network connections
            intra_network_data[net_i]['true'].append(true_val)
            intra_network_data[net_i]['pred'].append(pred_val)
        else: # Inter-network connections
            for net in [net_i, net_j]:
                inter_network_data[net]['true'].append(true_val)
                inter_network_data[net]['pred'].append(pred_val)

    # Process connections between indices
    n_pairs = len(list(combinations(indices, 2)))
    for idx, (i, j) in enumerate(combinations(indices, 2)):
        net_i, net_j = network_labels[i], network_labels[j]
        process_region_pair(net_i, net_j, y_true[idx], y_pred[idx])

    # Process connections with shared indices if provided
    if shared_indices is not None:
        offset = n_pairs
        for i_idx, i in enumerate(indices):
            for j_idx, j in enumerate(shared_indices):
                idx = offset + i_idx * len(shared_indices) + j_idx
                if idx < len(y_true):  # Ensure index is within bounds
                    net_i, net_j = network_labels[i], network_labels[j]
                    process_region_pair(net_i, net_j, y_true[idx], y_pred[idx])

    # Calculate metrics
    metrics = {}
    
    # Calculate intra-network correlations
    for network in networks:
        if len(intra_network_data[network]['true']) > 1:
            metrics[f'intra_network_{network}_r'] = pearsonr(
                intra_network_data[network]['true'],
                intra_network_data[network]['pred'])[0]
        else:
            metrics[f'intra_network_{network}_r'] = np.nan
    
    # Calculate inter-network correlations  
    for network in networks:
        if len(inter_network_data[network]['true']) > 1:
            metrics[f'inter_network_{network}_r'] = pearsonr(
                inter_network_data[network]['true'],
                inter_network_data[network]['pred'])[0]
        else:
            metrics[f'inter_network_{network}_r'] = np.nan
            
    return metrics

# Plotting functions
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

def plot_connectome_predictions_full(Y, Y_true, Y_pred, indices, network_labels=None, binarize=False, config=None, region_pair_dataset=None, shared_indices=None):
    """Generate full connectome visualization with prediction differences"""
    n = int(Y.shape[0])  # Get dimensions of square connectome
    if binarize: 
        split_mask = np.zeros((n, n)) + 0.1
    else: 
        split_mask = np.zeros((n, n))
        
    # Calculate prediction differences for region pairs
    diff = abs(Y_true - Y_pred)
    
    # Create matrix for predicted values
    pred_matrix = np.zeros_like(split_mask)
    
    # For each pair of regions in indices
    n_pairs = len(list(combinations(indices, 2)))
    for idx, (i, j) in enumerate(combinations(indices, 2)): # Each pair appears twice in diff - once in each direction
        split_mask[i,j] = diff[2*idx]   # First direction: i->j is at 2*idx
        split_mask[j,i] = diff[2*idx + 1]   # Second direction: j->i is at 2*idx + 1
        
        # Also store the actual predictions
        pred_matrix[i,j] = Y_pred[2*idx]
        pred_matrix[j,i] = Y_pred[2*idx + 1]
    
    # Process connections with shared indices if provided
    if shared_indices is not None:
        offset = n_pairs
        for i_idx, i in enumerate(indices):
            for j_idx, j in enumerate(shared_indices):
                idx = offset + i_idx * len(shared_indices) + j_idx
                if idx < len(diff):  # Ensure index is within bounds
                    split_mask[i,j] = diff[idx]
                    split_mask[j,i] = diff[idx]
                    
                    # Also store the actual predictions
                    pred_matrix[i,j] = Y_pred[idx]
                    pred_matrix[j,i] = Y_pred[idx]
            
    save_pred_matrix = False
    if save_pred_matrix:
        print(f"Split mask type: {type(split_mask)}")
        print(f"Split mask shape: {split_mask.shape}")
        print("Split mask:")
        print(split_mask)
        
        print(f"Pred matrix type: {type(pred_matrix)}")
        print(f"Pred matrix shape: {pred_matrix.shape}")
        print("Pred matrix:")
        print(pred_matrix)
        fold = 0
        save_dir = f"notebooks/NeurIPS/schaefer_preds/{region_pair_dataset.dataset}"
        os.makedirs(save_dir, exist_ok=True)
        while os.path.exists(f"{save_dir}/pred_matrix_fold{fold}.npy"):
            fold += 1
        np.save(f"{save_dir}/pred_matrix_fold{fold}.npy", pred_matrix)


    plt.figure(figsize=(16, 6))
    
    # Plot full connectome with network labels
    plt.subplot(121)
    plt.imshow(Y, cmap='RdBu_r', vmin=-0.8, vmax=0.8)
    plt.colorbar(shrink=0.5)
    mode_prefix = f"{config.mode_title} " if config and config.mode_title else ""
    plt.title('Full Connectome', fontsize=14)
    
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

def plot_distance_scatter(y_true, y_pred, distances, mode, config):
    """Generate distance-colored scatter plot"""
    plt.figure(figsize=config.plot_params['figsize'])
    
    # Compute correlation
    r = pearsonr(y_true, y_pred)[0]
    
    scatter = plt.scatter(y_true, y_pred, c=distances, cmap='viridis',
                         alpha=config.plot_params['alpha'], 
                         s=config.plot_params['point_size'])
    
    # Add line of best fit and reference lines
    z = np.polyfit(y_true, y_pred, 1)
    plt.plot(y_true, np.poly1d(z)(y_true), "r:", alpha=0.5, label=f'Line of best fit (r={r:.3f})')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    
    # Formatting
    plt.xlabel('True Values', fontsize=config.plot_params['font_size'])
    plt.ylabel('Predicted Values', fontsize=config.plot_params['font_size'])
    plt.title(f'{config.mode_title} Distance-Based Predictions')
    plt.legend()
    
    # Set equal aspect ratio and limits
    plt.gca().set_aspect('equal')
    lims = [-0.4, 1.0]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xticks(np.arange(-0.4, 1.2, 0.2))
    plt.yticks(np.arange(-0.4, 1.2, 0.2))
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Distance (mm)', fontsize=config.plot_params['font_size'])
    
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
    
    # Set equal aspect ratio and limits
    plt.gca().set_aspect('equal')
    lims = [-0.4, 1.0]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xticks(np.arange(-0.4, 1.2, 0.2))
    plt.yticks(np.arange(-0.4, 1.2, 0.2))
    
    plt.xlabel('True Values', fontsize=config.plot_params['font_size'])
    plt.ylabel('Predicted Values', fontsize=config.plot_params['font_size'])
    plt.title(f'{config.mode_title} Hemispheric Predictions')
    plt.legend(fontsize=config.plot_params['font_size']-2, markerscale=5)
    
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

    try:
        # For each pair of regions in test set
        for idx, (i, j) in enumerate(combinations(indices, 2)):
            # Skip if indices are out of bounds
            if i >= len(network_labels) or j >= len(network_labels):
                continue
                
            net_i = network_labels[i]
            net_j = network_labels[j]
            
            # Only store if regions are in same network
            if net_i == net_j and net_i in network_colors:
                # Check for index bounds
                if 2*idx < len(y_true) and 2*idx < len(y_pred):
                    network_true[net_i].append(y_true[2*idx])
                    network_pred[net_i].append(y_pred[2*idx])

        # Plot each network's correlations
        for network in network_colors:
            if len(network_true[network]) > 0:
                true_vals = np.array(network_true[network])
                pred_vals = np.array(network_pred[network])
                
                try:
                    # Calculate correlation, handle case where correlation fails
                    corr = pearsonr(true_vals, pred_vals)[0]
                    corr_str = f' (r={corr:.3f})'
                except:
                    corr_str = ''
                
                # Create scatter plot
                plt.scatter(true_vals, pred_vals,
                          c=network_colors[network], alpha=0.4, s=3,
                          label=f'{network}{corr_str}')

        # Add line of best fit and reference lines if there's data
        if len(y_true) > 0 and len(y_pred) > 0:
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
        
    except Exception as e:
        print(f"Warning: Error in plotting network scatter: {str(e)}")
        # Still create a basic plot even if there's an error
        plt.text(0.5, 0.5, 'Error plotting network data', 
                horizontalalignment='center', verticalalignment='center')
    
    plt.show()
    return plt.gcf()

def format_metrics_output(metrics, mode):
    """Format metrics into organized categories for clean printing"""
    # Define metric categories
    global_metrics = ['mse', 'mae', 'r2', 'pearson_r', 'spearman_r', 'geodesic_distance', 'accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    distance_metrics = ['short_r', 'mid_r', 'long_r'] 
    hemispheric_metrics = ['left_hemi_r', 'right_hemi_r', 'inter_hemi_r']
    strength_metrics = ['strong_neg_r', 'weak_r', 'strong_pos_r']
    graph_theory_metrics = ['geodesic_distance']
    network_names = ['Cont', 'Default', 'SalVentAttn', 'Limbic', 'DorsAttn', 'SomMot', 'Vis', 'Subcortical', 'Cerebellum']
    
    print(f"{mode.upper()} METRICS")
    print(f"{'='*50}")

    # Global Metrics
    global_found = {k: v for k, v in metrics.items() if k in global_metrics and not np.isnan(v)}
    if global_found:
        print("GLOBAL:", end=" ")
        metrics_str = []
        for k, v in global_found.items():
            if k in ['mse', 'mae']:
                metrics_str.append(f"{k}={v:.6f}")
            else:
                metrics_str.append(f"{k}={v:.4f}")
        print(", ".join(metrics_str))

    # Distance, Hemispheric and Strength metrics
    distance_found = {k: v for k, v in metrics.items() if k in distance_metrics and not np.isnan(v)}
    if distance_found:
        print("DISTANCE-BASED:", end=" ")
        print(", ".join([f"{k.replace('_r','')}={v:.4f}" for k,v in distance_found.items()]))
    
    hemispheric_found = {k: v for k, v in metrics.items() if k in hemispheric_metrics and not np.isnan(v)}
    if hemispheric_found:
        print("HEMISPHERIC:", end=" ")
        labels = {"left_hemi": "left", "right_hemi": "right", "inter_hemi": "inter"}
        print(", ".join([f"{labels[k.replace('_r','')]}={v:.4f}" for k,v in hemispheric_found.items()]))
    
    strength_found = {k: v for k, v in metrics.items() if k in strength_metrics and not np.isnan(v)}
    if strength_found:
        print("CONNECTION STRENGTH:", end=" ")
        labels = {"strong_neg": "neg", "weak": "weak", "strong_pos": "pos"}
        print(", ".join([f"{labels[k.replace('_r','')]}={v:.4f}" for k,v in strength_found.items()]))

    # Network metrics
    intra_metrics = {k: v for k, v in metrics.items() if k.startswith('intra_network_') and not np.isnan(v)}
    inter_metrics = {k: v for k, v in metrics.items() if k.startswith('inter_network_') and not np.isnan(v)}
    
    if intra_metrics or inter_metrics:
        print("NETWORK CORRELATIONS:")
        print("  NETWORK      INTRA      INTER")
        print("  " + "-" * 30)
        for net in network_names:
            intra_val = metrics.get(f'intra_network_{net}_r')
            inter_val = metrics.get(f'inter_network_{net}_r')
            if not (np.isnan(intra_val) if intra_val is not None else True) or \
               not (np.isnan(inter_val) if inter_val is not None else True):
                intra_str = f"{intra_val:.4f}" if intra_val is not None and not np.isnan(intra_val) else "N/A"
                inter_str = f"{inter_val:.4f}" if inter_val is not None and not np.isnan(inter_val) else "N/A"
                print(f"  {net:<10}  {intra_str:>8}  {inter_str:>8}")

    print(f"{'='*50}" + '\n')