from env.imports import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from ipywidgets import interact
import ipywidgets as widgets
import imageio


def get_gene_expression_colors(X, valid_genes=None, gene_name=None):
    """
    Get color values based on gene expression, handling NaN values.
    
    Parameters:
    -----------
    X : array-like, shape (n_regions, n_genes)
        Gene expression matrix
    valid_genes : list, optional
        List of valid gene names corresponding to X columns
    gene_name : str, optional
        Specific gene to use for coloring. If None, randomly select one
    Returns:
    --------
    expression_values : array
        Normalized gene expression values
    gene_used : str
        Name of the gene used for coloring
    """
    if valid_genes is not None:
        if gene_name is not None:
            if gene_name in valid_genes:
                gene_idx = valid_genes.index(gene_name)
            else:
                raise ValueError(f"Gene {gene_name} not found in valid_genes")
        else:
            gene_idx = np.random.randint(len(valid_genes))
            gene_name = valid_genes[gene_idx]
        
        expression_values = X[:, gene_idx]
        
        # Handle NaN values by setting them to the minimum of non-NaN values
        nan_mask = np.isnan(expression_values)
        if np.any(nan_mask):
            non_nan_min = np.nanmin(expression_values)
            expression_values[nan_mask] = non_nan_min
        
        # Normalize non-NaN values to [0, 1]
        expression_values = (expression_values - np.nanmin(expression_values)) / \
                          (np.nanmax(expression_values) - np.nanmin(expression_values))
        
        return expression_values, gene_name
    return None, None

def create_color_gradient(base_color, expression_values):
    """
    Create a color gradient based on expression values.
    
    Parameters:
    -----------
    base_color : str or tuple
        Base color to create gradient from
    expression_values : array
        Normalized expression values [0, 1]
        
    Returns:
    --------
    colors : array
        Array of RGBA colors
    """
    # Convert base color to RGB
    base_rgb = plt.cm.colors.to_rgb(base_color)
    
    # Create color array
    colors = np.zeros((len(expression_values), 4))
    for i, val in enumerate(expression_values):
        # Interpolate between white and base color based on expression
        colors[i, :3] = tuple(1 - (1 - c) * val for c in base_rgb)  # Light to dark
        colors[i, 3] = 0.9  # Alpha value
        
    return colors

def visualize_splits_3d(splits, coords, Y=None, X=None, edge_threshold=0.5, valid_genes=None, gene_name=None,
                       title_prefix="CV Split", save_gif=False):
    """
    General helper function to visualize train/test splits in 3D space with weighted connectivity edges
    and optional gene expression coloring.
    
    Parameters:
    -----------
    splits : iterator
        Iterator yielding (train_indices, test_indices) tuples
    coords : array-like, shape (n_samples, 3)
        3D coordinates for each region/point
    Y : array-like, shape (n_samples, n_samples), optional
        Connectivity matrix. If provided, edges will be drawn between connected regions
        with width and opacity proportional to connection strength
    X : array-like, optional, shape (n_regions, n_genes)
        Gene expression matrix
    valid_genes : list, optional
        List of valid gene names corresponding to X columns
    gene_name : str, optional
        Specific gene to use for coloring. If None, randomly select one
    title_prefix : str, optional
        Prefix for the plot title to indicate split type
    edge_threshold : float, optional
        Threshold for displaying edges (only edges with weight > threshold are shown)
        
    Returns:
    --------
    None
        Displays matplotlib plots for each fold
    """
    # Convert splits to list to allow multiple iterations
    splits = list(splits)
    
    # Store figures if saving GIF
    if save_gif:
        # Generate gif path based on split type and number of splits
        n_splits = len(splits)  # Count number of splits
        gene_suffix = "_gene" if valid_genes is not None else ""
        gif_path = f"cv_split_{title_prefix.lower()}_{n_splits}{gene_suffix}.gif"
        figures = []
    
    # Get gene expression colors if X is provided
    expression_values, gene_used = None, None
    if X is not None and (valid_genes is not None or gene_name is not None):
        expression_values, gene_used = get_gene_expression_colors(X, valid_genes, gene_name)
        if gene_used:
            title_prefix = f"{title_prefix}, {gene_used} Expression"
    
    for fold_idx, (train_indices, test_indices) in enumerate(splits, 1):
        fig = plt.figure(figsize=(14, 12), dpi=300)
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
        
        # Plot edges if connectivity matrix is provided
        if Y is not None:
            # Get all pairs of indices where connectivity > threshold
            connected_pairs = np.where(Y > edge_threshold)
            
            # Get the range of connectivity values for normalization
            max_weight = np.max(Y[Y > edge_threshold])
            min_weight = np.min(Y[Y > edge_threshold])
            
            # Draw edges with different colors based on train/test membership
            for i, j in zip(*connected_pairs):
                if i < j:  # Only draw each edge once
                    start = coords[i]
                    end = coords[j]
                    
                    # Normalize weight for visual properties
                    weight = Y[i, j]
                    norm_weight = (weight - min_weight) / (max_weight - min_weight)
                    
                    # Base alpha and width on connection strength
                    base_alpha = norm_weight * 0.5  # Scale alpha with weight
                    base_width = norm_weight * 2    # Scale width with weight
                    
                    # Only draw edges within train or within test sets
                    if i in train_indices and j in train_indices:
                        color = '#1f77b4'  # Darker blue
                        alpha = base_alpha * 0.8
                        width = base_width * 1.0
                        ax.plot([start[0], end[0]], 
                               [start[1], end[1]], 
                               [start[2], end[2]], 
                               color=color, alpha=alpha, linewidth=width)
                    elif i in test_indices and j in test_indices:
                        color = 'orange'
                        alpha = base_alpha * 1.0
                        width = base_width * 1.5
                        ax.plot([start[0], end[0]], 
                               [start[1], end[1]], 
                               [start[2], end[2]], 
                               color=color, alpha=alpha, linewidth=width)
        
        # Plot points with gene expression coloring if available
        if expression_values is not None:
            train_colors = create_color_gradient('blue', expression_values[train_indices])
            test_colors = create_color_gradient('orange', expression_values[test_indices])
            
            train_scatter = ax.scatter(coords[train_indices, 0], 
                                     coords[train_indices, 1], 
                                     coords[train_indices, 2], 
                                     c=train_colors, label='Train', 
                                     s=50, edgecolor='gray', linewidth=0.5)
            
            test_scatter = ax.scatter(coords[test_indices, 0], 
                                    coords[test_indices, 1], 
                                    coords[test_indices, 2], 
                                    c=test_colors, label='Test', 
                                    s=50, edgecolor='gray', linewidth=0.5)
        else:
            # Plot points with smaller size
            train_scatter = ax.scatter(coords[train_indices, 0], 
                                     coords[train_indices, 1], 
                                     coords[train_indices, 2], 
                                     c='blue', label='Train', alpha=0.9,
                                     s=50, edgecolor='gray', linewidth=0.5)
            
            test_scatter = ax.scatter(coords[test_indices, 0], 
                                    coords[test_indices, 1], 
                                    coords[test_indices, 2], 
                                    c='orange', label='Test', alpha=0.9,
                                    s=50, edgecolor='gray', linewidth=0.5)
        
        # Increase font sizes for axes labels
        ax.set_xlabel('X (Lateral)', fontsize=20, labelpad=5)
        ax.set_ylabel('Y (Posterior-Anterior)', fontsize=20, labelpad=5)
        ax.set_zlabel('Z (Dorsal-Ventral)', fontsize=20, labelpad=5)
        
        # Increase title font size substantially
        plt.suptitle(f'Fold {fold_idx}, {title_prefix} Visualization', 
                    fontsize=24, y=0.1)
        
        # Increase tick label font size
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Create legend with larger font and adjusted position
        legend_elements = [
            plt.scatter([], [], c='blue', alpha=0.9, s=50),
            plt.scatter([], [], c='orange', alpha=0.9, s=50)
        ]
        legend_labels = ['Train Regions', 'Test Regions']
        
        if Y is not None:
            legend_elements.extend([
                Line2D([0], [0], color='#1f77b4', alpha=0.6, linewidth=2),
                Line2D([0], [0], color='orange', alpha=0.6, linewidth=2)
            ])
            legend_labels.extend(['Train Connections', 
                                'Test Connections'])
        
        # Adjust legend font size and position
        ax.legend(legend_elements, legend_labels, 
                 fontsize=18, loc='upper right',
                 bbox_to_anchor=(0.97, 0.85))  # Move legend closer to plot
        
        # Style adjustments
        ax.view_init(elev=20, azim=45)
        ax.grid(False)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Remove tight_layout and use figure adjustments instead
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        
        if save_gif:
            # Convert figure to image
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            figures.append(image)
            plt.close()
        else:
            plt.show()

    if save_gif and figures:  # Only try to save if we have figures
        # Create the glass directory if it doesn't exist
        os.makedirs('glass', exist_ok=True)
        # Save as GIF
        imageio.mimsave(f"glass/{gif_path}", figures, fps=1)
        print(f"GIF saved to glass/{gif_path}")

def visualize_3d(X, Y, coords, edge_threshold=0.5, valid_genes=None, gene_name=None):
    """
    Visualize brain network in 3D space with optional gene expression coloring.
    
    Parameters:
    -----------
    X : array-like, shape (n_regions, n_genes)
        Gene expression matrix
    Y : array-like, shape (n_regions, n_regions)
        Connectivity matrix
    coords : array-like, shape (n_regions, 3)
        3D coordinates for each region
    edge_threshold : float, optional
        Threshold for displaying edges (only edges with weight > threshold are shown)
    valid_genes : list, optional
        List of valid gene names corresponding to X columns
    gene_name : str, optional
        Specific gene to use for coloring. If None and valid_genes provided, randomly select one
    """
    # Get gene expression colors if provided
    expression_values, gene_used = None, None
    title = "Brain Network Visualization"
    if X is not None and (valid_genes is not None or gene_name is not None):
        expression_values, gene_used = get_gene_expression_colors(X, valid_genes, gene_name)
        if gene_used:
            title = f"{gene_used} Expression"
    
    # Create figure with increased size and left margin
    fig = plt.figure(figsize=(40, 12), dpi=300, constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot edges
    if Y is not None:
        connected_pairs = np.where(Y > edge_threshold)
        max_weight = np.max(Y[Y > edge_threshold])
        min_weight = np.min(Y[Y > edge_threshold])
        
        for i, j in zip(*connected_pairs):
            if i < j:  # Only draw each edge once
                start = coords[i]
                end = coords[j]
                
                # Normalize weight for visual properties
                weight = Y[i, j]
                norm_weight = (weight - min_weight) / (max_weight - min_weight)
                
                # Base alpha and width on connection strength
                alpha = norm_weight * 0.5
                width = norm_weight * 2
                
                # Draw edge
                ax.plot([start[0], end[0]], 
                       [start[1], end[1]], 
                       [start[2], end[2]], 
                       color='#1f77b4', alpha=alpha, linewidth=width)
    
    # Plot nodes
    if expression_values is not None:
        # Use plasma colormap for gene expression
        colors = plt.cm.plasma(expression_values)
        scatter = ax.scatter(coords[:, 0], 
                           coords[:, 1], 
                           coords[:, 2], 
                           c=expression_values,
                           cmap='plasma',
                           s=80,  # Increased marker size
                           edgecolor='white', 
                           linewidth=0.5)
        
        # Add colorbar for gene expression - smaller size
        cbar = plt.colorbar(scatter, shrink=0.4)  # Add shrink parameter
        cbar.set_label(f'{gene_used} Expression', fontsize=20)
        cbar.ax.tick_params(labelsize=18)
    else:
        # Default solid blue coloring if no gene expression
        scatter = ax.scatter(coords[:, 0], 
                           coords[:, 1], 
                           coords[:, 2], 
                           c='#1f77b4',  # Single solid color
                           s=80,  # Increased marker size
                           alpha=0.9,
                           edgecolor='white', 
                           linewidth=0.5)
    
    # Add labels and title with increased font sizes
    ax.set_xlabel('X (Coronal)', fontsize=20, labelpad=5)
    ax.set_ylabel('Y (Sagittal)', fontsize=20, labelpad=5)
    ax.set_zlabel('Z (Axial)', fontsize=20, labelpad=5)
    
    # Move title to bottom of plot with increased size
    # ax.set_title(title, fontsize=18, pad=0, y=0.05)
    
    # Adjust tick label size
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Create legend for edges with larger font
    if Y is not None:
        legend_elements = [
            Line2D([0], [0], color='#1f77b4', alpha=0.6, linewidth=3,
                  label='Connections')
        ]
        legend_labels = ['Connections']
        ax.legend(legend_elements, legend_labels, fontsize=16, loc='upper right',
                 bbox_to_anchor=(.95, .85))
    
    # Style adjustments
    ax.view_init(elev=20, azim=45)
    ax.grid(False)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Remove tight_layout and use figure adjustments instead
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    
    plt.show()


def plot_connectome(Y):
    """
    Function to plot the connectome without network labels.
    
    Parameters:
    Y (ndarray): Connectivity matrix.
    """
    # Visualize the matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(Y, cmap='Reds', vmin=0, vmax=1)
    
    # Add colorbar
    plt.colorbar(cax, shrink=0.8)
    
    #plt.title('Connectivity Matrix', fontsize=16)
    plt.xlabel('Regions', fontsize=14)
    plt.ylabel('Regions', fontsize=14)
    plt.show()


def plot_transcriptome(X):
    """
    Function to plot the transcriptome data.
    
    Parameters:
    X (ndarray): Transcriptome data matrix.
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    cax = ax.imshow(X, aspect=40, cmap='viridis')
    
    # Add colorbar
    plt.colorbar(cax, shrink=0.5)
    
    #plt.title('Transcriptome Data', fontsize=16)
    plt.xlabel('Genes', fontsize=14)
    plt.ylabel('Regions', fontsize=14)
    plt.show()

def plot_connectome_with_labels(Y, labels):
    """
    Function to plot the connectome with network labels.
    
    Parameters:
    Y (ndarray): Connectivity matrix.
    labels (list): List of region labels with network information.
    """
    networks = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'Subcortical']
    
    # Create a new ordering of nodes based on network
    new_order = []
    network_indices = {network: [] for network in networks}
    
    for idx, label in enumerate(labels):
        for network in networks:
            if network in label:
                network_indices[network].append(idx)
                break
        else:  # Handle Subcortical case
            if label.startswith(('L', 'R')):
                network_indices['Subcortical'].append(idx)
    
    for network in networks:
        new_order.extend(network_indices[network])
    
    # Map the original indices to the new ordering
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(new_order)}
    
    # Reorder the matrix
    Y_reordered = Y[np.ix_(new_order, new_order)]
    
    # Visualize the reordered matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(Y_reordered, cmap='viridis', vmin=0, vmax=np.max(Y))

    # Add red boxes around networks and place labels to the right
    start = 0
    for network in networks:
        size = len(network_indices[network])
        if size > 0:
            rect = patches.Rectangle((start, start), size, size, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(start + size + 2, start + size / 2, network, ha='left', va='center', fontsize=16, color='yellow')
            start += size
    
    #plt.title('Reordered Connectivity Matrix by Network Structure', fontsize=16)
    plt.xlabel('Regions', fontsize=14)
    plt.ylabel('Regions', fontsize=14)
    plt.show()





