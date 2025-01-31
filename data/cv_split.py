# Gene2Conn/cv_split/cv_split.py

from imports import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D  # Add this import
from ipywidgets import interact
import ipywidgets as widgets


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
                       title_prefix="CV Split"):
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
                        width = base_width * 1.2
                        ax.plot([start[0], end[0]], 
                               [start[1], end[1]], 
                               [start[2], end[2]], 
                               color=color, alpha=alpha, linewidth=width)
                    elif i in test_indices and j in test_indices:
                        color = 'orange'
                        alpha = base_alpha * 0.9
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
        ax.set_xlabel('X (Coronal)', fontsize=16, labelpad=5)
        ax.set_ylabel('Y (Sagittal)', fontsize=16, labelpad=5)
        ax.set_zlabel('Z (Axial)', fontsize=16, labelpad=5)
        
        # Increase title font size substantially
        plt.suptitle(f'Fold {fold_idx}, {title_prefix} Visualization', 
                    fontsize=18, y=0.05)
        
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
                 fontsize=16, loc='upper right',
                 bbox_to_anchor=(0.95, 0.85))  # Move legend closer to plot
        
        # Style adjustments
        ax.view_init(elev=20, azim=45)
        ax.grid(False)
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Remove tight_layout and use figure adjustments instead
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        
        plt.show()

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
        cbar.set_label(f'{gene_used} Expression', fontsize=16)
        cbar.ax.tick_params(labelsize=16)
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
    ax.set_xlabel('X (Coronal)', fontsize=16, labelpad=5)
    ax.set_ylabel('Y (Sagittal)', fontsize=16, labelpad=5)
    ax.set_zlabel('Z (Axial)', fontsize=16, labelpad=5)
    
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

class RandomCVSplit:
    """
    Cross-validation class for random or spatial-split testing.

    Parameters:
    labels (list): List of region labels with network information.
    """
    
    def __init__(self, X, Y, num_splits, shuffled=True, use_random_state=True,random_seed=42):
        self.X = X
        self.Y = Y
        self.num_splits = num_splits
        self.random_seed = random_seed
        
        self.shuffled = shuffled
        self.use_random_state = use_random_state
        self.kf = self.define_splits()
        self.folds = self.create_folds()
        self.networks = self.create_networks()

    def define_splits(self):
        if self.shuffled:
            if self.use_random_state:
                kf = KFold(n_splits=self.num_splits, shuffle=True, random_state=self.random_seed)
            else:
                kf = KFold(n_splits=self.num_splits, shuffle=True)
        else:
            kf = KFold(n_splits=self.num_splits, shuffle=False)
        return kf
    
    def get_splits(self):
        return self.kf.split(self.X, self.Y)

    def create_folds(self):
        folds = []
        for train_indices, test_indices in self.get_splits():
            X_train, X_test = self.X[train_indices], self.X[test_indices]
            Y_train, Y_test = self.Y[train_indices], self.Y[test_indices]
            folds.append((X_train, X_test, Y_train, Y_test))
        return folds
    
    def create_networks(self):
        networks = {}
        for fold_idx, (train_indices, test_indices) in enumerate(self.get_splits(), 1):
            networks[str(fold_idx)] = test_indices
        return networks

    def split(self, X=None, y=None, groups=None):
        for train_indices, test_indices in self.get_splits():
            yield train_indices, test_indices
            
    def display_splits(self):
        for train_index, test_index in self.get_splits():
            print("TRAIN:", train_index, "TEST:", test_index)

    def visualize_splits_3d(self, coords, edge_threshold=0.5, valid_genes=None, gene_name=None):
        """Display each fold's train/test split in 3D space."""
        visualize_splits_3d(self.get_splits(), 
                            coords, self.Y, self.X, 
                            edge_threshold, valid_genes, gene_name,
                            "Random Split")


class SchaeferCVSplit(BaseCrossValidator):
    """
    Custom cross-validation class for held-out subnetwork testing.

    Parameters:
    labels (list): List of region labels with network information.
    """
    
    def __init__(self, omit_subcortical=False):
        fc_combined_mat_schaef_100, fc_combined_labels_schaef_100 = load_fc_as_one(parcellation='schaefer_100')
        
        self.labels = fc_combined_labels_schaef_100
        self.networks = {}
        self.omit_subcortical = omit_subcortical
        self.folds = self.create_folds(self.labels)
    
    def create_folds(self, labels):
        """
        Create folds based on the Schaefer network each region is a part of.

        Parameters:
        labels (list): List of region labels with network information.

        Returns:
        list of tuples: Each tuple contains (train_indices, test_indices).
        """
        # Initialize a dictionary to group regions by network
        # networks = {}
        
        # Iterate through the region labels and group them by network
        for index, label in enumerate(labels):
            # Extract the network name from the label
            network_name = label.split('_')[2] if '7Networks_' in label else 'Subcortical'

            # Add the index to the appropriate network list in the dictionary
            if network_name not in self.networks:
                self.networks[network_name] = []
            self.networks[network_name].append(index)

        # Initialize a list to store the folds (training and testing indices)
        network_folds = []

        # Iterate through each network
        for held_out_network, test_indices in self.networks.items():
            if self.omit_subcortical and held_out_network == 'Subcortical':
                continue 
            
            # Combine all the training indices except for the held-out network
            train_indices = [index for network, indices in self.networks.items() if network != held_out_network for index in indices]

            # Append the training and testing indices for this fold to the list of folds
            network_folds.append((train_indices, test_indices))

        return network_folds
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of cross-validation splits."""
        return len(self.folds)
    
    def split(self, X=None, y=None, groups=None):
        """
        Generate indices to split data into training and testing sets.

        Returns:
        Generator yielding tuples of (train_indices, test_indices).
        """
        for train_indices, test_indices in self.folds:
            yield train_indices, test_indices

    def print_folds_with_networks(self):
        """Return folds with held-out network."""
        # Iterate through each network
        for held_out_network, test_indices in self.networks.items():
            # Combine all the training indices except for the held-out network
            train_indices = [index for network, indices in self.networks.items() if network != held_out_network for index in indices]
            
            print("HELD OUT NETWORK:", held_out_network)
            print("TRAIN:", train_indices, "TEST:", test_indices)
            

class CommunityCVSplit(BaseCrossValidator):
    """
    Custom cross-validation class based on Louvain community detection algorithm applied to the connectome.
    
    Parameters:
    X (array-like): Feature data.
    Y (array-like): Connectivity matrix.
    resolution (float): Resolution parameter for Louvain community detection.

    reference: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html#networkx.algorithms.community.louvain.louvain_communities
    """
    
    def __init__(self, X, Y, resolution=1.0, random_seed=42):
        self.X = X
        self.Y = Y
        self.resolution = resolution
        self.random_seed=random_seed
        self.connectome_net = self.create_connectome_net()
        self.communities = self.detect_communities()
        self.networks = self.create_networks()
        self.folds = self.create_folds()

    def create_connectome_net(self):
        """Create a NetworkX graph from the connectivity matrix."""
        connectome_net = nx.Graph(incoming_graph_data=self.Y)
        return connectome_net

    def detect_communities(self):
        """Detect communities using the Louvain community detection algorithm."""
        communities = nx.community.louvain_communities(self.connectome_net, seed=self.random_seed, resolution=self.resolution)
        return communities

    def create_networks(self):
        """Create a dictionary of networks based on the detected communities."""
        networks = {str(i+1): list(community) for i, community in enumerate(self.communities)}
        return networks

    def create_folds(self):
        """Create CV folds based on the network information."""
        folds = []
        for i, test_indices in enumerate(self.communities):
            train_indices = np.concatenate([list(self.communities[j]) for j in range(len(self.communities)) if j != i])
            test_indices = list(test_indices)  # Convert set to list
            folds.append((self.X[train_indices], self.X[test_indices], self.Y[train_indices], self.Y[test_indices]))
        return folds

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of cross-validation splits."""
        return len(self.folds)

    def split(self, X=None, y=None, groups=None):
        """Generate indices to split data into training and testing sets."""
        for test_indices in self.communities:
            train_indices = [idx for community in self.communities if community != test_indices for idx in community]
            yield train_indices, list(test_indices)

    def display_communities(self):
        """Visualize the reordered connectivity matrix with community structure."""
        # Create a new ordering of nodes based on Louvain community detection
        new_order = []
        for community in self.communities:
            new_order.extend(community)

        # Map the original indices to the new ordering
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(new_order)}

        # Reorder the matrix
        Y_reordered = self.Y[np.ix_(new_order, new_order)]

        # Visualize the reordered matrix
        fig, ax = plt.subplots()
        cax = ax.imshow(Y_reordered, cmap='viridis')

        # Add colorbar
        plt.colorbar(cax)

        # Add red boxes around communities
        start = 0
        for community in self.communities:
            size = len(community)
            rect = patches.Rectangle((start, start), size, size, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            start += size

        plt.title('Reordered Connectivity Matrix by Community Structure')
        plt.show()

    def visualize_splits_3d(self, coords, edge_threshold=0.5, valid_genes=None, gene_name=None):
        """Display each fold's train/test split in 3D space."""
        visualize_splits_3d(self.split(), 
                            coords, self.Y, self.X, 
                            edge_threshold, valid_genes, gene_name,
                            "Community Split")


class SpatialCVSplit(BaseCrossValidator):
    """
    Custom cross-validation class that creates spatially coherent networks,
    then uses these for train/test splits.
    
    Parameters:
    X (array): Input features
    Y (array): Target values 
    coords (array): 3D coordinates for each region
    num_splits (int): Number of networks/splits to create
    random_seed (int): Random seed for reproducibility
    """
    
    def __init__(self, X, Y, coords, num_splits, random_seed=42):
        self.X = X
        self.Y = Y
        self.coords = coords
        self.num_splits = num_splits
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        
        # Create networks first, then generate splits from them
        self.networks = self.create_spatial_networks()
        self.splits = self.create_splits_from_networks()
        self.folds = self.create_folds()

    def create_spatial_networks(self):
        """
        Create spatially coherent networks by iteratively selecting seed points
        and their closest unassigned neighbors.
        """
        n_regions = len(self.X)
        regions_per_network = n_regions // self.num_splits
        remaining_regions = set(range(n_regions))
        networks = {}
        
        for network_idx in range(self.num_splits):
            # Select random seed from remaining regions
            if remaining_regions:
                seed_idx = self.rng.choice(list(remaining_regions))
            else:
                print(f"Warning: No remaining regions for network {network_idx + 1}")
                break
                
            # Calculate distances from seed to all remaining points
            seed_coords = self.coords[seed_idx]
            distances = {}
            for idx in remaining_regions:
                dist = np.sqrt(np.sum((self.coords[idx] - seed_coords)**2))
                distances[idx] = dist
            
            # Sort remaining regions by distance to seed
            sorted_regions = sorted(distances.items(), key=lambda x: x[1])
            
            # Determine number of regions to assign to this network
            if network_idx == self.num_splits - 1:
                # Last network gets all remaining regions
                n_to_assign = len(remaining_regions)
            else:
                n_to_assign = min(regions_per_network, len(remaining_regions))
            
            # Assign closest n_to_assign regions to this network
            network_regions = [sorted_regions[i][0] for i in range(n_to_assign)]
            networks[str(network_idx + 1)] = network_regions
            
            # Update remaining regions
            remaining_regions -= set(network_regions)
            
        # Verify coverage
        all_assigned = set().union(*[set(regions) for regions in networks.values()])
        coverage = len(all_assigned) / n_regions * 100
        network_sizes = [len(regions) for regions in networks.values()]
        
        print(f"Network coverage: {coverage:.1f}% of regions")
        print(f"Network sizes: {network_sizes}")
        
        return networks

    def create_splits_from_networks(self):
        """Create train/test splits based on the networks."""
        splits = []
        for network_idx in range(self.num_splits):
            # Test set is the current network
            test_indices = np.array(self.networks[str(network_idx + 1)])
            
            # Train set is all other networks combined
            train_indices = np.array([
                idx for net_id, regions in self.networks.items()
                if net_id != str(network_idx + 1)
                for idx in regions
            ])
            
            splits.append((train_indices, test_indices))
            
        return splits

    def create_folds(self):
        """Create CV folds from the splits."""
        folds = []
        for train_indices, test_indices in self.splits:
            folds.append((self.X[train_indices], self.X[test_indices],
                         self.Y[train_indices], self.Y[test_indices]))
        return folds

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of cross-validation splits."""
        return len(self.splits)

    def split(self, X=None, y=None, groups=None):
        """Generate indices to split data into training and test sets."""
        for train_indices, test_indices in self.splits:
            yield train_indices, test_indices

    def display_splits(self):
        """Display the train/test indices for each fold"""
        for fold_idx, (train_indices, test_indices) in enumerate(self.splits, 1):
            print(f"Fold {fold_idx}:")
            print("TRAIN:", train_indices)
            print("TEST:", test_indices)
            print()

    def visualize_splits_3d(self, edge_threshold=0.5, valid_genes=None, gene_name=None):
        """Display each fold's train/test split in 3D space."""
        visualize_splits_3d(self.split(), 
                            self.coords, self.Y, self.X, 
                            edge_threshold, valid_genes, gene_name,
                            "Spatial Split")


class SubnetworkCVSplit(BaseCrossValidator):
    """
    Custom cross-validation class for held-out subnetwork testing. 
    Can be used for inner and outer splits. This depends on input network_dict. 

    Parameters:
    indices (list): List of training indices.
    network_dict (dict): Dictionary mapping indices to their respective subnetworks.
    """

    def __init__(self, indices, network_dict):
        self.indices = indices
        self.network_dict = network_dict
        self.folds = self.create_folds()

    def create_folds(self):
        """Create CV folds based on the network information."""
        network_folds = []

        for held_out_network, test_indices in self.network_dict.items():
            train_indices = []
            for network, indices in self.network_dict.items():
                if network != held_out_network:
                    train_indices.extend(indices)
            
            network_folds.append((train_indices, test_indices))
            print(held_out_network)

        return network_folds

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of cross-validation splits."""
        return len(self.folds)

    def split(self, X=None, y=None, groups=None):
        """Generate indices to split data into training and testing sets."""
        for train_indices, test_indices in self.folds:
            yield train_indices, test_indices

    def visualize_splits_3d(self, coords):
        """Display each fold's train/test split in 3D space."""
        visualize_splits_3d(self.split(), coords, "Subnetwork Split")

