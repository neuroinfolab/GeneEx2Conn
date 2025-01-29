# Gene2Conn/cv_split/cv_split.py

from imports import *
from mpl_toolkits.mplot3d import Axes3D

def visualize_splits_3d(splits, coords, title_prefix="CV Split"):
    """
    General helper function to visualize train/test splits in 3D space.
    
    Parameters:
    -----------
    splits : iterator
        Iterator yielding (train_indices, test_indices) tuples
    coords : array-like, shape (n_samples, 3)
        3D coordinates for each region/point
    title_prefix : str, optional
        Prefix for the plot title to indicate split type
        
    Returns:
    --------
    None
        Displays matplotlib plots for each fold
    """
    for fold_idx, (train_indices, test_indices) in enumerate(splits, 1):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot training points
        ax.scatter(coords[train_indices, 0], 
                  coords[train_indices, 1], 
                  coords[train_indices, 2], 
                  c='blue', label='Train', alpha=0.6)
        
        # Plot test points
        ax.scatter(coords[test_indices, 0], 
                  coords[test_indices, 1], 
                  coords[test_indices, 2], 
                  c='orange', label='Test', alpha=0.6)
        
        # Add labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Fold {fold_idx} - {title_prefix} Visualization')
        ax.legend()
        
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

    def visualize_splits_3d(self, coords):
        """Display each fold's train/test split in 3D space."""
        visualize_splits_3d(self.split(), coords, "Random Split")


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

    def visualize_splits_3d(self, coords):
        """Display each fold's train/test split in 3D space."""
        visualize_splits_3d(self.split(), coords, "Community Split")


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

    def visualize_splits_3d(self):
        """Display each fold's train/test split in 3D space."""
        visualize_splits_3d(self.split(), self.coords, "Spatial Split")


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

