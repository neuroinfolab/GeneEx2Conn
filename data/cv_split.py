from env.imports import *

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


class SchaeferCVSplit(BaseCrossValidator):
    """
    Custom cross-validation class that splits data based on Schaefer networks.
    
    Parameters:
    X (array): Input features
    Y (array): Target values
    network_labels (array): Network labels for each region
    omit_subcortical (bool): Whether to exclude subcortical regions from splits
    """
    
    def __init__(self, X, Y, network_labels, omit_subcortical=False):
        self.X = X
        self.Y = Y
        self.network_labels = network_labels
        self.omit_subcortical = omit_subcortical
        
        # Create networks first, then generate splits from them
        self.networks = self.create_schaefer_networks()
        self.splits = self.create_splits_from_networks()
        self.folds = self.create_folds()

    def create_schaefer_networks(self):
        """Create networks based on Schaefer parcellation labels."""
        networks = {}
        
        # Group regions by network
        for index, network_name in enumerate(self.network_labels):
            if self.omit_subcortical and network_name in ['Subcortical', 'Cerebellum']:
                continue
                
            if network_name not in networks:
                networks[network_name] = []
            networks[network_name].append(index)
            
        # Print network info
        network_sizes = {net: len(regions) for net, regions in networks.items()}
        n_regions = len(self.X)
        coverage = sum(network_sizes.values()) / n_regions * 100
        
        print(f"Network coverage: {coverage:.1f}% of regions")
        print(f"Network sizes: {network_sizes}")
        
        return networks

    def create_splits_from_networks(self):
        """Create train/test splits based on the networks."""
        splits = []
        for network_name, test_indices in self.networks.items():
            # Test set is the current network
            test_indices = np.array(test_indices)
            
            # Train set is all other networks combined
            train_indices = np.array([
                idx for net, regions in self.networks.items()
                if net != network_name
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
                            "Schaefer Split")

class LobeCVSplit(BaseCrossValidator):
    """
    Custom cross-validation class that splits data based on brain lobes.
    
    Parameters:
    X (array): Input features
    Y (array): Target values
    lobe_labels (array): Lobe labels for each region (Frontal, Temporal, Parietal, Occipital, Subcortex)
    """
    
    def __init__(self, X, Y, lobe_labels):
        self.X = X
        self.Y = Y
        self.lobe_labels = lobe_labels
        
        # Create networks first, then generate splits from them
        self.networks = self.create_lobe_networks()
        self.splits = self.create_splits_from_networks()
        self.folds = self.create_folds()

    def create_lobe_networks(self):
        """Create networks based on brain lobe labels."""
        networks = {}
        
        # Group regions by lobe
        for index, lobe_name in enumerate(self.lobe_labels):
            if lobe_name not in networks:
                networks[lobe_name] = []
            networks[lobe_name].append(index)
            
        # Print network info
        network_sizes = {net: len(regions) for net, regions in networks.items()}
        n_regions = len(self.X)
        coverage = sum(network_sizes.values()) / n_regions * 100
        
        print(f"Lobe coverage: {coverage:.1f}% of regions")
        print(f"Lobe sizes: {network_sizes}")
        
        return networks

    def create_splits_from_networks(self):
        """Create train/test splits based on the lobes."""
        splits = []
        for lobe_name, test_indices in self.networks.items():
            # Test set is the current lobe
            test_indices = np.array(test_indices)
            
            # Train set is all other lobes combined
            train_indices = np.array([
                idx for net, regions in self.networks.items()
                if net != lobe_name
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
                            "Lobe Split")
            

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

