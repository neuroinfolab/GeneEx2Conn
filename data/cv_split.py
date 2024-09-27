# Gene2Conn/cv_split/cv_split.py

from imports import *

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
            
            print(held_out_network)
            # Combine all the training indices except for the held-out network
            train_indices = [index for network, indices in self.networks.items() if network != held_out_network for index in indices]

            # Append the training and testing indices for this fold to the list of folds
            network_folds.append((train_indices, test_indices))
            
            print(held_out_network)

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


class ModularNetworkSplit:
    def __init__(self, data, networks):
        self.data = data
        self.networks = networks
    
    def get_splits(self):
        # Implement the logic for network-based splits
        pass

