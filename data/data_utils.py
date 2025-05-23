from env.imports import *
from torch.utils.data import WeightedRandomSampler, DataLoader, TensorDataset, Dataset


def reconstruct_connectome(Y, symmetric=True):
    """
    Reconstructs the full connectome matrix from a given input vector Y.

    Parameters:
    Y (numpy.ndarray): Input vector representing the connectome.
    symmetric (bool): Whether to make the connectome symmetric. Default is True.

    Returns:
    numpy.ndarray: Reconstructed connectome matrix, symmetric if specified.
    """
    num_regions = math.floor(np.sqrt(Y.shape[0]))

    Y_temp_upper = reconstruct_upper_triangle(Y[0::2], num_regions)
    column_of_zeros = np.zeros((num_regions, 1))
    row_of_zeros = np.zeros((1, num_regions + 1))
    matrix_with_column = np.concatenate((column_of_zeros, Y_temp_upper), axis=1)
    Y_temp_upper = np.concatenate((matrix_with_column, row_of_zeros), axis=0)

    Y_temp_lower = reconstruct_upper_triangle(Y[1::2], num_regions)
    Y_temp_lower = Y_temp_lower.T
    row_of_zeros = np.zeros((1, num_regions))
    column_of_zeros = np.zeros((num_regions + 1, 1))
    matrix_with_row = np.concatenate((row_of_zeros, Y_temp_lower), axis=0)
    Y_temp_lower = np.concatenate((matrix_with_row, column_of_zeros), axis=1)

    Y_connectome = Y_temp_upper + Y_temp_lower
    if symmetric:
        Y_connectome = make_symmetric(Y_connectome)
    
    return Y_connectome

def reconstruct_upper_triangle(vector, num_regions):
    """
    Reconstructs an upper triangular matrix from a given vector.

    Parameters:
    vector (numpy.ndarray): Input vector representing the upper triangle of a matrix.
    num_regions (int): Number of regions in the matrix.

    Returns:
    numpy.ndarray: Reconstructed upper triangular matrix.
    """
    matrix = np.zeros((num_regions, num_regions))
    vector_index = 0
    
    for i in range(num_regions):
        for j in range(i, num_regions):
            matrix[i, j] = vector[vector_index]
            vector_index += 1
            
    return matrix

def make_symmetric(matrix):
    """
    Ensures the matrix is symmetric by averaging the matrix with its transpose.

    Parameters:
    matrix (numpy.ndarray): Input matrix.

    Returns:
    numpy.ndarray: Symmetric matrix.
    """
    return (matrix + matrix.T) / 2


def expand_shared_matrices(X_train, X_train2, Y_train2, Y_train_feats1=np.nan, Y_train_feats2=np.nan, incl_conn=False):
    """
    Expands matrices X_train and X_train2 to create X_train_shared, and expands Y_train2 accordingly.
    Includes connectivity profiles as features accordingly.

    Parameters:
    X_train (numpy.ndarray): Training gene expression matrix.
    X_train2 (numpy.ndarray): Shared rectangular training matrix.
    Y_train2 (numpy.ndarray): Rectangular connectivity matrix.
    Y_train_feats1 (numpy.ndarray, optional): Connectivity features for X_train.
    Y_train_feats2 (numpy.ndarray, optional): Connectivity features for X_train2.
    incl_conn (bool, optional): Flag to include connectivity profiles.

    Returns:
    tuple: Expanded shared X and Y matrices.
    """
    train_size, num_genes = X_train.shape
    test_size = X_train2.shape[0]

    if not incl_conn:
        expanded_size = train_size * test_size * 2
        X_train_shared = np.empty((expanded_size, num_genes * 2))
        Y_expanded = np.empty(expanded_size)

        index = 0
        for i in range(train_size):
            for j in range(test_size):
                X_train_shared[index] = np.hstack((X_train[i], X_train2[j]))
                Y_expanded[index] = Y_train2[i, j]
                index += 1

                X_train_shared[index] = np.hstack((X_train2[j], X_train[i]))
                Y_expanded[index] = Y_train2[i, j]
                index += 1
    else:
        num_regions = Y_train2.shape[0]
        num_combinations = train_size * test_size * 2
        X_train_shared = np.empty((num_combinations, 2 * (num_genes + num_regions)))
        Y_expanded = np.empty(num_combinations)

        index = 0
        for i in range(train_size):
            for j in range(test_size):
                X_train_shared[index] = np.hstack([X_train[i], Y_train_feats1[i], X_train2[j], Y_train_feats2[j]])
                Y_expanded[index] = Y_train2[i, j]
                index += 1

                X_train_shared[index] = np.hstack([X_train2[j], Y_train_feats2[j], X_train[i], Y_train_feats1[i]])
                Y_expanded[index] = Y_train2[i, j]
                index += 1

    return X_train_shared, Y_expanded


def expand_X_symmetric(X):
    """
    Expands the X matrix symmetrically by combining features from pairs of regions.
    For each pair of regions, creates two rows by concatenating their features in both orders.

    Parameters:
    X (numpy.ndarray): Input matrix of gene expressions, shape (num_regions, num_genes)
                      where num_regions is the number of brain regions and 
                      num_genes is the number of gene features per region

    Returns:
    numpy.ndarray: Expanded symmetric matrix, shape (region pairs * 2, 2 * num_genes)
                  where region pairs = (num_regions choose 2)
                  Each row contains concatenated features from two regions
                  For each region pair (i,j), creates rows [features_i|features_j] and [features_j|features_i]
    """
    num_regions, num_genes = X.shape
    region_combinations = list(combinations(range(num_regions), 2))
    num_combinations = len(region_combinations)  # Equal to (num_regions * (num_regions-1))/2

    expanded_X = np.zeros((num_combinations * 2, 2 * num_genes))
    
    for i, (region1, region2) in enumerate(region_combinations):
        expanded_X[i * 2] = np.concatenate((X[region1], X[region2]))
        expanded_X[i * 2 + 1] = np.concatenate((X[region2], X[region1]))

    return expanded_X


def expand_Y_symmetric(Y):
    """
    Expands the Y matrix symmetrically by extracting pairwise connectivity values.

    Parameters:
    Y (numpy.ndarray): Input matrix of connectome values, shape (num_regions, num_regions)
                      where num_regions is the number of brain regions

    Returns:
    numpy.ndarray: Expanded symmetric vector, shape (region pairs * 2,)
                  where region pairs = (num_regions choose 2) = (num_regions * (num_regions-1))/2
                  For each region pair (i,j), contains both Y[i,j] and Y[j,i] values
                  Total length is num_regions * (num_regions-1)
    """
    num_regions = Y.shape[0]
    region_combinations = list(combinations(range(num_regions), 2))
    num_combinations = len(region_combinations)  # Equal to (num_regions * (num_regions-1))/2
    
    expanded_Y = np.zeros(num_combinations * 2)  # Length = num_regions * (num_regions-1)
    
    for i, (region1, region2) in enumerate(region_combinations):
        expanded_Y[i * 2] = Y[region1, region2]
        expanded_Y[i * 2 + 1] = Y[region2, region1]
    
    return expanded_Y


def expanded_inner_folds_combined_plus_indices_connectome(X_train, X_test, Y_train, Y_test):
    """
    Combines the training and testing data from inner cross-validation folds and generates train-test indices.
    Adapted for when connectomic data is part of features. 

    Parameters:
    X_train (np.ndarray): Training features from the outer fold.
    X_test (np.ndarray): Testing features from the outer fold.
    Y_train (np.ndarray): Training labels from the outer fold.
    Y_test (np.ndarray): Testing labels from the outer fold.

    Returns:
    X_combined (np.ndarray): Combined feature matrix from all inner folds.
    Y_combined (np.ndarray): Combined label vector from all inner folds.
    train_test_indices (list): List of tuples containing train and test indices for each inner fold.
    """
    current_index = 0
    
    train_idx = np.arange(current_index, current_index + len(X_train))
    test_idx = np.arange(current_index + len(X_train), current_index + len(X_train) + len(X_test))
             
    train_test_indices = [(train_idx, test_idx)]
    
    X_combined = []
    Y_combined = []
    
    X_combined.append(X_train)
    X_combined.append(X_test)
    Y_combined.append(Y_train)
    Y_combined.append(Y_test)
    
    # Combine the lists into arrays
    X_combined = np.vstack(X_combined)
    Y_combined = np.hstack(Y_combined)

    return X_combined, Y_combined, train_test_indices        


def create_data_loader(X, y, batch_size, device, shuffle=True, weight=False):
    X = torch.FloatTensor(X).to(device)
    y = torch.FloatTensor(y).to(device)
    dataset = TensorDataset(X, y)
    
    # Handle class imbalance for binary classification, but only during training
    if len(y.unique()) == 2 and weight:
        # Calculate class weights
        class_counts = torch.bincount(y.long())
        print('class counts', class_counts)
        total_samples = len(y)
        class_weights = total_samples / (2 * class_counts)
        print(f'Class weights - 0: {class_weights[0]:.4f}, 1: {class_weights[1]:.4f}')
        sample_weights = class_weights[y.long()]

        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    
    # For validation or non-binary cases, use regular DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def augment_batch(batch_idx, dataset, device, verbose=False):
    '''
    Helper function to swap out targets of a population batch with individualized targets from population data
    '''
    start_time = time.time()    
    
    # Convert expanded index to upper triangle index
    batch_idx = batch_idx - (batch_idx % 2) 
    # Convert to tuple index of true connectome
    true_pairs = np.array([dataset.expanded_idx_to_true_pair[idx.item()] for idx in batch_idx])
    # Convert tuple index to expanded index for population data (might be different indexing system)
    pop_edge_indices = np.array([dataset.upper_tri_map[tuple(pair)] for pair in true_pairs])
    # Get mask for all subjects for these edges
    valid_subjects_mask = dataset.masks[:, pop_edge_indices]
    # For each edge, randomly select a subject with valid data for that edge
    random_subjects = np.array([
        np.random.choice(np.where(valid_subjects_mask[:, i])[0])
        for i in range(len(pop_edge_indices))])
    # Use vectorized indexing to store edge values for selected subjects
    batch_y = torch.tensor(dataset.connectomes[random_subjects, pop_edge_indices], dtype=torch.float32).to(device)    
    
    pop_time = time.time() - start_time
    if verbose:
        print(f"Augmentation time: {pop_time:.2f} seconds")
    
    # Return augmented target batch of shape (batch_size,)
    return batch_y

class RegionPairDataset(Dataset):
    def __init__(self, X, Y, coords, valid2true_mapping):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.coords = torch.tensor(coords, dtype=torch.float32)
        self.valid_indices = np.array(list(valid2true_mapping.keys()), dtype=np.int32) # subset indices
        self.true_indices = np.array(list(valid2true_mapping.values()), dtype=np.int32) # full dataset indices

        self.X_expanded = torch.tensor(expand_X_symmetric(X), dtype=torch.float32)
        self.Y_expanded = torch.tensor(expand_Y_symmetric(Y), dtype=torch.float32)
        self.coords_expanded = torch.tensor(expand_X_symmetric(coords), dtype=torch.float32)
        self.distances_expanded = torch.sqrt(torch.sum((self.coords_expanded[:, :3] - self.coords_expanded[:, 3:])**2, dim=1))
        self.valid_indices_expanded = expand_X_symmetric(self.valid_indices.reshape(-1,1)).astype(np.int32)
        self.true_indices_expanded = expand_X_symmetric(self.true_indices.reshape(-1,1)).astype(np.int32)
        valid_pairs = tuple(map(tuple, self.valid_indices_expanded))
        true_pairs = tuple(map(tuple, self.true_indices_expanded))

        self.valid_pair_to_expanded_idx = dict(zip(valid_pairs, range(len(valid_pairs))))
        self.true_pair_to_expanded_idx = dict(zip(true_pairs, range(len(true_pairs))))
        self.expanded_idx_to_valid_pair = {v: k for k, v in self.valid_pair_to_expanded_idx.items()}
        self.expanded_idx_to_true_pair = {v: k for k, v in self.true_pair_to_expanded_idx.items()}
    
        # UKBB connectomes
        self.connectomes = np.load(f'{data_dir}/connectomes_upper.npy', allow_pickle=True)
        self.masks = np.load(f'{data_dir}/masks.npy', allow_pickle=True)
        self.subject_ids = np.load(f'{data_dir}/subject_ids.npy', allow_pickle=True)
        self.upper_tri_map = np.load(f'{data_dir}/upper_triangle_index_map.npy', allow_pickle=True)
        self.upper_tri_map = self.upper_tri_map.item()
        
    def __len__(self):
        return len(self.X_expanded)
        
    def __getitem__(self, idx):
        # return features, target, coords, and index in expanded dataset (can be used to map to true region pairs)
        return self.X_expanded[idx], self.Y_expanded[idx], self.coords_expanded[idx], idx
        
    def get_all_data(self, idx):
        return {
            'features': self.X_expanded[idx],
            'target': self.Y_expanded[idx], 
            'coords': self.coords_expanded[idx],
            'valid_idx': self.valid_indices_expanded[idx],
            'true_idx': self.true_indices_expanded[idx]
        }
        
    def get_by_valid_indices(self, idx1, idx2):
        # Get data for a specific valid index pair using the mapping
        pair_idx = self.valid_pair_to_expanded_idx[(idx1, idx2)]
        return self.__getitem__(pair_idx)
        
    def get_by_true_indices(self, idx1, idx2):
        # Get data for a specific true index pair using the mapping
        # This is really only for returning target data
        pair_idx = self.true_pair_to_expanded_idx[(idx1, idx2)]
        return self.__getitem__(pair_idx)