# Gene2Conn/data/data_utils.py

from imports import *

def create_data_loader(X, y, batch_size, device, shuffle=True):
    X = torch.FloatTensor(X).to(device)
    y = torch.FloatTensor(y).to(device)    
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
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


def expand_X_symmetric_kron(X, kron_input_dim):
    """
    Expands the X matrix symmetrically by computing the Kronecker product of 
    gene expressions from pairs of regions.

    Parameters:
    X (numpy.ndarray): Input matrix of gene expressions or other feature types. 

    Returns:
    numpy.ndarray: Expanded symmetric matrix using Kronecker product.
    """
    num_regions, num_genes = X.shape
    region_combinations = list(combinations(range(num_regions), 2))
    num_combinations = len(region_combinations)
    
    if int(X.shape[1]) == kron_input_dim or kron_input_dim == None:
        expanded_X = np.zeros((num_combinations * 2, num_genes**2))
    else:
        expanded_X = np.zeros((num_combinations * 2, kron_input_dim**2 + (2*(X.shape[1]-kron_input_dim))))

    for i, (region1, region2) in enumerate(region_combinations):
        if int(X.shape[1]) == kron_input_dim or kron_input_dim == None: # kronecker specified for single feature type
            # Kronecker product of the region1 and region2 gene expressions            
            expanded_X[i * 2] = np.kron(X[region1], X[region2])
            expanded_X[i * 2 + 1] = np.kron(X[region2]+1, X[region1]+1)
        else: 
            # Kronecker product of the region1 and region2 gene expressions concatenated with other features
            expanded_X[i * 2] = np.hstack((
                                    np.kron(X[region1][:kron_input_dim], X[region2][:kron_input_dim]), # kronecker product vector
                                    np.concatenate((X[region1][kron_input_dim:], X[region2][kron_input_dim:])) # additional feature concatenated vector
                                    ))
            
            expanded_X[i * 2 + 1] = np.hstack((
                                        np.kron(X[region2][:kron_input_dim], X[region1][:kron_input_dim]), 
                                        np.concatenate((X[region2][kron_input_dim:], X[region1][kron_input_dim:]))
                                    ))

    return expanded_X

def expand_X_symmetric_struct_summ(X, include_input=False):
    """
    Expands the X matrix symmetrically by extracting pairwise connectivity values and correlations of the structural connectivity profiles.
    If include_input is True, also includes the original structural connectivity rows.
    
    Parameters:
    X (numpy.ndarray): Input square connectivity matrix.
    include_input (bool): Whether to include original structural connectivity rows.
    
    Returns:
    numpy.ndarray: Expanded matrix with connectivity values, correlations, and optionally original rows.
    """
    num_regions = X.shape[0]
    region_combinations = list(combinations(range(num_regions), 2))
    num_combinations = len(region_combinations)
    
    if include_input:
        expanded_X = np.zeros((num_combinations * 2, 2 + 2*num_regions))
    else:
        expanded_X = np.zeros((num_combinations * 2, 2))
        print('STRUCT SUMM MODALITY')
    
    for i, (region1, region2) in enumerate(region_combinations):
        # Get connectivity values
        expanded_X[i * 2, 0] = X[region1, region2]
        expanded_X[i * 2 + 1, 0] = X[region2, region1]
        
        # Compute correlations
        corr = np.corrcoef(X[region1, :], X[region2, :])[0,1]
        expanded_X[i * 2, 1] = corr
        expanded_X[i * 2 + 1, 1] = corr

        # Include original structural connectivity rows if requested
        if include_input:
            expanded_X[i * 2, 2:2+num_regions] = X[region1]
            expanded_X[i * 2, 2+num_regions:] = X[region2]
            expanded_X[i * 2 + 1, 2:2+num_regions] = X[region2]
            expanded_X[i * 2 + 1, 2+num_regions:] = X[region1]

    return expanded_X

def expand_X_symmetric_spatial_null(X):
    """
    Expands X matrix to extract euclidean distances and structural connectivity values for each pair of regions.
    
    Parameters:
    X (numpy.ndarray): Array where first 3 columns are coordinates and remaining columns contain structural connectivity matrix
        
    Returns:
    numpy.ndarray: Matrix containing euclidean distances and structural connectivity values for each region pair
    """
    coords = X[:, :3]  # First 3 columns are coordinates
    Y_sc = X[:, 3:]   # Remaining columns are structural connectivity
    
    num_regions = coords.shape[0]
    region_combinations = list(combinations(range(num_regions), 2))
    num_combinations = len(region_combinations)
    
    expanded_X = np.zeros((num_combinations * 2, 2))
    
    for i, (region1, region2) in enumerate(region_combinations):
        # Calculate euclidean distance between regions
        dist = np.linalg.norm(coords[region1] - coords[region2])
        
        # Get structural connectivity values
        sc_value = Y_sc[region1, region2]
        
        # Store values for both directions
        expanded_X[i * 2, 0] = dist
        expanded_X[i * 2, 1] = sc_value
        
        expanded_X[i * 2 + 1, 0] = dist  
        expanded_X[i * 2 + 1, 1] = sc_value

    return expanded_X


def expand_X_symmetric_shared(X_train1, X_train2, Y_train2):
    """
    Expands rectangular X matrix symmetrically (including shared test regions).

    Parameters:
    X_train1 (numpy.ndarray): Square training matrix.
    X_train2 (numpy.ndarray): Rectangular training matrix connecting to shared test regions.
    Y_train2 (numpy.ndarray): Rectangular connectivity matrix.

    Returns:
    tuple: Expanded symmetric X and Y matrices.
    """
    train_size, num_regions = X_train2.shape
    test_size = X_train2.shape[1]
    num_combinations = train_size * test_size * 2
    
    expanded_X = np.zeros((num_combinations, 2 * num_regions))
    expanded_Y = np.zeros(num_combinations)

    index = 0
    for i in range(train_size):
        for j in range(test_size):
            expanded_X[index] = np.hstack((X_train2[i], X_train1[j]))
            expanded_Y[index] = Y_train2[i, j]
            index += 1

            expanded_X[index] = np.hstack((X_train1[j], X_train2[i]))
            expanded_Y[index] = Y_train2[i, j]
            index += 1
    
    return expanded_X, expanded_Y


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
    Expands the X matrix symmetrically by combining featuresfrom pairs of regions.

    Parameters:
    X (numpy.ndarray): Input matrix of gene expressions.

    Returns:
    numpy.ndarray: Expanded symmetric matrix.
    """
    num_regions, num_genes = X.shape
    region_combinations = list(combinations(range(num_regions), 2))
    num_combinations = len(region_combinations)

    expanded_X = np.zeros((num_combinations * 2, 2 * num_genes))
    
    for i, (region1, region2) in enumerate(region_combinations):
        expanded_X[i * 2] = np.concatenate((X[region1], X[region2]))
        expanded_X[i * 2 + 1] = np.concatenate((X[region2], X[region1]))

    return expanded_X


def expand_Y_symmetric(Y):
    """
    Expands the Y matrix symmetrically by extracting pairwise connectivity values.

    Parameters:
    Y (numpy.ndarray): Input matrix of connectome values.

    Returns:
    numpy.ndarray: Expanded symmetric vector.
    """
    num_regions = Y.shape[0]
    region_combinations = list(combinations(range(num_regions), 2))
    num_combinations = len(region_combinations)
    
    expanded_Y = np.zeros(num_combinations * 2)
    
    for i, (region1, region2) in enumerate(region_combinations):
        expanded_Y[i * 2] = Y[region1, region2]
        expanded_Y[i * 2 + 1] = Y[region2, region1]
    
    return expanded_Y


def process_cv_splits(X, Y, cv_obj, all_train=False, test_shared=False, spatial_null=False, struct_summ=False, kron=False, kron_input_dim=None):
    """
    Function to process cross-validation splits, expand training and test data as needed.

    Parameters:
    - X (np.ndarray): Feature matrix.
    - Y (np.ndarray): Connectivity matrix.
    - cv_obj: Cross-validation object with a split method.
    - all_train (bool): Whether to include all training data in expansion.
    - incl_conn (bool): Whether to include connectivity profiles in the expansion.
    - test_shared (bool): Whether to include shared test data in the expansion.

    Returns:
    - list of tuples: Each tuple contains (X_train, X_test, Y_train, Y_test) for a fold.
    """
    results = []

    for fold_idx, (train_index, test_index) in enumerate(cv_obj.split(X, Y)):        
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index][:, train_index], Y[test_index][:, test_index]
        Y_test_conn = Y_test

        #print(f"INPUT: Fold {fold_idx} shapes - X_train: {X_train.shape}, X_test: {X_test.shape}, Y_train: {Y_train.shape}, Y_test: {Y_test.shape}")   
        
        if all_train:
            X_train2, Y_train2 = X[test_index], Y[train_index][:, test_index]
            X_train2, Y_train2 = expand_shared_matrices(X_train, X_train2, Y_train2)
            
            X_train = expand_X_symmetric(X_train)
            Y_train = expand_Y_symmetric(Y_train)

            X_test = expand_X_symmetric(X_test)
            Y_test = expand_Y_symmetric(Y_test)

            if not test_shared: 
                X_train = np.concatenate((X_train, X_train2))
                Y_train = np.concatenate((Y_train, Y_train2))
            else: 
                X_test = np.concatenate((X_test, X_train2))
                Y_test = np.concatenate((Y_test, Y_train2))
        else: 
            if kron: # can implement this elsewhere if necessary
                X_train = expand_X_symmetric_kron(X_train, kron_input_dim)
                Y_train = expand_Y_symmetric(Y_train)

                X_test = expand_X_symmetric_kron(X_test, kron_input_dim)
                Y_test = expand_Y_symmetric(Y_test)
            elif struct_summ: # struct summ case where we use the strength and correlation from structural connectivity
                X_train = expand_X_symmetric_struct_summ(X_train)
                Y_train = expand_Y_symmetric(Y_train)

                X_test = expand_X_symmetric_struct_summ(X_test)
                Y_test = expand_Y_symmetric(Y_test) 
            elif spatial_null:
                X_train = expand_X_symmetric_spatial_null(X_train)
                Y_train = expand_Y_symmetric(Y_train)
                X_test = expand_X_symmetric_spatial_null(X_test)
                Y_test = expand_Y_symmetric(Y_test)
            else:
                X_train = expand_X_symmetric(X_train)
                Y_train = expand_Y_symmetric(Y_train)

                X_test = expand_X_symmetric(X_test)
                Y_test = expand_Y_symmetric(Y_test)

        #print(f"PROCESSED: Fold {fold_idx} shapes - X_train: {X_train.shape}, X_test: {X_test.shape}, Y_train: {Y_train.shape}, Y_test: {Y_test.shape}")
        results.append((X_train, X_test, Y_train, Y_test))

    return results


def expanded_inner_folds_combined_plus_indices(inner_fold_splits):
    """
    Combines the training and testing data from inner cross-validation folds and generates train-test indices.

    Parameters:
    inner_fold_splits (list): List of tuples containing (X_train, X_test, Y_train, Y_test) for each inner fold.

    Returns:
    X_combined (np.ndarray): Combined feature matrix from all inner folds.
    Y_combined (np.ndarray): Combined label vector from all inner folds.
    train_test_indices (list): List of tuples containing train and test indices for each inner fold.
    """
    X_combined, Y_combined, train_test_indices = [], [], []

    current_index = 0
    for X_train, X_test, Y_train, Y_test in inner_fold_splits:
        # Append the training and testing data to the combined lists
        X_combined.append(X_train)
        X_combined.append(X_test)
        Y_combined.append(Y_train)
        Y_combined.append(Y_test)

        # Create train and test indices for the current fold, this is sequential due to how it's now stored in inner_fold_splits 
        train_idx = np.arange(current_index, current_index + len(X_train))
        test_idx = np.arange(current_index + len(X_train), current_index + len(X_train) + len(X_test))

        # Append the train-test indices as a tuple
        train_test_indices.append((train_idx, test_idx))

        # Update the current index
        current_index += len(X_train) + len(X_test)
    
    # Combine the lists into arrays
    X_combined = np.vstack(X_combined)
    Y_combined = np.hstack(Y_combined)

    return X_combined, Y_combined, train_test_indices

'''
def expand_X_Y_symmetric_conn_only(X, Y):
    """
    Expands X matrix symmetrically and vectorizes Y matrix for connectivity only model.

    Parameters:
    X (numpy.ndarray): Input matrix of gene expressions.
    Y (numpy.ndarray): Input connectivity matrix.

    Returns:
    tuple: Expanded symmetric X and Y matrices.
    """
    num_regions, num_genes = X.shape
    region_combinations = list(combinations(range(num_regions), 2))
    num_combinations = len(region_combinations)

    expanded_X = np.zeros((num_combinations * 2, 2 * num_genes))
    expanded_Y = np.zeros((num_combinations * 2))

    for i, (region1, region2) in enumerate(region_combinations):
        expanded_X[i * 2] = np.concatenate((X[region1], X[region2]))
        expanded_X[i * 2 + 1] = np.concatenate((X[region2], X[region1]))
        expanded_Y[i * 2] = Y[region1, region2]
        expanded_Y[i * 2 + 1] = Y[region2, region1]

    return expanded_X, expanded_Y
'''

'''
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
'''