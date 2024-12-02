# Gene2Conn/data/data_load.py

from imports import *

def load_transcriptome(parcellation='schaefer_100', stability='0.2', dataset='AHBA', run_PCA=False, omit_subcortical=False):
    """
    Load transcriptome data from various datasets with optional PCA dimensionality reduction.
    
    Parameters:
    -----------
    parcellation : str, default='schaefer_100'
        Brain parcellation scheme to use
    stability : str, default='0.2'
        Stability threshold for AHBA data. Options: '0.2' or '-1'
    dataset : str, default='AHBA'
        Dataset to load. Options: 'AHBA', 'GTEx', 'AHBA in GTEx', 'UTSW', 'AHBA in UTSW'
    run_PCA : bool, default=False
        If True, applies PCA with 95% variance threshold
    omit_subcortical : bool, default=False
        If True, excludes subcortical regions
    
    Returns:
    --------
    np.ndarray
        Processed gene expression data
    """
    def _apply_pca(data, var_thresh=0.95):
        """Apply PCA with variance threshold."""
        pca = PCA(n_components=100)
        data_pca = pca.fit_transform(data)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= var_thresh) + 1
        return data_pca[:, :n_components]

    def _load_ahba_data():
        """Load and process AHBA dataset."""
        ssl._create_default_https_context = ssl._create_unverified_context
        genes = fetch_ahba()
        
        # Load gene expression data
        genes_data = pd.read_csv(f"./data/enigma/allgenes_stable_r{stability}_schaefer_100.csv")
        genes_data.set_index('label', inplace=True)
        genes_data = np.array(genes_data)
        
        if omit_subcortical:
            genes_data = genes_data[:100, :]
        
        if run_PCA:
            genes_data = _apply_pca(genes_data)
            
        return genes_data

    # Dataset paths configuration
    relative_data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)

    dataset_configs = {
        'GTEx': {
            'path': '/GeneEx2Conn_data/region_map_pickles/RxG_data_gtex_mean_gtex_ahba_space.pkl',
            'transform': lambda x: np.array(x)
        },
        'AHBA in GTEx': {
            'path': '/GeneEx2Conn_data/region_map_pickles/RxG_data_ahba_mean_gtex_ahba_space.pkl',
            'transform': lambda x: np.array(x)
        },
        'UTSW': {
            'path': '/GeneEx2Conn_data/region_map_pickles/RxG_data_utsmc_mean_ahba_utsmc_space.pkl',
            'transform': lambda x: np.array(np.log1p(x))
        },
        'AHBA in UTSW': {
            'path': '/GeneEx2Conn_data/region_map_pickles/RxG_data_ahba_mean_ahba_utsmc_space.pkl',
            'transform': lambda x: np.array(x)
        }
    }

    # Load and process data
    if dataset == 'AHBA':
        return _load_ahba_data()
    
    if dataset in dataset_configs:
        config = dataset_configs[dataset]
        with open(relative_data_path + config['path'], 'rb') as f:
            data = pickle.load(f)
            return config['transform'](data)
            
    raise ValueError(f"Unknown dataset: {dataset}")

def load_connectome(parcellation='schaefer_100', dataset='AHBA', omit_subcortical=False, measure='FC', spectral=None):
    """
    Load and process connectome data from various datasets with optional spectral decomposition.
    
    Parameters:
    -----------
    parcellation : str, default='schaefer_100'
        Brain parcellation scheme to use
    dataset : str, default='AHBA'
        Dataset to load. Options: 'AHBA', 'GTEx', 'UTSW'
    omit_subcortical : bool, default=False
        If True, excludes subcortical regions
    measure : str, default='FC'
        Type of connectivity measure. Options: 'FC' (functional), 'SC' (structural)
    spectral : str or None, default=None
        Type of spectral decomposition. Options: 'L' (Laplacian), 'A' (Adjacency), None
    
    Returns:
    --------
    np.ndarray
        Processed connectome data
    """
    relative_data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)

    def _apply_spectral_decomposition(matrix, method):
        if method == 'L':
            L = laplacian(matrix, normed=True)
            _, eigenvectors = eig(L)
            k = int(L.shape[1]) - 1
            return eigenvectors[:, 1:k+1]  # Skip first eigenvector (zero eigenvalue)
        
        elif method == 'A':
            _, eigenvectors = eig(matrix)
            k = int(matrix.shape[1])
            return eigenvectors[:, :k]
        
        return matrix

    def _load_ahba_connectome():
        if measure == 'FC':
            matrix, _ = load_fc_as_one(parcellation='schaefer_100')
        else:  # measure == 'SC'
            matrix, _ = load_sc_as_one(parcellation='schaefer_100')
        
        if omit_subcortical:
            matrix = matrix[:100, :100]
            
        return _apply_spectral_decomposition(matrix, spectral)

    # Dataset loading logic
    replication_dataset_paths = {
        'GTEx': '/GeneEx2Conn_data/region_map_pickles/HCP_Connectome_GTEX_Regions.pkl',
        'UTSW': '/GeneEx2Conn_data/region_map_pickles/HCP_Connectome_UTSMC_Regions.pkl'
    }

    if dataset == 'AHBA':
        return _load_ahba_connectome()
    
    # Load other datasets
    if dataset in replication_dataset_paths:
        with open(relative_data_path + replication_dataset_paths[dataset], 'rb') as f:
            return np.array(pickle.load(f))
            
    raise ValueError(f"Unknown dataset: {dataset}")


def load_coords():
    """
    Return x, y, z coordinates of Schaefer 114 parcellation.
    """
    relative_data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)        

    hcp_schaef = pd.read_csv(relative_data_path + '/GeneEx2Conn_data/atlas_info/schaef114.csv')
    
    # Extract the coordinates from the DataFrame
    coordinates = hcp_schaef[['mni_x', 'mni_y', 'mni_z']].values
    
    return coordinates


    
