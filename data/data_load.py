# Gene2Conn/data/data_load.py

from imports import *

def load_transcriptome(parcellation='S100', stability='0.2', dataset='AHBA', run_PCA=False, omit_subcortical=True):
    """
    Load transcriptome data from various datasets with optional PCA dimensionality reduction.
    
    Parameters:
    -----------
    parcellation : str, default='S100'. Options: 'S100', 'S456'
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
        pca = PCA()
        data_pca = pca.fit_transform(data)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= var_thresh) + 1
        return data_pca[:, :n_components]

    def _load_ahba_data():
        """Load and process AHBA dataset in respectiv parcellation"""
        ssl._create_default_https_context = ssl._create_unverified_context
        genes = fetch_ahba()
        
        if parcellation == 'S100':
            genes_data = pd.read_csv(f"./data/enigma/allgenes_stable_r{stability}_schaefer_100.csv")
            genes_data.set_index('label', inplace=True)
            genes_data = np.array(genes_data)
        elif parcellation == 'S456':
            AHBA_UKBB_path = os.path.normpath(os.getcwd() + os.sep + os.pardir) + '/GeneEx2Conn_data/Penn_UKBB_data/AHBA_population_MH/'
            AHBA_S456_transcriptome = pd.read_csv(os.path.join(AHBA_UKBB_path, 'AHBA_schaefer456_mean.csv'))
            AHBA_S456_transcriptome = AHBA_S456_transcriptome.drop('label', axis=1)
            if stability == '0.2':
                genes_list = pd.read_csv(f"./data/enigma/allgenes_stable_r{stability}_schaefer_400.csv").columns.tolist()
            else: # use stable 100 genes as maximal gene list
                genes_list = pd.read_csv(f"./data/enigma/allgenes_stable_r{stability}_schaefer_100.csv").columns.tolist()
            
            genes_list.remove('label')
            genes_data = np.array(AHBA_S456_transcriptome[genes_list])
            
        if omit_subcortical:
            # Retain only cortical regions by rounding down to nearest hundred
            n = (genes_data.shape[0] // 100) * 100
            genes_data = genes_data[:n, :]
        
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

def load_connectome(parcellation='S100', dataset='AHBA', omit_subcortical=True, measure='FC', spectral=None):
    """
    Load and process connectome data from various datasets with optional spectral decomposition.
    
    Parameters:
    -----------
    parcellation : str, default='S100'. Options: 'S100', 'S456'
        Brain parcellation scheme to use
    dataset : str, default='AHBA'
        Dataset to load. Options: 'AHBA', 'GTEx', 'UTSW'
    omit_subcortical : bool, default=True
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
    UKBB_path = relative_data_path + '/GeneEx2Conn_data/Penn_UKBB_data/'
    HCP_path = relative_data_path + '/GeneEx2Conn_data/HCP1200_DTI/'

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
        # pull from HCP Enigma data
        if parcellation == 'S100':
            if measure == 'FC':
                matrix, _ = load_fc_as_one(parcellation='schaefer_100')
            elif measure == 'SC':
                matrix, _ = load_sc_as_one(parcellation='schaefer_100')
        # pull from UKBB+HCP1200 data
        elif parcellation == 'S456':
            if measure == 'FC':
                matrix = np.array(pd.read_csv('./data/UKBB/UKBB_S456_functional_conn.csv'))
            elif measure == 'SC':
                matrix = np.log1p(loadmat(HCP_path + '/4S456/4S456_DTI_count.mat')['connectivity'])
        
        if omit_subcortical:
            # Retain only cortical regions by rounding down to nearest hundred
            n = (matrix.shape[0] // 100) * 100
            matrix = matrix[:n, :n]
            
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


def load_coords(parcellation='S100'):
    """
    Return x, y, z coordinates of parcellation.
    """
    relative_data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)        

    if parcellation == 'S100':
        hcp_schaef = pd.read_csv(relative_data_path + '/GeneEx2Conn_data/atlas_info/schaef114.csv')
        coordinates = hcp_schaef[['mni_x', 'mni_y', 'mni_z']].values
    elif parcellation == 'S456':
        UKBB_S456_atlas_info_path = relative_data_path + '/GeneEx2Conn_data/atlas_info/atlas-4S456Parcels_dseg_reformatted.csv'
        UKBB_S456_atlas_info = pd.read_csv(UKBB_S456_atlas_info_path)
        # Store MNI coordinates from atlas info as list of [x,y,z] coordinates
        mni_coords = [[x, y, z] for x, y, z in zip(UKBB_S456_atlas_info['mni_x'], 
                                              UKBB_S456_atlas_info['mni_y'],
                                              UKBB_S456_atlas_info['mni_z'])]
        coordinates = mni_coords
        
    return coordinates


    
