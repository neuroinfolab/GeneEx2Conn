# Gene2Conn/data/data_load.py

from imports import *
from data.data_utils import _apply_pca

def load_transcriptome(parcellation='S100', gene_list='0.2', dataset='AHBA', run_PCA=False, omit_subcortical=True, hemisphere='left'):
    """
    Load transcriptome data from various datasets with optional PCA dimensionality reduction.
    
    Parameters:
    -----------
    parcellation : str, default='S100'. Options: 'S100', 'S400' (S400 is same as S456)
        Brain parcellation scheme to use
    gene_list : str, default='0.2'
       Gene lists to subset from AHBA data. Options: '0.2', '1', 'brain', 'neuron', 'oligodendrocyte', 'synaptome', 'layers', 'all_abagen', 'syngo'
    dataset : str, default='AHBA'
        Dataset to load. Options: 'AHBA', 'GTEx', 'AHBA in GTEx', 'UTSW', 'AHBA in UTSW'
    run_PCA : bool, default=False
        If True, applies PCA with 95% variance threshold
    omit_subcortical : bool, default=False
        If True, excludes subcortical regions
    hemisphere : str, default='both'
        Options: 'both', 'left', 'right'    # 'inter' not implemented

    Returns:
    --------
    np.ndarray
        Processed gene expression data
    """
    if dataset == 'AHBA':
        # Choose parcellation
        if parcellation == 'S100':
            genes_data = pd.read_csv(f"./data/enigma/allgenes_stable_r1_schaefer_{parcellation[1:]}.csv")
            region_labels = pd.read_csv('./data/enigma/schaef114_regions.txt', header=None).values.flatten().tolist()
            region_labels = [label.replace('L', 'LH_', 1) if label.startswith('L') else label.replace('R', 'RH_', 1) if label.startswith('R') else label for label in region_labels]
        elif parcellation == 'S400':
            AHBA_UKBB_path = os.path.normpath(os.getcwd() + os.sep + os.pardir) + '/GeneEx2Conn_data/Penn_UKBB_data/AHBA_population_MH/'
            genes_data = pd.read_csv(os.path.join(AHBA_UKBB_path, 'AHBA_schaefer456_mean.csv'))
            genes_data = genes_data.drop('label', axis=1)
            atlas_info = pd.read_csv('./data/UKBB/schaefer456_atlas_info.txt', sep='\t') #['label_7network'].tolist()
            region_labels = [row['label_7network'] if pd.notna(row['label_7network']) else row['label'] 
                            for _, row in atlas_info.iterrows()]
            
        
        # Choose gene list
        if gene_list == '0.2' or gene_list == '1':
            genes_list = pd.read_csv(f"./data/enigma/gene_lists/stable_r{gene_list}_schaefer_{parcellation[1:]}.txt", header=None)[0].tolist()
        elif gene_list in ['brain', 'neuron', 'oligodendrocyte', 'synaptome', 'layers']:
            genes_list = abagen.fetch_gene_group(gene_list)
        elif gene_list == 'all_abagen':  
            genes_list = set.union(
                        set(abagen.fetch_gene_group('brain')),
                        set(abagen.fetch_gene_group('neuron')), 
                        set(abagen.fetch_gene_group('oligodendrocyte')),
                        set(abagen.fetch_gene_group('synaptome')),
                        set(abagen.fetch_gene_group('layers')))
        elif gene_list == 'richiardi2015':
            genes_list = pd.read_csv('./data/enigma/gene_lists/richiardi2015.txt', header=None)[0].tolist()
        elif gene_list == 'syngo':
            genes_list = pd.read_csv('./data/enigma/gene_lists/syngo.txt', header=None)[0].tolist()

        # Subset data based on genes_list
        genes_data = np.array(genes_data[[gene for gene in genes_list if gene in genes_data.columns]])
    
        # Apply PCA 
        if run_PCA:
            genes_data = _apply_pca(genes_data)

        # Drop subcortical regions
        if omit_subcortical:
            n = (genes_data.shape[0] // 100) * 100 # Hacky way to retain only cortical regions by rounding down to nearest hundred
            genes_data = genes_data[:n, :]
            region_labels = region_labels[:n]

        lh_indices = [i for i, label in enumerate(region_labels) if 'LH' in label]
        rh_indices = [i for i, label in enumerate(region_labels) if 'RH' in label]
        lh_labels = [region_labels[i] for i in lh_indices] 
        rh_labels = [region_labels[i] for i in rh_indices]        
        
        # subset genes_data and region_labels based on hemisphere
        print('hemisphere', hemisphere)
        if hemisphere == 'left':
            genes_data = genes_data[lh_indices, :]
            region_labels = lh_labels
        elif hemisphere == 'right':
            genes_data = genes_data[rh_indices, :]
            region_labels = rh_labels

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

    if dataset in dataset_configs:
        config = dataset_configs[dataset]
        with open(relative_data_path + config['path'], 'rb') as f:
            data = pickle.load(f)
            return config['transform'](data)

    raise ValueError(f"Unknown dataset: {dataset}")

def load_connectome(parcellation='S100', omit_subcortical=True, dataset='AHBA', measure='FC', spectral=None, hemisphere='left'):
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

    def _load_ahba_connectome(parcellation, omit_subcortical, spectral):
        # pull from HCP Enigma data

        if parcellation == 'S100':
            if measure == 'FC':
                matrix, _ = load_fc_as_one(parcellation='schaefer_100')
            elif measure == 'SC':
                matrix, _ = load_sc_as_one(parcellation='schaefer_100')
            
            region_labels = pd.read_csv('./data/enigma/schaef114_regions.txt', header=None).values.flatten().tolist()
            region_labels = [label.replace('L', 'LH_', 1) if label.startswith('L') else label.replace('R', 'RH_', 1) if label.startswith('R') else label for label in region_labels]
       
        elif parcellation == 'S400':
            if measure == 'FC':
                matrix = np.array(pd.read_csv('./data/UKBB/UKBB_S456_functional_conn.csv'))
            elif measure == 'SC':
                matrix = np.log1p(loadmat('./data/HCP1200/4S456_DTI_count.mat')['connectivity'])
            
            atlas_info = pd.read_csv('./data/UKBB/schaefer456_atlas_info.txt', sep='\t') #['label_7network'].tolist()
            region_labels = [row['label_7network'] if pd.notna(row['label_7network']) else row['label'] 
                            for _, row in atlas_info.iterrows()]

     
        if omit_subcortical:
            # Retain only cortical regions by rounding down to nearest hundred
            n = (matrix.shape[0] // 100) * 100
            matrix = matrix[:n, :n]
            region_labels = region_labels[:n]

        lh_indices = [i for i, label in enumerate(region_labels) if 'LH' in label]
        rh_indices = [i for i, label in enumerate(region_labels) if 'RH' in label]
        lh_labels = [region_labels[i] for i in lh_indices] 
        rh_labels = [region_labels[i] for i in rh_indices]        

        if hemisphere == 'left':
            matrix = matrix[lh_indices, :][:, lh_indices]
            region_labels = lh_labels
        elif hemisphere == 'right':
            matrix = matrix[rh_indices, :][:, rh_indices]
            region_labels = rh_labels
            
        return _apply_spectral_decomposition(matrix, spectral)

    # Dataset loading logic
    replication_dataset_paths = {
        'GTEx': '/GeneEx2Conn_data/region_map_pickles/HCP_Connectome_GTEX_Regions.pkl',
        'UTSW': '/GeneEx2Conn_data/region_map_pickles/HCP_Connectome_UTSMC_Regions.pkl'
    }

    if dataset == 'AHBA':
        return _load_ahba_connectome(parcellation, omit_subcortical, spectral)
    
    # Load other datasets
    if dataset in replication_dataset_paths:
        with open(relative_data_path + replication_dataset_paths[dataset], 'rb') as f:
            return np.array(pickle.load(f))
            
    raise ValueError(f"Unknown dataset: {dataset}")

def load_coords(parcellation='S100', omit_subcortical=True, hemisphere='left'):
    """
    Return x, y, z coordinates of parcellation.
    
    Parameters:
    -----------
    parcellation : str, default='S100'
        Brain parcellation scheme to use
    omit_subcortical : bool, default=True
        If True, excludes subcortical regions
    hemisphere : str, default='both'
        Options: 'both', 'left', 'right'
    """
    relative_data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)        

    if parcellation == 'S100':
        hcp_schaef = pd.read_csv(relative_data_path + '/GeneEx2Conn_data/atlas_info/schaef114.csv')
        coordinates = hcp_schaef[['mni_x', 'mni_y', 'mni_z']].values
        
        region_labels = pd.read_csv('./data/enigma/schaef114_regions.txt', header=None).values.flatten().tolist()
        region_labels = [label.replace('L', 'LH_', 1) if label.startswith('L') else label.replace('R', 'RH_', 1) if label.startswith('R') else label for label in region_labels]
    elif parcellation == 'S400':
        UKBB_S456_atlas_info_path = relative_data_path + '/GeneEx2Conn_data/atlas_info/atlas-4S456Parcels_dseg_reformatted.csv'
        UKBB_S456_atlas_info = pd.read_csv(UKBB_S456_atlas_info_path)
        # Store MNI coordinates from atlas info as list of [x,y,z] coordinates
        mni_coords = [[x, y, z] for x, y, z in zip(UKBB_S456_atlas_info['mni_x'], 
                                              UKBB_S456_atlas_info['mni_y'],
                                              UKBB_S456_atlas_info['mni_z'])]
        coordinates = np.array(mni_coords)
        
        atlas_info = pd.read_csv('./data/UKBB/schaefer456_atlas_info.txt', sep='\t')
        region_labels = [row['label_7network'] if pd.notna(row['label_7network']) else row['label'] 
                        for _, row in atlas_info.iterrows()]
        
    if omit_subcortical:
        # Retain only cortical regions by rounding down to nearest hundred
        n = (coordinates.shape[0] // 100) * 100
        coordinates = coordinates[:n, :]
        region_labels = region_labels[:n]

    lh_indices = [i for i, label in enumerate(region_labels) if 'LH' in label]
    rh_indices = [i for i, label in enumerate(region_labels) if 'RH' in label]

    if hemisphere == 'left':
        coordinates = coordinates[lh_indices, :]
    elif hemisphere == 'right':
        coordinates = coordinates[rh_indices, :]
    
    return coordinates

    
