# Gene2Conn/data/data_load.py

from imports import *

def load_transcriptome(parcellation='schaefer_100', stability = '0.2', dataset='AHBA', run_PCA=False, omit_subcortical=False):
    '''
    stability can be 0.2 or -1
    '''
    if dataset == 'AHBA':
        genes = fetch_ahba()
    
        # labels
        region_labels = genes['label'].to_list()
        gene_labels = list(genes.columns)
        
        # gene expression data in schaefer version with most stringent similarity threshold >0.2
        schaefer114_genes = pd.read_csv(f"./data/enigma/allgenes_stable_r{stability}_schaefer_100.csv")
        schaefer114_genes.set_index('label', inplace=True)
        schaefer114_genes = np.array(schaefer114_genes)
    
        if omit_subcortical: 
            schaefer114_genes = schaefer114_genes[:100, :]
        
        if run_PCA: 
            pca = PCA(n_components=100)
            
            schaefer114_genes_pca = pca.fit_transform(schaefer114_genes)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            var_thresh = 0.95
            
            # Find the number of components that explain at least 95% of the variance
            num_components_variance = np.argmax(cumulative_variance >= var_thresh) + 1
            
            print(f"Number of components explaining {var_thresh*100}% of the variance: {num_components_variance}")
            schaefer114_genes_var = schaefer114_genes_pca[:, :num_components_variance]
            schaefer114_genes = schaefer114_genes_var
        
        return schaefer114_genes
        
    elif dataset == 'GTEx':
        relative_data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)    
        gtex_in_ahba = relative_data_path + '/data/region_map_pickles/RxG_data_gtex_mean_gtex_ahba_space.pkl'
        with open(gtex_in_ahba, 'rb') as f:
            gtex_in_ahba = pickle.load(f)
        return np.array(gtex_in_ahba)
    elif dataset == 'AHBA in GTEx':
        relative_data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)    
        ahba_in_gtex = relative_data_path + '/data/region_map_pickles/RxG_data_ahba_mean_gtex_ahba_space.pkl'
        with open(ahba_in_gtex, 'rb') as f:
            ahba_in_gtex = pickle.load(f)
        return np.array(ahba_in_gtex)
    elif dataset == 'UTSW':
        relative_data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)        
        ut_in_ahba = relative_data_path + '/data/region_map_pickles/RxG_data_utsmc_mean_ahba_utsmc_space.pkl'
        with open(ut_in_ahba, 'rb') as f:
            ut_in_ahba = pickle.load(f)
        return np.array(np.log1p(ut_in_ahba))
    elif dataset == 'AHBA in UTSW':
        relative_data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
        ahba_in_utsw = relative_data_path + '/data/region_map_pickles/RxG_data_ahba_mean_ahba_utsmc_space.pkl'
        with open(ahba_in_utsw, 'rb') as f:
            ahba_in_utsw = pickle.load(f)
        return np.array(ahba_in_utsw)

def load_connectome(parcellation='schaefer_100', dataset='AHBA', omit_subcortical=False, measure='FC'):
    # measure can be 'FC', 'SC'
    relative_data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)        

    if dataset == 'AHBA':
        if measure == 'FC':
            fc_combined_mat_schaef_100, fc_combined_labels_schaef_100 = load_fc_as_one(parcellation='schaefer_100')
            if omit_subcortical: 
                fc_combined_mat_schaef_100 = fc_combined_mat_schaef_100[:100, :100]
            return fc_combined_mat_schaef_100
        elif measure == 'SC': 
            sc_combined_mat_schaef_100, sc_combined_labels_schaef_100 = load_sc_as_one(parcellation='schaefer_100')
            if omit_subcortical: 
                sc_combined_mat_schaef_100 = sc_combined_mat_schaef_100[:100, :100] 
            return sc_combined_mat_schaef_100
            
    elif dataset == 'GTEx': 
        gtex_connectome_path = relative_data_path + '/data/region_map_pickles/HCP_Connectome_GTEX_Regions.pkl'
        with open(gtex_connectome_path, 'rb') as f:
            gtex_connectome = pickle.load(f)
        return np.array(gtex_connectome)
    elif dataset == 'UTSW':
        ut_connectome_path = relative_data_path + '/data/region_map_pickles/HCP_Connectome_UTSMC_Regions.pkl'
        with open(ut_connectome_path, 'rb') as f:
            ut_connectome = pickle.load(f)
        
        return np.array(ut_connectome)

def load_coords():
    relative_data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)        

    hcp_schaef = pd.read_csv(relative_data_path + '/data/atlas_info/schaef114.csv')
    
    # Extract the coordinates from the DataFrame
    coordinates = hcp_schaef[['mni_x', 'mni_y', 'mni_z']].values
    
    return coordinates


    
