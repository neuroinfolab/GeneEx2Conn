# Gene2Conn/data/data_load.py

from imports import *

def load_transcriptome(parcellation='schaefer_100', dataset='AHBA', run_PCA=False, omit_subcortical=False):
    if dataset == 'AHBA':
        genes = fetch_ahba()
    
        # labels
        region_labels = genes['label'].to_list()
        gene_labels = list(genes.columns)
        
        # gene expression data in schaefer version with most stringent similarity threshold >0.2
        schaefer114_genes = pd.read_csv('./data/enigma/allgenes_stable_r0.2_schaefer_100.csv')
        schaefer114_genes.set_index('label', inplace=True)
        schaefer114_genes = np.array(schaefer114_genes)
    
        if omit_subcortical: 
            schaefer114_genes = schaefer114_genes[:100, :]
        
        if run_PCA: 
            # Initialize PCA and set the number of components to 100
            pca = PCA(n_components=100)
            
            # Fit the PCA model to the data and transform it
            schaefer114_genes_pca = pca.fit_transform(schaefer114_genes)
            
            # Calculate cumulative variance
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            
            # Find the number of components that explain at least 80% of the variance
            num_components_95_variance = np.argmax(cumulative_variance >= 0.95) + 1  # +1 because np.argmax returns the index
            print(f"Number of components explaining 95% of the variance: {num_components_95_variance}")
            
            # Get the principal components explaining 95% of the variance
            schaefer114_genes_95_var = schaefer114_genes_pca[:, :num_components_95_variance]
    
            return schaefer114_genes_95_var
        else:
            return schaefer114_genes
            
    elif dataset == 'GTEx':
        relative_data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)    
        print(relative_data_path)
        gtex_in_ahba = relative_data_path + '/data/region_map_pickles/RxG_data_gtex_mean_gtex_ahba_space.pkl'
        with open(gtex_in_ahba, 'rb') as f:
            gtex_in_ahba = pickle.load(f)
        return np.array(gtex_in_ahba)
        
    elif dataset == 'UTSW':
        relative_data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)        
        ut_in_ahba = relative_data_path + '/data/region_map_pickles/RxG_data_utsmc_mean_ahba_utsmc_space.pkl'
        with open(ut_in_ahba, 'rb') as f:
            ut_in_ahba = pickle.load(f)
        return np.array(np.log1p(ut_in_ahba))
            

def load_connectome(parcellation='schaefer_100', dataset='AHBA', omit_subcortical=False, structural=False):
    if dataset == 'AHBA':
        if not structural:
            fc_combined_mat_schaef_100, fc_combined_labels_schaef_100 = load_fc_as_one(parcellation='schaefer_100')
            if omit_subcortical: 
                fc_combined_mat_schaef_100 = fc_combined_mat_schaef_100[:100, :100]
            return fc_combined_mat_schaef_100
        else: 
            sc_combined_mat_schaef_100, sc_combined_labels_schaef_100 = load_sc_as_one(parcellation='schaefer_100')
            if omit_subcortical: 
                sc_combined_mat_schaef_100 = sc_combined_mat_schaef_100[:100, :100] 
            return sc_combined_mat_schaef_100
    elif dataset == 'GTEx': 
        relative_data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)        
        gtex_connectome_path = relative_data_path + '/data/region_map_pickles/HCP_Connectome_GTEX_Regions.pkl'
        with open(gtex_connectome_path, 'rb') as f:
            gtex_connectome = pickle.load(f)
        return np.array(gtex_connectome)
    elif dataset == 'UTSW':
        relative_data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)        
        ut_connectome_path = relative_data_path + '/data/region_map_pickles/HCP_Connectome_UTSMC_Regions.pkl'
        with open(ut_connectome_path, 'rb') as f:
            ut_connectome = pickle.load(f)
        
        return np.array(ut_connectome)


