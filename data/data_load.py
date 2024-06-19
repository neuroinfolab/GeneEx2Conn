# Gene2Conn/data/data_load.py

from imports import *

def load_transcriptome(parcellation='schaefer_100'):
    genes = fetch_ahba()

    # labels 
    region_labels = genes['label'].to_list()
    gene_labels = list(genes.columns)
    
    # gene expression data in schaefer version with most stringent similarity threshold >0.2
    schaefer114_genes = pd.read_csv('./data/enigma/allgenes_stable_r0.2_schaefer_100.csv')
    schaefer114_genes.set_index('label', inplace=True)
    
    return np.array(schaefer114_genes)

def load_connectome(parcellation='schaefer_100'):
    fc_combined_mat_schaef_100, fc_combined_labels_schaef_100 = load_fc_as_one(parcellation='schaefer_100')
    
    return fc_combined_mat_schaef_100
    



