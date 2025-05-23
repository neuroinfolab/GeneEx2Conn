from env.imports import *

relative_data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir) + '/GeneEx2Conn_data'
absolute_data_path = relative_data_path

def _apply_pca(data, var_thresh=0.95):
        """Helper to apply PCA with variance threshold."""
        # Find rows without NaNs
        valid_rows = ~np.isnan(data).any(axis=1)
        
        # Fit PCA on valid rows only
        pca = PCA()
        data_valid = data[valid_rows]
        data_pca_valid = pca.fit_transform(data_valid)
        
        # Get number of components based on variance threshold
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= var_thresh) + 1
        print(f"Number of components for 95% variance PCA: {n_components}")
        
        data_pca = np.full((data.shape[0], n_components), np.nan)
        
        # Fill in transformed valid rows
        data_pca[valid_rows] = data_pca_valid[:, :n_components]
        
        return data_pca

def get_iPA_masks(parcellation):
    """
    Get hemisphere and subcortical masks for a given parcellation.
    
    Args:
        parcellation (str): Name of parcellation (e.g. 'iPA_391')
        
    Returns:
        tuple: (hemi_mask_list, subcort_mask_list, n_cortical)
            - hemi_mask_list: List of 0s and 1s indicating right (1) vs left (0) hemisphere
            - subcort_mask_list: List of 0s and 1s indicating subcortical (1) vs cortical (0) regions
            - n_cortical: Number of cortical regions (optional)
    """
    BHA2_path = absolute_data_path + '/BHA2/'
    metadata = pd.read_csv(os.path.join(BHA2_path, parcellation, f'{parcellation}.csv'), index_col=0)

    # Create hemisphere mask based on right-sided regions
    right_cols = [col for col in metadata.columns if '_R' in col]
    hemi_mask = (metadata[right_cols] > 0).any(axis=1).astype(int)
    hemi_mask_list = hemi_mask.tolist()

    # Create subcortical mask based on subcortical regions 
    subcort_cols = [col for col in metadata.columns if 'Subcortical' in col]
    subcort_mask = (metadata[subcort_cols] > 0).any(axis=1).astype(int)
    subcort_mask_list = subcort_mask.tolist()

    # Count number of cortical regions (where subcortical mask is 0)
    n_cortical = sum(x == 0 for x in subcort_mask_list)

    return hemi_mask_list, subcort_mask_list, n_cortical

def load_transcriptome(parcellation='S100', gene_list='0.2', dataset='AHBA', run_PCA=False, omit_subcortical=False, hemisphere='both', impute_strategy='mirror_interpolate', sort_genes='refgenome', return_valid_genes=False, null_model='none', random_seed=42):
    """
    Load transcriptome data with optional PCA reduction.
    
    Args:
        parcellation (str): Brain parcellation ('S100', 'S400'). Default: 'S100'
        gene_list (str): Gene subset ('0.2', '1', 'brain', 'neuron', 'oligodendrocyte', 
            'synaptome', 'layers', 'all_abagen', 'syngo'). Default: '0.2'
        dataset (str): Source dataset ('AHBA', 'GTEx', 'AHBA in GTEx', 'UTSW', 
            'AHBA in UTSW'). Default: 'AHBA'
        run_PCA (bool): Apply PCA with 95% variance threshold. Default: False
        omit_subcortical (bool): Exclude subcortical regions. Default: False
        hemisphere (str): Brain hemisphere ('both', 'left', 'right'). Default: 'both'
        impute_strategy (str): How to impute missing values ('mirror', 'interpolate', 'mirror_interpolate'). Default: None (otherwise: 'mirror_interpolate' recommended)
        sort_genes (str): Sort genes based on reference genome order 'refgenome', or by mean expression across brain 'expression', or alphabetically (None). Default: 'expression'
        null_model (str): Shuffle gene expression data as in spin test null model. Default: none, spin, random
    Returns
        np.ndarray: Processed gene expression data
    """
    if dataset == 'AHBA':
        # Choose parcellation
        if parcellation == 'S100':
            region_labels = [label.replace('L', 'LH_', 1) if label.startswith('L') else label.replace('R', 'RH_', 1) if label.startswith('R') else label for label in pd.read_csv('./data/enigma/schaef114_regions.txt', header=None).values.flatten().tolist()]
            genes_data = pd.read_csv(f"./data/enigma/allgenes_stable_r1_schaefer_{parcellation[1:]}.csv") # from https://github.com/saratheriver/enigma-extra/tree/master/ahba
        elif parcellation == 'S400':
            AHBA_UKBB_path = absolute_data_path + '/UKBB_data/AHBA_population_MH/'
            if impute_strategy == 'mirror':
                genes_data = pd.read_csv(os.path.join(AHBA_UKBB_path, 'AHBA_schaefer456_mean_mirror.csv'))
            elif impute_strategy == 'interpolate':
                genes_data = pd.read_csv(os.path.join(AHBA_UKBB_path, 'AHBA_schaefer456_mean_interpolate.csv'))
            elif impute_strategy == 'mirror_interpolate':
                genes_data = pd.read_csv(os.path.join(AHBA_UKBB_path, 'AHBA_schaefer456_mean_mirror_interpolate.csv'))
            else:
                genes_data = pd.read_csv(os.path.join(AHBA_UKBB_path, 'AHBA_schaefer456_mean.csv'))
            genes_data = genes_data.drop('label', axis=1)
            region_labels = [row['label_7network'] if pd.notna(row['label_7network']) else row['label'] for _, row in pd.read_csv('./data/UKBB/atlas-4S456Parcels_dseg_reformatted.csv').iterrows()]
        elif 'iPA' in parcellation:
            # options are iPA_183, iPA_391, iPA_568. iPA_729
            BHA2_path = absolute_data_path + '/BHA2/'
            genes_data = pd.read_csv(os.path.join(BHA2_path, parcellation, 'transcriptomics.csv'), index_col=0).T
            genes_list = genes_data.columns.tolist()
        
        # Choose gene list
        if 'iPA' in parcellation:
            genes_list_abagen = pd.read_csv(f"./data/enigma/gene_lists/stable_r{gene_list}_schaefer_400.txt", header=None)[0].tolist()
            genes_list = list(set(genes_list).intersection(set(genes_list_abagen))) # drop last 3 genes to make token divisible
        elif gene_list == '0.2' or gene_list == '1':
            genes_list = pd.read_csv(f"./data/enigma/gene_lists/stable_r{gene_list}_schaefer_{parcellation[1:]}.txt", header=None)[0].tolist()
        elif gene_list in ['brain', 'neuron', 'oligodendrocyte', 'synaptome', 'layers']:
            genes_list = abagen.fetch_gene_group(gene_list)
        elif gene_list == 'richiardi2015':
            genes_list = pd.read_csv('./data/enigma/gene_lists/richiardi2015.txt', header=None)[0].tolist()
        elif gene_list == 'syngo':
            genes_list = pd.read_csv('./data/enigma/gene_lists/syngo.txt', header=None)[0].tolist()
        elif gene_list == 'all_abagen':  
            genes_list = set.union(
                        set(abagen.fetch_gene_group('brain')),
                        set(abagen.fetch_gene_group('neuron')), 
                        set(abagen.fetch_gene_group('oligodendrocyte')),
                        set(abagen.fetch_gene_group('synaptome')),
                        set(abagen.fetch_gene_group('layers')))

        # Temporarily subset all sorting methods to ref genome for experiments
        human_refgenome = pd.read_csv(absolute_data_path + '/human_refgenome/human_refgenome_ordered.csv')
        ordered_genes = human_refgenome['gene_id'].tolist()
        gene_order_dict = {gene: idx for idx, gene in enumerate(ordered_genes)}
        valid_genes = [gene for gene in genes_list if gene in gene_order_dict and gene in genes_data.columns]
        
        if sort_genes == 'refgenome': # this may drop some genes if the gene list symbol does not directly match to the gene_id of the reference genome (ususally drops <5% of genes)
            human_refgenome = pd.read_csv(absolute_data_path + '/human_refgenome/human_refgenome_ordered.csv')
            ordered_genes = human_refgenome['gene_id'].tolist()
            gene_order_dict = {gene: idx for idx, gene in enumerate(ordered_genes)}
            # Filter and sort genes based on reference genome order
            valid_genes = [gene for gene in genes_list if gene in gene_order_dict and gene in genes_data.columns]
            valid_genes.sort(key=lambda x: gene_order_dict[x])
            genes_data = np.array(genes_data[valid_genes])
        elif sort_genes == 'expression':
            genes_data = np.array(genes_data[valid_genes])
            mean_expr = np.nanmean(genes_data, axis=0)
            sort_idx = np.argsort(mean_expr)
            genes_data = genes_data[:, sort_idx]
            valid_genes = [valid_genes[i] for i in sort_idx]
        elif sort_genes == 'random':
            random_genes = np.random.permutation(valid_genes)
            genes_data = np.array(genes_data[random_genes])
            valid_genes = random_genes
        else:
            genes_data = np.array(genes_data[valid_genes])

        # Apply PCA if specified
        if run_PCA:
            genes_data = _apply_pca(genes_data)

        # base return for iPA parcellation
        if 'iPA' in parcellation:
            if null_model == 'spin':
                hemi_mask, subcortical_mask, n_cortical = get_iPA_masks(parcellation)
                # load spins
                spins_df_10k = pd.read_csv(f'./data/enigma/10000_{parcellation}_null_spins.csv')
                spins_df_10k = spins_df_10k.sort_values('mean_error_rank', ascending=True)
                seed_to_index = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 42: 9}
                # Get spin index based on random seed
                spin_idx = seed_to_index.get(random_seed, 0)# Get spin indices
                cortical_spins_list = spins_df_10k['cortical_spins'].tolist()
                cortical_spins_list = [eval(x) for x in cortical_spins_list]
                cortical_spin_indices = np.array(cortical_spins_list)
                subcortical_spins_list = spins_df_10k['subcortical_spins'].tolist()
                subcortical_spins_list = [eval(x) for x in subcortical_spins_list]
                subcortical_spin_indices = np.array(subcortical_spins_list)
        
                cortical_spin_idx = cortical_spin_indices[spin_idx]
                subcortical_spin_idx = subcortical_spin_indices[spin_idx]

                genes_data = np.vstack([genes_data[cortical_spin_idx], genes_data[subcortical_spin_idx+n_cortical]])

            return genes_data

        # Drop subcortical regions if specified
        if omit_subcortical:
            n = (genes_data.shape[0] // 100) * 100 # Hacky way to retain only cortical regions by rounding down to nearest hundred
            genes_data = genes_data[:n, :]
            region_labels = region_labels[:n]
        
        # Subset genes_data and region_labels based on hemisphere
        lh_indices, rh_indices = [i for i, label in enumerate(region_labels) if 'LH' in label], [i for i, label in enumerate(region_labels) if 'RH' in label]
        lh_labels, rh_labels = [region_labels[i] for i in lh_indices], [region_labels[i] for i in rh_indices]
        if hemisphere == 'left':
            genes_data = genes_data[lh_indices, :]
            region_labels = lh_labels
        elif hemisphere == 'right':
            genes_data = genes_data[rh_indices, :]
            region_labels = rh_labels

        if return_valid_genes:
            print("valid genes", valid_genes)
            return genes_data, valid_genes
        
        if null_model == 'spin' and parcellation == 'S400':
            print('Spinning gene expression')
            spins_df_10k = pd.read_csv('./data/enigma/10000_null_spins.csv')
            spins_df_10k = spins_df_10k.sort_values('mean_error_rank', ascending=True)
            seed_to_index = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 42: 9}
             # Get spin index based on random seed
            spin_idx = seed_to_index.get(random_seed, 0)
            print(f"Spin index for seed {random_seed}: {spin_idx}")
            print(f"Mean error rank for spin index {spin_idx}: {spins_df_10k['mean_error_rank'].iloc[spin_idx]}")

            # Get spin indices
            cortical_spins_list = spins_df_10k['cortical_spins'].tolist()
            cortical_spins_list = [eval(x) for x in cortical_spins_list]
            cortical_spin_indices = np.array(cortical_spins_list)
            subcortical_spins_list = spins_df_10k['subcortical_spins'].tolist()
            subcortical_spins_list = [eval(x) for x in subcortical_spins_list]
            subcortical_spin_indices = np.array(subcortical_spins_list)
    
            cortical_spin_idx = cortical_spin_indices[spin_idx]
            subcortical_spin_idx = subcortical_spin_indices[spin_idx]
            
            if omit_subcortical:
                genes_data = genes_data[cortical_spin_idx]
            else:
                genes_data = np.vstack([genes_data[cortical_spin_idx], genes_data[subcortical_spin_idx+400]])
            
        elif null_model == 'random':
            print('permuting gene expression')
            rng = np.random.default_rng(random_seed)
            genes_data = rng.permutation(genes_data)
        
        return genes_data

def load_connectome(parcellation='S100', omit_subcortical=True, dataset='AHBA', measure='FC', spectral=None, hemisphere='both', include_labels=False, diag=0, binarize=False):
    """
    Load and process connectome data with optional spectral decomposition.
    
    Args:
        parcellation (str): Brain parcellation ('S100', 'S456'). Default: 'S100'
        dataset (str): Dataset to load. Default: 'AHBA'
        omit_subcortical (bool): Exclude subcortical regions. Default: True
        measure (str): Connectivity type ('FC', 'SC'). Default: 'FC'
        spectral (str): Decomposition type ('L', 'A', None). Default: None
    
    Returns:
        np.ndarray: Processed connectome data
    """
    if dataset == 'AHBA':
        # Load relevant data and corresponding region labels from parcellation
        if parcellation == 'S100':
            region_labels = [label.replace('L', 'LH_', 1) if label.startswith('L') else label.replace('R', 'RH_', 1) if label.startswith('R') else label for label in pd.read_csv('./data/enigma/schaef114_regions.txt', header=None).values.flatten().tolist()]
            if measure == 'FC':
                matrix, _ = load_fc_as_one(parcellation='schaefer_100')
                matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min()) # a few values are slightly negative and some above 1 so scale into [0,1] range
            elif measure == 'SC':
                matrix, _ = load_sc_as_one(parcellation='schaefer_100')
                matrix[matrix < 0] = 0 # a handful of values are slightly negative so set to 0
                matrix = matrix / matrix.max()
        elif parcellation == 'S400' or parcellation == 'S456':
            region_labels = [row['label_7network'] if pd.notna(row['label_7network']) else row['label'] for _, row in pd.read_csv('./data/UKBB/atlas-4S456Parcels_dseg_reformatted.csv').iterrows()]
            if measure == 'FC':
                matrix = np.array(pd.read_csv('./data/UKBB/UKBB_S456_FC_mu.csv'))
            elif measure == 'SC':
                matrix = np.log1p(loadmat('./data/HCP1200/4S456_DTI_count.mat')['connectivity'])
                matrix = matrix / matrix.max()
        elif 'iPA' in parcellation:
            BHA2_path = absolute_data_path + '/BHA2/'
            matrix = np.array(pd.read_csv(os.path.join(BHA2_path, parcellation, 'fc/fc_mean.csv'), header=None))
            return matrix
        
        # Add diagonal as 1 if specified (diagonal is ignored in edge-wise reconstruction)
        if diag == 1:
            matrix = matrix + np.eye(matrix.shape[0])
        elif diag == 0:
            np.fill_diagonal(matrix, 0)

        if binarize:
            threshold = 0.01 if measure == 'SC' else 0.3 # 0.3 is an empirical hyperparameter for FC 
            matrix = (matrix >= threshold).astype(int)
            print(f'Number of 1s: {np.sum(matrix)}, Number of 0s: {np.sum(matrix==0)}, Class balance (1s): {np.mean(matrix):.3f}')

        # Remove subcortical if specified
        if omit_subcortical:
            n = (matrix.shape[0] // 100) * 100
            matrix = matrix[:n, :n]
            region_labels = region_labels[:n]

        # Subset based on hemisphere
        lh_indices, rh_indices = [i for i, label in enumerate(region_labels) if 'LH' in label], [i for i, label in enumerate(region_labels) if 'RH' in label]
        lh_labels, rh_labels = [region_labels[i] for i in lh_indices], [region_labels[i] for i in rh_indices]
        if hemisphere == 'left':
            matrix = matrix[lh_indices, :][:, lh_indices]
            region_labels = lh_labels
        elif hemisphere == 'right':
            matrix = matrix[rh_indices, :][:, rh_indices]
            region_labels = rh_labels
        
        # Apply appropriate spectral decomposition
        if spectral == 'L':
            L = laplacian(matrix, normed=True)
            _, eigenvectors = eig(L)
            k = int(L.shape[1]) - 1
            return eigenvectors[:, 1:k+1]  # Skip first eigenvector (zero eigenvalue) 
        elif spectral == 'A':
            _, eigenvectors = eig(matrix)
            k = int(matrix.shape[1])
            return eigenvectors[:, :k]
        
        if include_labels:
            return matrix, region_labels, lh_indices, rh_indices
        
        return matrix

def load_coords(parcellation='S100', omit_subcortical=True, hemisphere='both'):
    """
    Get MNI coordinates for brain regions in specified parcellation.

    Args:
        parcellation (str): Parcellation scheme ('S100' or 'S400'). Default: 'S100'
        omit_subcortical (bool): Exclude subcortical regions. Default: True
        hemisphere (str): 'both', 'left', or 'right'. Default: 'both'
    """
    if parcellation == 'S100':
        region_labels = [label.replace('L', 'LH_', 1) if label.startswith('L') else label.replace('R', 'RH_', 1) if label.startswith('R') else label for label in pd.read_csv('./data/enigma/schaef114_regions.txt', header=None).values.flatten().tolist()]
        hcp_schaef = pd.read_csv(absolute_data_path + '/atlas_info/schaef114.csv')
        coordinates = hcp_schaef[['mni_x', 'mni_y', 'mni_z']].values
    elif parcellation == 'S400':
        region_labels = [row['label_7network'] if pd.notna(row['label_7network']) else row['label'] for _, row in pd.read_csv('./data/UKBB/atlas-4S456Parcels_dseg_reformatted.csv').iterrows()]
        UKBB_S456_atlas_info = pd.read_csv('./data/UKBB/atlas-4S456Parcels_dseg_reformatted.csv')
        mni_coords = [[x, y, z] for x, y, z in zip(UKBB_S456_atlas_info['mni_x'], 
                                              UKBB_S456_atlas_info['mni_y'],
                                              UKBB_S456_atlas_info['mni_z'])]
        coordinates = np.array(mni_coords)
    elif 'iPA' in parcellation:
        BHA2_path = absolute_data_path + '/BHA2/'
        atlas_info = pd.read_csv(os.path.join(BHA2_path, parcellation, f'{parcellation}.csv'))
        coordinates = atlas_info[['X_MNI', 'Y_MNI', 'Z_MNI']].values
        return np.array(coordinates)
    
    if omit_subcortical:
        n = (coordinates.shape[0] // 100) * 100 # retain only cortical regions by rounding down to nearest hundred
        coordinates = coordinates[:n, :]
        region_labels = region_labels[:n]

    lh_indices, rh_indices = [i for i, label in enumerate(region_labels) if 'LH' in label], [i for i, label in enumerate(region_labels) if 'RH' in label]
    if hemisphere == 'left':
        coordinates = coordinates[lh_indices, :]
    elif hemisphere == 'right':
        coordinates = coordinates[rh_indices, :]
    
    return coordinates

def load_network_labels(parcellation='S100', omit_subcortical=False, dataset='HCP', hemisphere='both'):
    """
    Load network labels for a given parcellation and dataset.

    Args:
        parcellation (str): Parcellation scheme ('S100' or 'S400'). Default: 'S100'
        omit_subcortical (bool): Exclude subcortical regions. Default: False
        dataset (str): Dataset to load ('HCP', 'UKBB'). Default: 'HCP'
        hemisphere (str): Brain hemisphere ('both', 'left', 'right'). Default: 'both'
    """
    if parcellation == 'S100': 
        schaef156_atlas_info = pd.read_csv('./data/UKBB/atlas-4S156Parcels_dseg_reformatted.csv')       
        schaef156_atlas_info.loc[schaef156_atlas_info['atlas_name'] == 'Cerebellum', 'network_label'] = 'Cerebellum'
        schaef156_atlas_info.loc[(schaef156_atlas_info['network_label'].isna()) & 
                                (schaef156_atlas_info['atlas_name'] != 'Cerebellum'), 'network_label'] = 'Subcortical'

        if omit_subcortical:
            schaef156_atlas_info = schaef156_atlas_info.iloc[:100]
        elif dataset == 'HCP':
            schaef156_atlas_info = schaef156_atlas_info.iloc[:114]
        elif dataset == 'UKBB':
            schaef156_atlas_info = schaef156_atlas_info.iloc[:156]
        
        labels = schaef156_atlas_info['label'].tolist()
        network_labels = schaef156_atlas_info['network_label'].values

    elif parcellation == 'S400':
        schaef456_atlas_info = pd.read_csv('./data/UKBB/atlas-4S456Parcels_dseg_reformatted.csv')
        schaef456_atlas_info.loc[schaef456_atlas_info['atlas_name'] == 'Cerebellum', 'network_label'] = 'Cerebellum'
        schaef456_atlas_info.loc[(schaef456_atlas_info['network_label'].isna()) & 
                                (schaef456_atlas_info['atlas_name'] != 'Cerebellum'), 'network_label'] = 'Subcortical'
        
        if omit_subcortical:
            schaef456_atlas_info = schaef456_atlas_info.iloc[:400]
        else:
            schaef456_atlas_info = schaef456_atlas_info.iloc[:456]
        
        labels = schaef456_atlas_info['label'].tolist()
        network_labels = schaef456_atlas_info['network_label'].values
    elif 'iPA' in parcellation:
        BHA2_path = absolute_data_path + '/BHA2/'
        atlas_info = pd.read_csv(os.path.join(BHA2_path, parcellation, f'{parcellation}.csv'))
        labels = atlas_info['ROI_number'].tolist()
        network_labels = atlas_info['VOL'].values
        return labels, network_labels

    # Add hemisphere prefix to labels
    labels = [label.replace('L', 'LH_', 1) if label.startswith('L') else label.replace('R', 'RH_', 1) if label.startswith('R') else label for label in labels]

    # Subset based on hemisphere
    lh_indices = [i for i, label in enumerate(labels) if 'LH' in label]
    rh_indices = [i for i, label in enumerate(labels) if 'RH' in label]
    
    if hemisphere == 'left':
        labels = [labels[i] for i in lh_indices]
        network_labels = network_labels[lh_indices]
    elif hemisphere == 'right':
        labels = [labels[i] for i in rh_indices]
        network_labels = network_labels[rh_indices]

    return labels, network_labels