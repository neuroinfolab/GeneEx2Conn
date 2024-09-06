# GeneEx2Conn/harmonize/harmonizer.py

from imports import *

# useful global paths
par_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
data_dir = par_dir + '/data'
samples_dir = data_dir + '/samples_files'

class Harmonizer:
    def __init__(self, combined_df):
        """
        Initialize the Harmonizer object with both original and processed versions of the dataset.
        The original dataset is kept unchanged, and processing is done on the copy.
        """
        self.original_df = combined_df.copy()  # Store the original dataset
        self.processed_df = None # combined_df.copy()  # Create a processed version of the dataset
        
        # Determine gene and metadata columns
        self.metadata_columns, self.gene_columns = self.get_gene_columns()
    
    def get_gene_columns(self):
        """
        Function to dynamically determine the gene columns and metadata columns.
        Assumes that the 'sequencing_type' column is the last metadata column.
        Returns both metadata and gene columns.
        """
        metadata_cols = self.original_df.columns[:self.original_df.columns.get_loc('sequencing_type') + 1]
        gene_cols = self.original_df.columns[self.original_df.columns.get_loc('sequencing_type') + 1:]
        
        return metadata_cols, gene_cols
        
    def statistical_process_gene_data(self, preprocessing_steps):
        """
        Minimal preprocessing: log-transform and min-max scaling as specified per dataset.
        Updates the processed version of the dataset.
        """
        # Create a copy of the original_df to ensure original remains unchanged
        processed_df = self.original_df.copy()
        
        # Loop over each unique dataset and apply respective preprocessing
        for dataset in processed_df['dataset'].unique():
            scaler = MinMaxScaler()
            dataset_df = processed_df[processed_df['dataset'] == dataset].copy()  # Copy to avoid warnings
            
            steps = preprocessing_steps.get(dataset, [])
            
            if 'scaled_robust_sigmoid' in steps:
                # Apply scaled robust sigmoid logic for AHBA dataset
                if dataset == 'AHBA': # Load AHBA_srs_samples.csv
                    srs_df = pd.read_csv(samples_dir + '/AHBA_srs_samples.csv')
                    srs_gene_df = srs_df[self.gene_columns]
                    # drop null rows
                    threshold = int(srs_gene_df.shape[1] * 0.5)  
                    srs_gene_df = srs_gene_df.dropna(thresh=threshold)                
                    srs_gene_values = np.array(srs_gene_df)                    
                    processed_df.loc[dataset_df.index, self.gene_columns] = srs_gene_values
                else: 
                    # Apply scaled robust sigmoid (using RobustScaler followed by sigmoid)
                    robust_scaler = RobustScaler()
                    robust_scaled_values = robust_scaler.fit_transform(processed_df.loc[dataset_df.index, self.gene_columns])
                    sigmoid_scaled_values = 1 / (1 + np.exp(-robust_scaled_values))
                    processed_df.loc[dataset_df.index, self.gene_columns] = sigmoid_scaled_values
            if 'log_transform' in steps:
                # Apply log(1+x) transformation
                processed_df.loc[dataset_df.index, self.gene_columns] = np.log1p(processed_df.loc[dataset_df.index, self.gene_columns])

            if 'min_max' in steps:
                # Apply Min-Max scaling
                scaled_values = scaler.fit_transform(processed_df.loc[dataset_df.index, self.gene_columns])
                processed_df.loc[dataset_df.index, self.gene_columns] = scaled_values

            if 'z_score' in preprocessing_steps[dataset]:
                # Apply Z-scoring (Standardization)
                zscaler = StandardScaler()
                zscored_values = zscaler.fit_transform(processed_df.loc[dataset_df.index, self.gene_columns])
                processed_df.loc[dataset_df.index, self.gene_columns] = zscored_values
                
        self.processed_df = processed_df  # Store processed version
        return processed_df
    
    def run_umap(self, dataframe, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', n_jobs=-1):
        """
        Run UMAP on the specified dataframe and return the UMAP results.
        """
        gene_array = dataframe[self.gene_columns].to_numpy()
        umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, n_jobs=n_jobs)
        umap_results = umap_model.fit_transform(gene_array)
        return umap_results

    def plot_umap(self, umap_results, dataframe, title="UMAP Plot", color_by='dataset'):
        """
        Create a 2D or 3D UMAP plot using Plotly.
        """
        hover_data = {
            "Macro region": dataframe['macro region'],
            "Region": dataframe['tissue sample local name']
        }
        
        if umap_results.shape[1] == 2:
            fig = px.scatter(x=umap_results[:, 0], y=umap_results[:, 1], color=dataframe[color_by], labels={'color': color_by}, hover_data=hover_data, title=title)
            fig.update_traces(marker=dict(size=3))
        elif umap_results.shape[1] == 3:
            fig = px.scatter_3d(x=umap_results[:, 0], y=umap_results[:, 1], z=umap_results[:, 2], color=dataframe[color_by], labels={'color': color_by}, hover_data=hover_data, title=title)
            fig.update_traces(marker=dict(size=2))
        fig.show()

    def run_pca(self, dataframe, n_components=2):
        """
        Run PCA on the specified dataframe and return the PCA results.
        """
        gene_array = dataframe[self.gene_columns].to_numpy()
        pca_model = PCA(n_components=n_components)
        pca_results = pca_model.fit_transform(gene_array)
        return pca_results

    def plot_pca(self, pca_results, dataframe, title="PCA Plot", color_by='dataset'):
        """
        Create a 2D or 3D PCA plot using Plotly.
        """
        hover_data = {
            "Macro region": dataframe['macro region'],
            "Region": dataframe['tissue sample local name']
        }

        if pca_results.shape[1] == 2:
            fig = px.scatter(x=pca_results[:, 0], y=pca_results[:, 1], color=dataframe[color_by], labels={'color': color_by}, hover_data=hover_data, title=title)
            fig.update_traces(marker=dict(size=3))
        elif pca_results.shape[1] == 3:
            fig = px.scatter_3d(x=pca_results[:, 0], y=pca_results[:, 1], z=pca_results[:, 2], color=dataframe[color_by], labels={'color': color_by}, hover_data=hover_data, title=title)
            fig.update_traces(marker=dict(size=2))
        fig.show()

    def plot_gene_expression_heatmap(self, view='both'):
        """
        Create a gene expression heatmap with options to compare original and processed data.
        The view parameter controls whether to plot 'original', 'processed', or 'both'.
        """
        if view not in ['original', 'processed', 'both']:
            raise ValueError("View must be 'original', 'processed', or 'both'.")

        if view == 'original' or view == 'both':
            self._plot_single_heatmap(self.original_df, title="Original Data Heatmap")

        if view == 'processed' or view == 'both':
            self._plot_single_heatmap(self.processed_df, title="Processed Data Heatmap")

    def _plot_single_heatmap(self, dataframe, title):
        """
        Helper function to plot a single heatmap for gene expression.
        """
        dataset_labels = dataframe['dataset'].values
        unique_datasets = np.unique(dataset_labels)
        dataset_mapping = {label: idx for idx, label in enumerate(unique_datasets)}
        mapped_labels = np.array([dataset_mapping[label] for label in dataset_labels])
        
        gene_array = dataframe[self.gene_columns].to_numpy()
        cmap = plt.get_cmap('tab10', len(unique_datasets))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [5, 1]})
        heatmap = ax1.imshow(gene_array, aspect='auto', cmap='viridis')
        ax1.set_title(title)
        ax1.set_xlabel('Genes')
        ax1.set_ylabel('Samples')
        plt.colorbar(heatmap, ax=ax1)
        ax2.imshow(mapped_labels[:, np.newaxis], aspect='auto', cmap=cmap)
        ax2.set_title('Dataset Mapping')
        legend_handles = [Patch(color=cmap(i), label=unique_datasets[i]) for i in range(len(unique_datasets))]
        ax2.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), title="Datasets")
        plt.tight_layout()
        plt.show()


    def evaluate_harmonization_for_macro_region(self, macro_region, skip_num=500):
        """
        Evaluate harmonization quality for a specific macro region.
        Generates pre- and post-harmonization line plots showing median and std deviation for the gene expressions.
        """
        # Check if the processed data exists
        if self.processed_df is None:
            print("Please process the dataset before evaluating harmonization.")
            return
        
        # Plot for both original and processed data
        print(f"Evaluating harmonization for macro-region: {macro_region}")
        self._plot_macro_region_expression(self.original_df, macro_region, "Original", skip_num)
        self._plot_macro_region_expression(self.processed_df, macro_region, "Processed", skip_num)
    
    def _plot_macro_region_expression(self, df, macro_region, data_type="Original", skip_num=500):
        """
        Internal function to plot gene expression overview for a specific macro region (either original or processed).
        """
        # Subset data to the specified macro region
        macro_region_df = df[df['macro region'] == macro_region]
    
        if macro_region_df.empty:
            print(f"No data found for macro region: {macro_region}")
            return
    
        # Initialize a figure
        plt.figure(figsize=(10, 7))
    
        # Loop over each unique dataset
        for dataset in macro_region_df['dataset'].unique():
            # Subset the DataFrame by the current dataset
            dataset_df = macro_region_df[macro_region_df['dataset'] == dataset]
    
            # Get the gene columns (assuming they start after the metadata columns)
            gene_columns = self.gene_columns
    
            # Calculate the median and standard deviation for each gene
            median_values = dataset_df[gene_columns].median()[0:len(gene_columns):skip_num]
            std_values = dataset_df[gene_columns].std()[0:len(gene_columns):skip_num]
            mean_values = dataset_df[gene_columns].mean()[0:len(gene_columns):skip_num]
    
            # Apply Savitzky-Golay filter to smooth the median values
            smoothed_median_values = savgol_filter(median_values, window_length=3, polyorder=2)
    
            # Plot the line for the smoothed median values
            plt.plot(median_values.index, smoothed_median_values, label=f"{dataset} Median", linewidth=2)
    
            # Plot the dot for the mean values
            plt.scatter(mean_values.index, mean_values, label=f"{dataset} Mean", marker='o', s=20)
    
            # Add the shaded area representing standard deviation with lighter transparency
            plt.fill_between(median_values.index, 
                             median_values - std_values, 
                             median_values + std_values, 
                             alpha=0.1)  # Lighter transparency
        
        # Customize the plot
        plt.title(f'{macro_region} Gene Expression ({data_type} Data)\nEvery {skip_num}th Gene by Dataset')
        plt.xlabel('Genes')
        plt.ylabel('Expression Level')
    
        # Reduce legend size and number of columns for clarity
        plt.legend(title='Dataset', loc='upper right', fontsize='small', ncol=2, title_fontsize='small')
    
        plt.grid(True)
        plt.xticks([])  # Remove x-axis labels
    
        # Show the plot
        plt.tight_layout()
        plt.show()
