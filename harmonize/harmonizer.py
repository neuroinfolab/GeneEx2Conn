# GeneEx2Conn/harmonize/harmonizer.py

from imports import *
from scipy.stats import entropy
from anndata import AnnData

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
        self.metadata_columns, self.gene_columns = self.get_gene_columns()
        
        self.minmax_df = self.statistical_process_gene_data(preprocessing_steps = {
                                                            'AHBA': ['min_max'], 
                                                            'Chinese Brain Bank': ['min_max'],
                                                            'UTSW Medical Center': ['min_max'],
                                                            'GTEx': ['min_max'],
                                                            'Yale HBT': ['min_max'],
                                                            'HBT Brainspan': ['min_max']
                                                            })
        self.processed_df = None # combined_df.copy()  # Create a processed version of the dataset
        
        
    def get_gene_columns(self):
        """
        Function to dynamically determine the gene columns and metadata columns.
        Assumes that the 'sequencing_type' column is the last metadata column.
        Returns both metadata and gene columns.
        """
        metadata_cols = self.original_df.columns[:self.original_df.columns.get_loc('sequencing_type') + 1]
        gene_cols = self.original_df.columns[self.original_df.columns.get_loc('sequencing_type') + 1:]
        
        return metadata_cols, gene_cols

    ### PROCESSING
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
        
    ### APPLY COMBAT
    def apply_combat(self, df, covariates=["age decade", "sex", "macro region"]):
        """
        Apply ComBat batch effect correction on the provided dataframe.
        
        Parameters:
        - df: DataFrame, the dataframe on which to apply ComBat.
        - covariates: list, list of covariates to include in the batch correction (categorical variables).
        
        Returns:
        - combat_df: DataFrame, the batch-corrected dataframe.
        """
        # Make a copy of the input dataframe to avoid modifying the original
        combat_df = df.copy()

        # Extract the gene expression data (rows: genes, columns: samples)
        gene_data = combat_df[self.gene_columns].T  # Transpose for Combat (samples as columns)

        # Extract the batch information (which dataset each sample belongs to)
        batch_info = combat_df['dataset'].values

        # Ensure that the covariates are categorical and properly formatted
        for covar in covariates:
            if combat_df[covar].dtype != 'category':
                combat_df[covar] = combat_df[covar].astype('category')

        # Prepare the covariate matrix (one-hot encoded)
        covariate_data = combat_df[covariates]

        # Rename covariate columns to make them compatible with patsy (replace spaces, hyphens)
        covariate_data = covariate_data.rename(columns=lambda x: x.replace(' ', '_').replace('-', '_'))

        # Clean up the categorical values (replace spaces and hyphens in the covariate values)
        for covar in covariate_data.columns:
            covariate_data[covar] = covariate_data[covar].apply(lambda x: str(x).replace(' ', '_').replace('-', '_'))

        # Run pycombat to adjust for batch effects
        adjusted_data = pycombat_norm(
            counts=gene_data, 
            batch=batch_info, 
            covar_mod=covariate_data, 
            par_prior=True,  # Parametric adjustment
            mean_only=False  # Adjust both means and individual batch effects
        )

        # Reconstruct the dataframe by updating the gene columns with the Combat-adjusted data
        combat_df[self.gene_columns] = adjusted_data.T  # Transpose back to the original shape
        
        return combat_df

    ### SCIB
    def df_to_anndata(self, df):
        """
        Converts a DataFrame to AnnData format, required for scib metrics.
        
        Parameters:
        - df: pd.DataFrame, the DataFrame to convert
        
        Returns:
        - AnnData object with metadata and gene expression data
        """
        gene_data = df[self.gene_columns].to_numpy()
        adata = AnnData(X=gene_data)
        
        # Add metadata as annotations
        for col in self.metadata_columns:
            adata.obs[col] = df[col].values
        
        return adata

    def run_scib_metrics(self, df_before, df_after, batch_key="dataset", label_key="macro region"):
        """
        Runs scib 'metrics_fast' on the pre-processed and post-processed data.

        Parameters:
        - df_before: pd.DataFrame, DataFrame before batch correction
        - df_after: pd.DataFrame, DataFrame after batch correction
        - batch_key: str, the column name to be used as batch identifier (default: "dataset")
        - label_key: str, the column name to be used as biological label (default: "macro region")

        Returns:
        - pd.DataFrame with scib metrics for both biological conservation and batch correction
        """
        # Convert the dataframes to AnnData format
        adata_pre = self.df_to_anndata(df_before)
        adata_post = self.df_to_anndata(df_after)

        # Run scib fast metrics
        metrics_result = scib.metrics.metrics_fast(
            adata=adata_pre,
            adata_int=adata_post,
            batch_key=batch_key,  # Treat dataset as batch
            label_key=label_key  # Treat macro region as biological label
        )

        return metrics_result

    ### Example run for evaluating harmonization using scib metrics ###
    def evaluate_harmonization_with_scib(self, df_before=None, df_after=None, batch_key="dataset", label_key="macro region"):
        """
        Evaluate the harmonization quality of the dataset before and after processing using scib metrics.
        The user can specify which DataFrames to compare (e.g., original vs processed, minmax vs processed).

        Parameters:
        - df_before: pd.DataFrame, DataFrame before harmonization (default: self.original_df)
        - df_after: pd.DataFrame, DataFrame after harmonization (default: self.processed_df)
        - batch_key: str, the column name to be used as batch identifier (default: "dataset")
        - label_key: str, the column name to be used as biological label (default: "macro region")

        Returns:
        - pd.DataFrame with scib metrics for both biological conservation and batch correction
        """
        # Default to original and processed DataFrames if not provided
        if df_before is None:
            df_before = self.original_df

        if df_after is None:
            df_after = self.processed_df

        # Run the metrics
        metrics_result = self.run_scib_metrics(df_before, df_after, batch_key=batch_key, label_key=label_key)

        # Print or return the metrics for evaluation
        print(f"Metrics between {df_before} and {df_after}:")
        print(metrics_result)

        return metrics_result
    
    
    ### VISUALIZATION
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
            "Dataset": dataframe['dataset'], 
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
            "Dataset": dataframe['dataset'], 
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

    ### CUSTOM PLOTS
    def plot_gene_expression_heatmap(self, view='both'):
        """
        Create a gene expression heatmap with options to compare original and processed data.
        The view parameter controls whether to plot 'original', 'processed', or 'both'.
        """
        if view not in ['original', 'processed', 'both']:
            raise ValueError("View must be 'original', 'processed', or 'both'.")

        if view == 'original' or view == 'both':
            self._plot_single_heatmap(self.minmax_df, title="Original Data Heatmap")

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
        self._plot_macro_region_expression(self.minmax_df, macro_region, "Original", skip_num)
        self._plot_macro_region_expression(self.processed_df, macro_region, "Processed", skip_num)

    
    def _plot_macro_region_expression(self, df, macro_region, data_type="Original", skip_num=500):
        """
        Internal function to plot gene expression overview for a specific macro region (either original, minmax, or processed).
        """
        # Subset data to the specified macro region
        macro_region_df = df[df['macro region'] == macro_region]
    
        if macro_region_df.empty:
            print(f"No data found for macro region: {macro_region}")
            return
    
        # Initialize a figure
        plt.figure(figsize=(12, 8))
    
        # Loop over each unique dataset
        for dataset in macro_region_df['dataset'].unique():
            # Subset the DataFrame by the current dataset
            dataset_df = macro_region_df[macro_region_df['dataset'] == dataset]
    
            # Get the gene columns (assuming they start after the metadata columns)
            gene_columns = self.gene_columns
    
            # Calculate the median, mean, and standard deviation for each gene
            median_values = dataset_df[gene_columns].median()[0:len(gene_columns):skip_num]
            std_values = dataset_df[gene_columns].std()[0:len(gene_columns):skip_num]
            mean_values = dataset_df[gene_columns].mean()[0:len(gene_columns):skip_num]
    
            # Apply Savitzky-Golay filter to smooth the median values
            smoothed_median_values = savgol_filter(median_values, window_length=3, polyorder=2)
    
            # Plot the line for the smoothed median values (with label for the legend)
            plt.plot(median_values.index, smoothed_median_values, label=f"{dataset} Median", linewidth=2)
    
            # Plot the dots for the mean values (with color matching the line but no legend entry)
            plt.scatter(mean_values.index, mean_values, color=plt.gca().lines[-1].get_color(), marker='o', s=20, label='_nolegend_')
    
            # Add the shaded area representing standard deviation with lighter transparency
            plt.fill_between(median_values.index, 
                             median_values - std_values, 
                             median_values + std_values, 
                             color=plt.gca().lines[-1].get_color(), 
                             alpha=0.1)  # Lighter transparency
    
        # Customize the plot
        plt.title(f'{macro_region} Gene Expression ({data_type} Data)\nEvery {skip_num}th Gene by Dataset')
        plt.xlabel('Genes')
        plt.ylabel('Expression Level')
    
        # Reduce legend size and number of columns for clarity (only for median values)
        plt.legend(title='Dataset', loc='upper right', fontsize='x-small', ncol=1, title_fontsize='small')
    
        plt.grid(True)
        plt.xticks([])  # Remove x-axis labels
    
        # Show the plot
        plt.tight_layout()
        plt.show()


    def create_and_display_macro_region_stats(self, macro_region):
        """
        Create and display a table of mean, median, and standard deviation of gene expression values 
        for a given macro region, across all samples within each dataset.
        This will be done for both the minmax and processed versions of the dataset.
        The values will be color-coded with a uniform colormap.
        """
        def calculate_stats(df, region_name, data_type):
            """
            Helper function to calculate mean, median, and std for the given macro region.
            """
            # Subset data to the specified macro region
            macro_region_df = df[df['macro region'] == region_name]
    
            # Dictionary to store statistics
            stats_dict = {'Dataset': [], 'Mean': [], 'Median': [], 'Std Dev': [], 'Data Type': []}
    
            for dataset in macro_region_df['dataset'].unique():
                # Subset the DataFrame by the current dataset
                dataset_df = macro_region_df[macro_region_df['dataset'] == dataset]
    
                # Get the gene columns
                gene_columns = self.gene_columns
    
                # Calculate the mean, median, and std across all samples for the dataset
                mean_values = dataset_df[gene_columns].mean().mean()  # Mean of all means
                median_values = dataset_df[gene_columns].median().median()  # Median of all medians
                std_values = dataset_df[gene_columns].std().mean()  # Mean of std deviations
    
                # Append statistics to the dictionary
                stats_dict['Dataset'].append(dataset)
                stats_dict['Mean'].append(mean_values)
                stats_dict['Median'].append(median_values)
                stats_dict['Std Dev'].append(std_values)
                stats_dict['Data Type'].append(data_type)
    
            return pd.DataFrame(stats_dict)
    
        # Compute statistics for the minmax scaled version
        minmax_stats = calculate_stats(self.minmax_df, macro_region, 'Minmax')
    
        # If processed_df exists, compute stats for the processed version as well
        if self.processed_df is not None:
            processed_stats = calculate_stats(self.processed_df, macro_region, 'Processed')
            # Combine minmax and processed stats into two tables
            combined_stats = pd.concat([minmax_stats, processed_stats], ignore_index=True)
        else:
            combined_stats = minmax_stats
    
        # Display both tables side by side using pandas Styler with a uniform colormap
        def style_table(df):
            # Set a colormap for the entire table using a uniform scale
            styled_df = df.style.background_gradient(cmap="Greens", subset=['Mean', 'Median', 'Std Dev'], axis=None)
            return styled_df
    
        # Create styled tables for Minmax and Processed
        minmax_table = style_table(minmax_stats)
        processed_table = None
        if self.processed_df is not None:
            processed_table = style_table(processed_stats)
    
        # Display the two tables horizontally using side-by-side layout
        from IPython.display import display_html
        if processed_table is not None:
            display_html(minmax_table._repr_html_() + processed_table._repr_html_(), raw=True)
        else:
            display_html(minmax_table._repr_html_(), raw=True)
    
        # Return the combined stats in case we want to use it further
        return combined_stats


    
    
    def compute_distribution_similarity(self, df, macro_region="Frontal Lobe", metric="kl_divergence", bins=50):
        """
        Compute pairwise similarity (KL-Divergence or Correlation) between datasets for a given macro region.
        
        Parameters:
        - df: DataFrame (either original, minmax, or processed)
        - macro_region: str, macro region to subset data for
        - metric: str, either "kl_divergence" or "correlation"
        - bins: int, number of bins for the histograms (default is 30)
        
        Returns:
        - similarity_matrix: DataFrame containing the pairwise similarity scores
        """
        # Helper funcs
        def compute_kl_divergence(dist1, dist2):
            """
            Compute the KL-Divergence between two probability distributions (relative histograms).
            """
            # Use np.clip to avoid division by zero and log of zero
            dist1 = np.clip(dist1, 1e-10, None)
            dist2 = np.clip(dist2, 1e-10, None)
        
            return entropy(dist1, dist2)
    
        def compute_correlation(dist1, dist2):
            """
            Compute the Pearson correlation between two probability distributions (relative histograms).
            """
            return pearsonr(dist1, dist2)[0]

        
        # Subset data to the specified macro region
        macro_region_df = df[df['macro region'] == macro_region]
    
        # Store datasets
        datasets = macro_region_df['dataset'].unique()
        n_datasets = len(datasets)
    
        # Create an empty similarity matrix
        similarity_matrix = np.zeros((n_datasets, n_datasets))
    
        # Compute histograms (relative PDFs) for each dataset
        histograms = {}
        for dataset in datasets:
            dataset_df = macro_region_df[macro_region_df['dataset'] == dataset]
            gene_columns = dataset_df.columns[len(self.metadata_columns):]  # Gene expression columns
            gene_values = dataset_df[gene_columns].values.flatten()
    
            # Compute the relative histogram (PDF) for the gene expression values
            hist, _ = np.histogram(gene_values, bins=bins, density=True)
            histograms[dataset] = hist
    
        # Compute pairwise similarity
        for i, dataset1 in enumerate(datasets):
            for j, dataset2 in enumerate(datasets):
                if i == j:
                    similarity_matrix[i, j] = 1  # KL-Divergence or correlation with itself
                else:
                    dist1 = histograms[dataset1]
                    dist2 = histograms[dataset2]
    
                    if metric == "kl_divergence":
                        similarity_matrix[i, j] = compute_kl_divergence(dist1, dist2)
                    elif metric == "correlation":
                        similarity_matrix[i, j] = compute_correlation(dist1, dist2)
                    else:
                        raise ValueError("Unknown metric specified. Use 'kl_divergence' or 'correlation'.")
    
        # Convert the matrix to a DataFrame for easier plotting
        similarity_df = pd.DataFrame(similarity_matrix, index=datasets, columns=datasets)
        
        return similarity_df
    
    def plot_similarity_heatmap(self, similarity_df, metric="KL-Divergence"):
        """
        Plot the similarity matrix as a heatmap.
        
        Parameters:
        - similarity_df: DataFrame, the pairwise similarity matrix
        - metric: str, name of the metric to display in the title
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity_df, annot=True, cmap="coolwarm", cbar_kws={'label': metric}, fmt=".2f")
        plt.title(f"Dataset Similarity - {metric}")
        plt.xlabel('Dataset')
        plt.ylabel('Dataset')
        plt.tight_layout()
        plt.show()
        
    
    