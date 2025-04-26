import pandas as pd
import numpy as np
import argparse
import scanpy as sc
import logging

"""
How to run:
python -m gene_expression_transform \
    --input corrected_gene_expression.csv \
    --output corrected_gene_expression_transformed.csv \
    --drop_sparse \
    --log_transform \
    --mean_impute \
    --select_hvg
"""

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    """Loads gene expression data from a CSV file."""
    logging.info(f"Loading data from {filepath}...")
    try:
        # Assuming the first column is the gene index
        df = pd.read_csv(filepath, index_col=0)
        logging.info(f"Data loaded successfully: {df.shape[0]} genes, {df.shape[1]} neurons.")
        # Ensure data is numeric, coercing errors
        df = df.apply(pd.to_numeric, errors='coerce')
        # Check if coercion introduced NaNs where there weren't zeros before
        if df.isnull().values.any():
             logging.warning("Non-numeric values found and converted to NaN. Consider cleaning the input data.")
             # Optionally, fill NaNs introduced by coercion, e.g., with 0 or mean
             # df.fillna(0, inplace=True)
        return df
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def drop_sparse_genes(df, threshold=0.5):
    """Drops genes that are zero in more than a threshold proportion of neurons."""
    logging.info(f"Applying sparsity filter (threshold: >{threshold*100}% zeros)...")
    n_neurons = df.shape[1]
    zero_counts = (df == 0).sum(axis=1)
    genes_to_drop = zero_counts[zero_counts > n_neurons * threshold].index
    if len(genes_to_drop) > 0:
        df_filtered = df.drop(index=genes_to_drop)
        logging.info(f"Dropped {len(genes_to_drop)} sparse genes. New shape: {df_filtered.shape}")
        return df_filtered
    else:
        logging.info("No genes met the sparsity criteria for dropping.")
        return df

def log_transform(df):
    """Applies log1p transformation (log(x+1)) to the dataframe."""
    logging.info("Applying log1p transformation...")
    # Check for negative values before log transform
    if (df < 0).any().any():
        logging.warning("Negative values found in data before log transform. Clamping to 0.")
        df = df.clip(lower=0)
    df_log = np.log1p(df)
    logging.info("Log transformation applied.")
    return df_log

def mean_impute_zeros(df, min_presence_threshold=0.5):
    """Imputes zeros with the row (gene) mean if the gene is present in at least min_presence_threshold proportion of neurons."""
    logging.info(f"Applying mean imputation for zeros (min presence: {min_presence_threshold*100}%)...")
    df_imputed = df.copy()
    n_neurons = df.shape[1]
    imputed_count = 0
    for gene in df_imputed.index:
        gene_row = df_imputed.loc[gene]
        non_zero_count = (gene_row != 0).sum()
        # Check if gene is present enough to warrant imputation
        if non_zero_count >= n_neurons * min_presence_threshold and non_zero_count < n_neurons:
            # Calculate mean excluding zeros
            mean_val = gene_row[gene_row != 0].mean()
            # Impute only where the original value was exactly 0
            zero_mask = (df.loc[gene] == 0) # Use original df to identify true zeros
            df_imputed.loc[gene, zero_mask] = mean_val
            imputed_count += zero_mask.sum()
    if imputed_count > 0:
         logging.info(f"Imputed {imputed_count} zero values with row means.")
    else:
         logging.info("No zero values met the criteria for mean imputation.")
    return df_imputed

def select_highly_variable_genes(df, n_top_genes=2000):
    """Selects highly variable genes using scanpy."""
    logging.info(f"Selecting top {n_top_genes} highly variable genes using scanpy...")
    # Scanpy expects AnnData object, genes as columns (variables)
    adata = sc.AnnData(df.T) # Transpose df: neurons=observations, genes=variables
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=min(n_top_genes, adata.shape[1]), flavor='seurat_v3', subset=False) # Use seurat_v3 or another flavor if needed
        highly_variable_genes = adata.var.highly_variable[adata.var.highly_variable].index
        df_hvg = df.loc[highly_variable_genes] # Filter original df (genes as rows)
        logging.info(f"Selected {len(highly_variable_genes)} highly variable genes. New shape: {df_hvg.shape}")
        return df_hvg
    except Exception as e:
        logging.error(f"Error during HVG selection: {e}. Ensure data has sufficient variance and no NaNs/Infs.")
        logging.warning("Skipping HVG selection due to error.")
        return df # Return original df if HVG fails

def save_data(df, filepath):
    """Saves the transformed data to a CSV file."""
    logging.info(f"Saving transformed data to {filepath}...")
    try:
        df.to_csv(filepath)
        logging.info("Data saved successfully.")
    except Exception as e:
        logging.error(f"Error saving data: {e}")

def main():
    parser = argparse.ArgumentParser(description="Transform gene expression matrix.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input gene expression CSV file (genes x neurons).")
    parser.add_argument("-o", "--output", required=True, help="Path to save the transformed gene expression CSV file.")
    parser.add_argument("--drop_sparse", action="store_true", help="Enable dropping sparse genes (default: >50% zeros).")
    parser.add_argument("--sparse_threshold", type=float, default=0.5, help="Threshold for dropping sparse genes (proportion of zeros).")
    parser.add_argument("--log_transform", action="store_true", help="Enable log1p transformation.")
    parser.add_argument("--mean_impute", action="store_true", help="Enable mean imputation for zeros (requires log_transform to be meaningful for some downstream analyses, but can be run independently).")
    parser.add_argument("--impute_threshold", type=float, default=0.5, help="Minimum presence threshold for mean imputation.")
    parser.add_argument("--select_hvg", action="store_true", help="Enable highly variable gene selection.")
    parser.add_argument("--n_top_genes", type=int, default=2000, help="Number of top highly variable genes to select.")

    args = parser.parse_args()

    # Load data
    df = load_data(args.input)
    if df is None:
        return # Exit if loading failed

    # Apply transformations sequentially based on flags
    if args.drop_sparse:
        df = drop_sparse_genes(df, threshold=args.sparse_threshold)

    if args.log_transform:
        df = log_transform(df)

    if args.mean_impute:
        df = mean_impute_zeros(df, min_presence_threshold=args.impute_threshold)

    if args.select_hvg:
        # Ensure HVG selection happens after log transform if both are enabled, as it's standard practice.
        if not args.log_transform:
             logging.warning("Running HVG selection without prior log transformation. This might not be standard.")
        df = select_highly_variable_genes(df, n_top_genes=args.n_top_genes)

    # Save transformed data
    save_data(df, args.output)

if __name__ == "__main__":
    main()