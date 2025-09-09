"""
Utility functions for model initialization with proper gene vocabulary mapping.

This module provides convenience functions to initialize models with appropriate
gene symbol mappings for pretrained token encoders (scBERT, Geneformer).
"""

from models.token_encoder import TokenEncoderFactory
from data.data_load import load_transcriptome


def create_model_with_gene_mapping(model_class, encoder_type, d_model, 
                                  parcellation='S100', gene_list='0.2', 
                                  dataset='AHBA', omit_subcortical=False, 
                                  hemisphere='both', impute_strategy='mirror_interpolate',
                                  sort_genes='refgenome', **model_kwargs):
    """
    Create a model with properly configured gene vocabulary mapping.
    
    This function automatically handles the gene symbol loading and vocabulary
    mapping setup needed for pretrained token encoders like scBERT and Geneformer.
    
    Args:
        model_class: Model class to instantiate (e.g., TransformerConnectomeModel)
        encoder_type (str): Token encoder type ('scbert', 'geneformer', 'linear', etc.)
        d_model (int): Model dimension
        parcellation (str): Brain parcellation ('S100', 'S400'). Default: 'S100'
        gene_list (str): Gene subset ('0.2', '1', etc.). Default: '0.2'  
        dataset (str): Source dataset ('AHBA', etc.). Default: 'AHBA'
        omit_subcortical (bool): Exclude subcortical regions. Default: False
        hemisphere (str): Brain hemisphere ('both', 'left', 'right'). Default: 'both'
        impute_strategy (str): Imputation method. Default: 'mirror_interpolate'
        sort_genes (str): Gene sorting method. Default: 'refgenome'
        **model_kwargs: Additional arguments passed to model constructor
        
    Returns:
        tuple: (model_instance, gene_symbols, token_dim)
            - model_instance: Initialized model with proper gene mapping
            - gene_symbols: List of gene symbols used
            - token_dim: Number of genes (token dimension)
            
    Example:
        >>> from models.transformer_models import TransformerConnectomeModel
        >>> model, genes, token_dim = create_model_with_gene_mapping(
        ...     TransformerConnectomeModel,
        ...     encoder_type='scbert',
        ...     d_model=512,
        ...     parcellation='S100',
        ...     gene_list='0.2'
        ... )
        >>> print(f"Model initialized with {len(genes)} genes")
    """
    
    # Load gene symbols from transcriptome data
    print(f"Loading gene symbols for {encoder_type} model...")
    _, gene_symbols = load_transcriptome(
        parcellation=parcellation,
        gene_list=gene_list,
        dataset=dataset, 
        omit_subcortical=omit_subcortical,
        hemisphere=hemisphere,
        impute_strategy=impute_strategy,
        sort_genes=sort_genes,
        return_valid_genes=True
    )
    
    token_dim = len(gene_symbols)
    print(f"Loaded {token_dim} gene symbols: {gene_symbols[:3]}...{gene_symbols[-3:] if len(gene_symbols) > 6 else ''}")
    
    # Create token encoder with gene mapping
    token_encoder = TokenEncoderFactory.create(
        encoder_type=encoder_type,
        token_dim=token_dim,
        d_model=d_model,
        gene_symbols=gene_symbols
    )
    
    # Initialize model with the configured token encoder
    model = model_class(
        token_encoder=token_encoder,
        d_model=d_model,
        **model_kwargs
    )
    
    return model, gene_symbols, token_dim


def load_data_and_create_model(model_class, encoder_type, d_model,
                              parcellation='S100', gene_list='0.2',
                              dataset='AHBA', omit_subcortical=False,
                              hemisphere='both', impute_strategy='mirror_interpolate', 
                              sort_genes='refgenome', **model_kwargs):
    """
    Load transcriptome data and create model with matching gene vocabulary.
    
    This is a complete convenience function that loads both the data and creates
    a properly configured model with gene vocabulary mapping.
    
    Args:
        Same as create_model_with_gene_mapping
        
    Returns:
        tuple: (model, gene_data, gene_symbols, token_dim)
            - model: Initialized model with proper gene mapping
            - gene_data: Loaded transcriptome data (regions x genes)
            - gene_symbols: List of gene symbols used
            - token_dim: Number of genes
            
    Example:
        >>> model, data, genes, token_dim = load_data_and_create_model(
        ...     TransformerConnectomeModel,
        ...     encoder_type='geneformer', 
        ...     d_model=768,
        ...     parcellation='S400',
        ...     gene_list='0.2'
        ... )
        >>> print(f"Data shape: {data.shape}, Genes: {len(genes)}")
    """
    
    print("Loading transcriptome data and creating model...")
    
    # Load the actual transcriptome data
    gene_data = load_transcriptome(
        parcellation=parcellation,
        gene_list=gene_list,
        dataset=dataset,
        omit_subcortical=omit_subcortical, 
        hemisphere=hemisphere,
        impute_strategy=impute_strategy,
        sort_genes=sort_genes,
        return_valid_genes=False
    )
    
    # Create model with gene mapping
    model, gene_symbols, token_dim = create_model_with_gene_mapping(
        model_class=model_class,
        encoder_type=encoder_type,
        d_model=d_model,
        parcellation=parcellation,
        gene_list=gene_list,
        dataset=dataset,
        omit_subcortical=omit_subcortical,
        hemisphere=hemisphere, 
        impute_strategy=impute_strategy,
        sort_genes=sort_genes,
        **model_kwargs
    )
    
    print(f"Data loaded: {gene_data.shape} (regions x genes)")
    print(f"Model created with {token_dim} gene vocabulary mapping")
    
    return model, gene_data, gene_symbols, token_dim


def verify_gene_mapping(encoder, gene_symbols, show_unmapped=True):
    """
    Verify that gene vocabulary mapping is working correctly.
    
    Args:
        encoder: Token encoder instance (scBERT or Geneformer)
        gene_symbols (list): List of gene symbols used
        show_unmapped (bool): Whether to display unmapped genes
        
    Returns:
        dict: Mapping statistics
    """
    if not hasattr(encoder, 'gene_mapper') or encoder.gene_mapper is None:
        return {
            'status': 'no_mapping',
            'message': 'No gene mapping configured - using identity mapping'
        }
    
    mapped_count = 0
    unmapped_count = 0
    
    if hasattr(encoder, 'gene_index_mapping'):
        for gene_idx, mapped_gene in encoder.gene_index_mapping.items():
            if mapped_gene is not None:
                mapped_count += 1
            else:
                unmapped_count += 1
    
    if hasattr(encoder, 'unmapped_genes') and show_unmapped and encoder.unmapped_genes:
        print(f"Unmapped genes ({len(encoder.unmapped_genes)}): {encoder.unmapped_genes[:10]}...")
    
    mapping_rate = mapped_count / (mapped_count + unmapped_count) if (mapped_count + unmapped_count) > 0 else 0
    
    stats = {
        'status': 'mapped',
        'total_genes': len(gene_symbols),
        'mapped_genes': mapped_count,
        'unmapped_genes': unmapped_count,
        'mapping_rate': mapping_rate,
        'message': f"Vocabulary mapping: {mapped_count}/{len(gene_symbols)} genes mapped ({mapping_rate:.1%})"
    }
    
    print(stats['message'])
    return stats