"""
Gene vocabulary mapping utilities for pretrained models.

This module provides functionality to map between different gene identifier formats
used by the project (HGNC gene symbols) and pretrained models (scBERT, Geneformer).
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import requests
import json
import warnings


class GeneMapper:
    """Base class for gene identifier mapping."""
    
    def __init__(self):
        self.mapping_cache = {}
    
    def map_genes(self, gene_symbols: List[str]) -> Dict[str, Optional[str]]:
        """Map gene symbols to target vocabulary."""
        raise NotImplementedError
    
    def get_unmapped_genes(self, gene_symbols: List[str]) -> Set[str]:
        """Get genes that cannot be mapped to target vocabulary."""
        mapping = self.map_genes(gene_symbols)
        return {gene for gene, mapped in mapping.items() if mapped is None}


class ScBERTGeneMapper(GeneMapper):
    """Maps HGNC gene symbols to scBERT vocabulary (NCBI gene symbols, Jan 2020)."""
    
    def __init__(self, cache_dir: str = "/scratch/sg8603/gene_mapping_cache"):
        super().__init__()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, "scbert_gene_mapping.pkl")
        
        # Load cached mapping if available
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.mapping_cache = pickle.load(f)
                print(f"Loaded scBERT gene mapping cache with {len(self.mapping_cache)} entries")
            except Exception as e:
                print(f"Warning: Could not load scBERT mapping cache: {e}")
                self.mapping_cache = {}
    
    def map_genes(self, gene_symbols: List[str]) -> Dict[str, Optional[str]]:
        """
        Map HGNC gene symbols to scBERT vocabulary.
        
        For scBERT, the vocabulary is based on NCBI gene symbols from Jan 2020.
        Since both input and target are gene symbols, we primarily do identity mapping
        with some normalization and handling of deprecated symbols.
        """
        mapping = {}
        unmapped_genes = []
        
        for gene in gene_symbols:
            if gene in self.mapping_cache:
                mapping[gene] = self.mapping_cache[gene]
            else:
                unmapped_genes.append(gene)
        
        # For unmapped genes, try identity mapping first (most common case)
        if unmapped_genes:
            for gene in unmapped_genes:
                # For scBERT, we assume most HGNC symbols map directly
                # This is a reasonable assumption since scBERT uses gene symbols
                normalized_gene = self._normalize_gene_symbol(gene)
                mapping[gene] = normalized_gene
                self.mapping_cache[gene] = normalized_gene
            
            # Save updated cache
            self._save_cache()
        
        return mapping
    
    def _normalize_gene_symbol(self, gene_symbol: str) -> str:
        """Normalize gene symbol for scBERT compatibility."""
        # Basic normalization - remove common suffixes/prefixes that might cause issues
        normalized = gene_symbol.strip().upper()
        
        # Handle some common gene symbol variations
        # This is a simplified approach - in practice, you might want more sophisticated mapping
        return normalized
    
    def _save_cache(self):
        """Save mapping cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.mapping_cache, f)
        except Exception as e:
            print(f"Warning: Could not save scBERT mapping cache: {e}")


class GeneformerGeneMapper(GeneMapper):
    """Maps HGNC gene symbols to Geneformer vocabulary (Ensembl gene IDs)."""
    
    def __init__(self, cache_dir: str = "/scratch/sg8603/gene_mapping_cache"):
        super().__init__()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, "geneformer_gene_mapping.pkl")
        
        # Load static mapping file first (preferred method)
        self.static_mapping = self._load_static_mapping()
        
        # Load cached mapping if available  
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.mapping_cache = pickle.load(f)
                print(f"Loaded Geneformer gene mapping cache with {len(self.mapping_cache)} entries")
            except Exception as e:
                print(f"Warning: Could not load Geneformer mapping cache: {e}")
                self.mapping_cache = {}
        
        # Show mapping source information
        if self.static_mapping:
            print(f"Using static gene mapping with {len(self.static_mapping)} entries")
        else:
            print("Warning: No static mapping found, will use API fallback (may be unreliable)")
    
    def _load_static_mapping(self) -> Dict[str, str]:
        """Load static HGNC -> Ensembl mapping from downloaded file."""
        static_mapping_paths = [
            "./data/gene_mappings/hgnc_to_ensembl_mapping.pkl",
            "./data/gene_mappings/backup_hgnc_to_ensembl_mapping.pkl",
            "/scratch/sg8603/gene_mapping_cache/hgnc_to_ensembl_mapping.pkl"
        ]
        
        for path in static_mapping_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        mapping = pickle.load(f)
                    print(f"Loaded static gene mapping from {path}")
                    return mapping
                except Exception as e:
                    print(f"Failed to load static mapping from {path}: {e}")
                    continue
        
        # If no static mapping found, create basic fallback
        print("No static mapping found, creating basic fallback...")
        try:
            return self._create_fallback_mapping()
        except Exception as e:
            print(f"Failed to create fallback mapping: {e}")
            return {}
    
    def _create_fallback_mapping(self) -> Dict[str, str]:
        """Create a basic mapping for common genes as fallback."""
        simple_mappings = {
            'A1BG': 'ENSG00000121410', 'A2M': 'ENSG00000175899', 'AAAS': 'ENSG00000094914',
            'AACS': 'ENSG00000081760', 'AADAT': 'ENSG00000109576', 'AAK1': 'ENSG00000115977',
            'AAMP': 'ENSG00000087884', 'AARS': 'ENSG00000090861', 'ABAT': 'ENSG00000183044',
            'ABCA1': 'ENSG00000165029', 'ACTB': 'ENSG00000075624', 'GAPDH': 'ENSG00000111640',
            'TUBB': 'ENSG00000196230', 'RPL13A': 'ENSG00000142676', 'ALB': 'ENSG00000163631',
            'APOE': 'ENSG00000130203', 'BDNF': 'ENSG00000176697', 'EGFR': 'ENSG00000146648',
            'MYC': 'ENSG00000136997', 'TP53': 'ENSG00000141510', 'VEGFA': 'ENSG00000112715'
        }
        
        # Save it for next time
        try:
            output_dir = "./data/gene_mappings/"
            os.makedirs(output_dir, exist_ok=True)
            mapping_file = os.path.join(output_dir, "fallback_hgnc_to_ensembl_mapping.pkl")
            with open(mapping_file, 'wb') as f:
                pickle.dump(simple_mappings, f)
            print(f"Created fallback mapping with {len(simple_mappings)} genes")
        except Exception as e:
            print(f"Could not save fallback mapping: {e}")
        
        return simple_mappings
    
    def map_genes(self, gene_symbols: List[str]) -> Dict[str, Optional[str]]:
        """
        Map HGNC gene symbols to Ensembl gene IDs for Geneformer.
        
        Uses static mapping file first, then cache, then API fallback.
        """
        mapping = {}
        unmapped_genes = []
        
        # Step 1: Check static mapping first (most reliable)
        for gene in gene_symbols:
            if self.static_mapping and gene in self.static_mapping:
                mapping[gene] = self.static_mapping[gene]
            elif gene in self.mapping_cache:
                mapping[gene] = self.mapping_cache[gene]
            else:
                unmapped_genes.append(gene)
        
        static_mapped = len(gene_symbols) - len(unmapped_genes)
        if static_mapped > 0:
            print(f"Static mapping found {static_mapped}/{len(gene_symbols)} genes")
        
        # Step 2: Query APIs for unmapped genes (fallback only)
        if unmapped_genes:
            if self.static_mapping:
                print(f"Static mapping available but {len(unmapped_genes)} genes not found")
                print(f"Missing genes (first 10): {unmapped_genes[:10]}")
                # Don't use API fallback if we have static mapping - just mark as unmapped
                for gene in unmapped_genes:
                    mapping[gene] = None
                    self.mapping_cache[gene] = None
            else:
                print(f"Querying APIs for {len(unmapped_genes)} unmapped genes...")
                biomart_mapping = self._query_biomart(unmapped_genes)
                
                for gene in unmapped_genes:
                    ensembl_id = biomart_mapping.get(gene)
                    mapping[gene] = ensembl_id
                    self.mapping_cache[gene] = ensembl_id
                
                # Save updated cache
                self._save_cache()
                
                mapped_count = sum(1 for v in biomart_mapping.values() if v is not None)
                print(f"API mapping: {mapped_count}/{len(unmapped_genes)} genes mapped")
        
        total_mapped = sum(1 for v in mapping.values() if v is not None)
        print(f"Total mapping result: {total_mapped}/{len(gene_symbols)} genes mapped to Ensembl IDs")
        
        return mapping
    
    def _query_biomart(self, gene_symbols: List[str]) -> Dict[str, Optional[str]]:
        """Query Ensembl biomart to convert gene symbols to Ensembl IDs."""
        mapping = {}
        
        # Try multiple API endpoints and methods
        api_methods = [
            self._query_ensembl_rest,
            self._query_biomart_xml,
            self._query_mygene_fallback
        ]
        
        for method in api_methods:
            try:
                print(f"Trying {method.__name__}...")
                mapping = method(gene_symbols)
                
                # Check if we got reasonable results
                mapped_count = sum(1 for v in mapping.values() if v is not None)
                if mapped_count > 0:
                    print(f"Success with {method.__name__}: {mapped_count}/{len(gene_symbols)} genes mapped")
                    return mapping
                else:
                    print(f"{method.__name__} returned no mappings, trying next method...")
                    
            except Exception as e:
                print(f"{method.__name__} failed: {e}, trying next method...")
                continue
        
        # If all methods fail, return empty mapping
        print("All API methods failed, returning unmapped genes")
        return {gene: None for gene in gene_symbols}
    
    def _query_ensembl_rest(self, gene_symbols: List[str]) -> Dict[str, Optional[str]]:
        """Query Ensembl REST API (simpler endpoint)."""
        mapping = {}
        
        # Use the lookup endpoint which is more reliable
        server = "https://rest.ensembl.org"
        
        # Process in smaller batches for REST API
        batch_size = 50
        for i in range(0, len(gene_symbols), batch_size):
            batch = gene_symbols[i:i + batch_size]
            
            # Create POST request data
            genes_data = {"symbols": batch}
            
            response = requests.post(
                f"{server}/lookup/symbol/homo_sapiens",
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                json=genes_data,
                timeout=60
            )
            
            if response.status_code == 200:
                results = response.json()
                for gene in batch:
                    if gene in results:
                        mapping[gene] = results[gene].get('id')
                    else:
                        mapping[gene] = None
            else:
                print(f"REST API batch {i//batch_size + 1} failed with status {response.status_code}")
                for gene in batch:
                    mapping[gene] = None
        
        return mapping
    
    def _query_biomart_xml(self, gene_symbols: List[str]) -> Dict[str, Optional[str]]:
        """Original biomart XML query method."""
        mapping = {}
        
        # Biomart REST API endpoint
        server = "https://rest.ensembl.org"
        
        # Process genes in batches to avoid overwhelming the API
        batch_size = 100
        for i in range(0, len(gene_symbols), batch_size):
            batch = gene_symbols[i:i + batch_size]
            
            # Create XML query for biomart
            xml_query = self._create_biomart_query(batch)
            
            # Make request
            response = requests.post(
                f"{server}/biomart/martservice",
                data=xml_query,
                headers={"Content-Type": "application/xml"},
                timeout=60
            )
            
            if response.status_code == 200:
                # Parse response
                batch_mapping = self._parse_biomart_response(response.text, batch)
                mapping.update(batch_mapping)
            else:
                print(f"Biomart XML batch {i//batch_size + 1} failed with status {response.status_code}")
                for gene in batch:
                    mapping[gene] = None
        
        return mapping
    
    def _query_mygene_fallback(self, gene_symbols: List[str]) -> Dict[str, Optional[str]]:
        """Fallback method using mygene.info API."""
        mapping = {}
        
        try:
            import requests
            
            # MyGene.info API endpoint  
            server = "https://mygene.info/v3"
            
            # Process in batches
            batch_size = 100
            for i in range(0, len(gene_symbols), batch_size):
                batch = gene_symbols[i:i + batch_size]
                
                # Query mygene.info
                params = {
                    'q': ','.join(batch),
                    'scopes': 'symbol',
                    'fields': 'ensembl.gene',
                    'species': 'human'
                }
                
                response = requests.post(
                    f"{server}/query",
                    data=params,
                    timeout=60
                )
                
                if response.status_code == 200:
                    results = response.json()
                    
                    # Create mapping from results
                    for gene in batch:
                        mapping[gene] = None  # Default to unmapped
                    
                    for result in results:
                        if 'query' in result and 'ensembl' in result:
                            gene_symbol = result['query']
                            ensembl_data = result['ensembl']
                            
                            # Handle both single ensembl ID and list of IDs
                            if isinstance(ensembl_data, list) and len(ensembl_data) > 0:
                                ensembl_id = ensembl_data[0].get('gene')
                            elif isinstance(ensembl_data, dict):
                                ensembl_id = ensembl_data.get('gene')
                            else:
                                ensembl_id = None
                                
                            if ensembl_id and gene_symbol in mapping:
                                mapping[gene_symbol] = ensembl_id
                else:
                    print(f"MyGene batch {i//batch_size + 1} failed with status {response.status_code}")
                    for gene in batch:
                        mapping[gene] = None
            
        except ImportError:
            raise Exception("requests not available for mygene fallback")
        
        return mapping
    
    def _create_biomart_query(self, gene_symbols: List[str]) -> str:
        """Create XML query for biomart."""
        genes_filter = ','.join(gene_symbols)
        
        xml_query = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="0" count="" datasetConfigVersion="0.6">
    <Dataset name="hsapiens_gene_ensembl" interface="default">
        <Filter name="hgnc_symbol" value="{genes_filter}"/>
        <Attribute name="hgnc_symbol"/>
        <Attribute name="ensembl_gene_id"/>
    </Dataset>
</Query>"""
        
        return xml_query
    
    def _parse_biomart_response(self, response_text: str, query_genes: List[str]) -> Dict[str, Optional[str]]:
        """Parse biomart TSV response."""
        mapping = {gene: None for gene in query_genes}  # Initialize all as unmapped
        
        lines = response_text.strip().split('\n')
        for line in lines:
            if line.strip():
                parts = line.split('\t')
                if len(parts) >= 2:
                    gene_symbol = parts[0]
                    ensembl_id = parts[1]
                    if gene_symbol in mapping and ensembl_id:
                        mapping[gene_symbol] = ensembl_id
        
        return mapping
    
    def _save_cache(self):
        """Save mapping cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.mapping_cache, f)
        except Exception as e:
            print(f"Warning: Could not save Geneformer mapping cache: {e}")


def create_gene_index_mapping(gene_symbols: List[str], mapper: GeneMapper) -> Tuple[Dict[int, Optional[str]], List[str]]:
    """
    Create mapping from gene indices to target vocabulary.
    
    Args:
        gene_symbols: List of gene symbols in order (indices correspond to positions)
        mapper: Gene mapper instance
        
    Returns:
        Tuple of (index_to_target_mapping, unmapped_genes_list)
    """
    symbol_mapping = mapper.map_genes(gene_symbols)
    
    index_mapping = {}
    unmapped_genes = []
    
    for i, gene_symbol in enumerate(gene_symbols):
        target_id = symbol_mapping.get(gene_symbol)
        index_mapping[i] = target_id
        if target_id is None:
            unmapped_genes.append(gene_symbol)
    
    return index_mapping, unmapped_genes


def get_geneformer_vocab_size() -> int:
    """Get the vocabulary size for Geneformer model."""
    # Geneformer V2 has ~20K protein-coding genes
    # V1 had ~25K genes including non-coding
    return 20000  # Approximate, should be loaded from actual model if available


def get_scbert_vocab_size() -> int:
    """Get the vocabulary size for scBERT model."""
    # scBERT uses expression binning, not direct gene vocabulary
    # The vocabulary size is determined by the number of expression bins
    return 7  # 5 expression bins + 2 special tokens (as per original implementation)
