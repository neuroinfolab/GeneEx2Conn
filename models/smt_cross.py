from env.imports import *
from models.train_val import train_model
from models.smt_utils import *
from models.smt import BaseTransformerModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm
from flash_attn import flash_attn_func
from torch.cuda.amp import autocast
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Subset

class CrossAttentionBlock(nn.Module):
    """
    FlashAttention Cross-Attention block
    
    This block performs cross-attention with standard dimensions:
    - Queries, Keys, Values map to d_model
    """
    def __init__(self, d_model, nhead, dropout=0.1, use_alibi=False):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.d_model = d_model
        self.dropout = dropout
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(d_model)
    
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
        # Add attention weight storage for analysis
        self.store_attn = False
        self.last_attn_weights = None
        
        self.use_alibi = use_alibi
        if use_alibi:
            slopes = FlashAttentionBlock.build_alibi_slopes(nhead)
            self.register_buffer("alibi_slopes", slopes)
    
    def split_heads(self, x):
        """Split heads for queries, keys and values"""
        return x.view(x.size(0), x.size(1), self.nhead, self.head_dim)
    
    def merge_heads(self, x):
        """Merge heads back to original shape"""
        return x.transpose(1, 2).reshape(x.size(0), x.size(2), -1)

    def forward(self, queries, keys, values):
        """
        Args:
            queries: (B, seq_len_q, d_model) - query tokens
            keys: (B, seq_len_kv, d_model) - key tokens
            values: (B, seq_len_kv, d_model) - value tokens
        
        Returns:
            output: (B, seq_len_q, d_model) - attended output
        """
        with autocast(dtype=torch.bfloat16):
            residual = queries

            # Project queries, keys, and values
            q = self.q_proj(queries)  # (B, seq_len_q, d_model)
            k = self.k_proj(keys)  # (B, seq_len_kv, d_model)
            v = self.v_proj(values)  # (B, seq_len_kv, d_model)
            
            # Reshape for multi-head attention
            # flash_attn_func expects (B, seqlen, nhead, head_dim) format
            q = self.split_heads(q)  # (B, seq_len_q, nhead, head_dim)
            k = self.split_heads(k)  # (B, seq_len_kv, nhead, head_dim)
            v = self.split_heads(v)  # (B, seq_len_kv, nhead, head_dim)
            
            if self.store_attn:
                # Use manual attention computation to get weights
                q_reshaped = q.transpose(1, 2)  # (B, nhead, seq_len_q, head_dim)
                k_reshaped = k.transpose(1, 2)  # (B, nhead, seq_len_kv, head_dim)
                v_reshaped = v.transpose(1, 2)  # (B, nhead, seq_len_kv, head_dim)
                
                attn_output, attn_weights = scaled_dot_product_cross_attention_with_weights(
                    q_reshaped, k_reshaped, v_reshaped, dropout_p=0.0, apply_dropout=False
                )
                self.last_attn_weights = attn_weights.detach().cpu()
                attn_output = self.merge_heads(attn_output)
            else:
                # Flash attention (cross-attention)
                attn_output = flash_attn_func(
                    q, k, v,
                    dropout_p=self.dropout,
                    causal=False,
                )  # (B, seq_len_q, nhead, head_dim)
                
                # Reshape back: (B, seq_len_q, d_model)
                attn_output = attn_output.reshape(attn_output.size(0), attn_output.size(1), self.d_model)
            
            # Apply dropout and residual
            attn_output = self.attn_dropout(attn_output)
            x = self.attn_norm(residual + attn_output)
            
            # Feed-forward with residual
            residual = x
            x = self.ffn(x)
            x = self.ffn_norm(residual + x)
            
            return x

class BaseGeneVecEncoder(nn.Module):
    """
    Base class for gene vector encoders with shared functionality.
    
    Provides common gene vector loading and region encoding functionality
    that can be shared across different attention mechanisms.
    """
    def __init__(self, valid_genes, genevec_type='gene2vec', expression_bins=5, d_model=32, device=None):
        super().__init__()
        self.valid_genes = valid_genes
        self.genevec_type = genevec_type
        self.expression_bins = expression_bins
        self.d_model = d_model
        self.device = device
        self._load_genevec_embeddings(genevec_type)
        
        # Gene and expression projections (moved from individual encoders)
        self.genevec_projection = nn.Linear(self.genevec_dim, d_model)
        self.bin_embedding = nn.Embedding(expression_bins, d_model)
        nn.init.normal_(self.bin_embedding.weight, mean=0.0, std=0.1)

        self.scalar_projection = nn.Linear(1, d_model)
        self.combined_projection = nn.Linear(2*d_model, d_model)

    def _load_genevec_embeddings(self, genevec_type='gene2vec'):
        """Load gene vector embeddings and create lookup table aligned with valid_genes
        Args:
            type: 'gene2vec', 'coexpression', or 'one_hot'
        """
        if genevec_type == 'gene2vec':
            # Load Gene2Vec dataframe
            self.gene2vec_path = '/scratch/asr655/neuroinformatics/GeneEx2Conn/data/gene_emb/gene2vec_dim_200_iter_9.txt'
            gene2vec_df = pd.read_csv(self.gene2vec_path, sep='\s+', header=None, index_col=0)
            
            # Find intersection between Gene2Vec genes and valid genes
            gene2vec_genes = set(gene2vec_df.index.tolist())
            valid_genes_set = set(self.valid_genes)
            shared_genes = list(gene2vec_genes.intersection(valid_genes_set))
        
            # Get indices of shared genes in the original valid_genes list
            self.shared_gene_indices = [self.valid_genes.index(gene) for gene in shared_genes if gene in self.valid_genes]
            self.shared_genes = [self.valid_genes[i] for i in self.shared_gene_indices]
            
            # Create Gene2Vec lookup matrix in the order of shared genes
            gene2vec_matrix = np.zeros((len(self.shared_genes), 200))
            for i, gene in enumerate(self.shared_genes):
                if gene in gene2vec_df.index:
                    gene2vec_matrix[i] = gene2vec_df.loc[gene].values
            print(f"Loaded Gene2Vec embeddings: {len(self.shared_genes)} genes with 200-dimensional embeddings")
            print(f"Gene overlap: {len(self.shared_genes)}/{len(self.valid_genes)} valid genes have Gene2Vec embeddings")
            
            genevec_matrix = gene2vec_matrix
            self.genevec_dim = genevec_matrix.shape[1]
        elif genevec_type == 'one_hot':
            # Create one-hot encoding matrix for valid genes
            num_genes = len(self.valid_genes)
            one_hot_matrix = np.eye(num_genes)
            
            # Use all valid genes since no external embedding lookup needed
            self.shared_genes = self.valid_genes
            self.shared_gene_indices = list(range(num_genes))
            
            genevec_matrix = one_hot_matrix
            self.genevec_dim = num_genes
            
            print(f"Created one-hot embeddings: {num_genes} genes with {num_genes}-dimensional embeddings")
            print("Using all valid genes for one-hot encoding")
        elif genevec_type == 'coexpression':
            # Load coexpression vector dataframe
            self.coexpression_path = '/scratch/asr655/neuroinformatics/GeneEx2Conn/data/gene_emb/genevec_cov.txt'
            coexpression_df = pd.read_csv(self.coexpression_path, sep=' ', skiprows=1, index_col=0)
            
            # Use all valid genes since coexpression matrix contains all genes
            self.shared_genes = self.valid_genes
            self.shared_gene_indices = list(range(len(self.valid_genes)))
            
            # Create coexpression lookup matrix in order of valid genes
            coexpression_matrix = np.zeros((len(self.valid_genes), 7380))
            for i, gene in enumerate(self.valid_genes):
                if gene in coexpression_df.index:
                    coexpression_matrix[i] = coexpression_df.loc[gene].values
                    
            genevec_matrix = coexpression_matrix
            self.genevec_dim = 7380  # Fixed dimension for coexpression vectors
            
            print(f"Loaded coexpression embeddings: {len(self.valid_genes)} genes with 7380-dimensional embeddings")
            print("Using all valid genes for coexpression embeddings")
        
        # Store as buffer (not trainable)
        self.register_buffer('genevec_embeddings', torch.FloatTensor(genevec_matrix))

    def encode_region(self, gene_expression):
        """
        Base region encoding method - creates gene token embeddings.
        
        Args:
            gene_expression: (B, num_genes) - gene expression values
            
        Returns:
            region_embedding: (B, num_shared_genes, d_model) - gene token embeddings
        """
        B, num_genes = gene_expression.shape
        
        # Subset to genes with genevec embeddings
        gene_expression_subset = gene_expression[:, self.shared_gene_indices]  # (B, num_shared_genes)
        
        # Get genevec embeddings, expand to batch, and project to d_model
        genevec_emb = self.genevec_embeddings.unsqueeze(0).expand(B, -1, -1)
        genevec_emb = self.genevec_projection(genevec_emb)  # (B, num_shared_genes, d_model)
        
        # Choose between binning and scalar projection based on expression_bins
        if self.expression_bins < 100:
            # Quantile-based binning of gene expression values
            if not hasattr(self, 'quantile_bin_edges') or self.quantile_bin_edges.shape[0] != self.expression_bins - 1:
                # Compute quantile edges across the model's relevant genes globally just once
                # For efficiency, use a representative input or precomputed stats; here estimated from the batch
                with torch.no_grad():
                    # Flatten gene_expression_subset across batch
                    gene_expr_flat = gene_expression_subset.detach().reshape(-1)
                    self.quantile_bin_edges = torch.quantile(
                        gene_expr_flat, 
                        torch.linspace(0, 1, self.expression_bins + 1, device=gene_expression.device)[1:-1]
                    )
            # Use quantile bin edges for bucketize
            gene_expr_binned = torch.bucketize(
                gene_expression_subset, 
                self.quantile_bin_edges
            )  # (B, num_shared_genes)

            # Clamp to valid bin indices
            gene_expr_binned = torch.clamp(gene_expr_binned, 0, self.expression_bins - 1)
            
            # Get bin embeddings
            expr_emb = self.bin_embedding(gene_expr_binned)  # (B, num_shared_genes, d_model)
        else:
            # Continuous scalar projection strategy
            expr_emb = self.scalar_projection(gene_expression_subset.unsqueeze(-1))  # (B, num_shared_genes, d_model)
        
        # Concatenate embeddings
        embeddings = torch.cat([genevec_emb, expr_emb], dim=-1)  # (B, num_shared_genes, 2*d_model)
        region_embedding = self.combined_projection(embeddings)  # Project back to d_model dimension

        return region_embedding

    def apply_pooling(self, x):
        """
        Apply pooling to attention output based on pooling_mode.
        
        Args:
            x: (B, seq_len, d_model) - attention output
            
        Returns:
            pooled: (B, d_model) or (B, num_tokens) - pooled representation
        """
        if self.pooling_mode == 'mean':
            return x.mean(dim=1)  # (B, d_model)
        elif self.pooling_mode == 'attention':
            return self.attention_pooling(x)  # (B, d_model)
        elif self.pooling_mode == 'linear':
            return self.linear_pooling(x).squeeze(-1)  # (B, num_tokens)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling_mode}")

    def get_attention_layers(self):
        """
        Get the attention layers for this encoder.
        Override in subclasses to return the appropriate attention layers.
        """
        raise NotImplementedError("Subclasses must implement get_attention_layers()")
    
    def get_sequence_length(self):
        """
        Get the sequence length for attention processing.
        Override in subclasses to return the appropriate sequence length.
        """
        return len(self.shared_genes)
    
    def setup_attention_collection(self):
        """
        Setup attention collection for this encoder type.
        Override in subclasses for encoder-specific attention collection.
        """
        raise NotImplementedError("Subclasses must implement setup_attention_collection()")
    
    def accumulate_attention_weights(self, is_first_batch=False):
        """
        Accumulate attention weights during prediction.
        Override in subclasses for encoder-specific attention accumulation.
        """
        raise NotImplementedError("Subclasses must implement accumulate_attention_weights()")
    
    def process_attention_weights(self, total_batches, save_attn_path):
        """
        Process accumulated attention weights.
        Override in subclasses for encoder-specific attention processing.
        """
        raise NotImplementedError("Subclasses must implement process_attention_weights()")

class CrossAttentionGeneVecEncoder(BaseGeneVecEncoder):
    """
    Cross-Attention encoder with gene vector lookup and value binning.
    
    Each gene is treated as a token with:
    - Gene vector lookup (gene2vec or coexpression vector) ~200D
    - Expression bin embedding (learned projection from one-hot bin)
    - Combined via element-wise addition

    Args:
        valid_genes: List of valid gene names
        expression_bins: Number of discrete expression bins (default: 5)
        d_model: Transformer hidden dimension (default: 128)
        nhead: Number of attention heads (default: 4)
        num_layers: Number of cross-attention layers (default: 4)
        dropout: Dropout rate (default: 0.1)
        use_alibi: Use ALiBi positional biases (default: False)
        pooling_mode: Pooling method - 'mean', 'attention', or 'linear' (default: 'mean')
        device: Device to place model on
    """
    def __init__(self, valid_genes, expression_bins=5, d_model=32, nhead=2, num_layers=2, 
                 dropout=0.1, use_alibi=False, pooling_mode='mean', genevec_type='gene2vec', device=None):
        # Initialize base class with gene vector loading AND layer creation
        super().__init__(valid_genes, genevec_type, expression_bins, d_model, device)
        
        self.pooling_mode = pooling_mode
    
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dropout, use_alibi=use_alibi)
            for _ in range(num_layers)
        ])
        
        # Pooling layers
        if self.pooling_mode == 'attention':
            self.attention_pooling = AttentionPooling(d_model, hidden_dim=32)
        elif self.pooling_mode == 'linear':
            self.linear_pooling = nn.Linear(d_model, 1)
    
    def get_attention_layers(self):
        """Return cross-attention layers for attention collection."""
        return self.cross_attn_layers
    
    def setup_attention_collection(self):
        """Setup attention collection for cross-attention."""
        collect_full_cross_attention_heads(self.cross_attn_layers)
    
    def accumulate_attention_weights(self, is_first_batch=False):
        """Accumulate cross-attention weights."""
        accumulate_cross_attention_weights(self.cross_attn_layers, is_first_batch=is_first_batch)
    
    def process_attention_weights(self, total_batches, save_attn_path):
        """Process accumulated cross-attention weights."""
        return process_full_cross_attention_heads(self.cross_attn_layers, total_batches, save_attn_path, self.get_sequence_length())
    
    def forward(self, gene_expr_i, gene_expr_j, coords_i, coords_j):
        """
        Forward pass with cross-attention.
        
        Args:
            gene_expr_i: (B, num_genes) - region i expression
            gene_expr_j: (B, num_genes) - region j expression
            coords_i: (B, 3) - region i coordinates
            coords_j: (B, 3) - region j coordinates
        
        Returns:
            output: (B, d_model) - pooled representation
        """
        B = gene_expr_i.size(0)
        
        # Encode both regions to gene token sequences
        embeddings_i = self.encode_region(gene_expr_i)  # (B, num_shared_genes, d_model)
        embeddings_j = self.encode_region(gene_expr_j)  # (B, num_shared_genes, d_model)
        
        # Apply cross-attention layers with proper gradient flow
        x = embeddings_i
        for layer in self.cross_attn_layers:
            x = layer(x, embeddings_j, embeddings_j)  # Update x for next layer
        attn_output = x
        
        # Pool the output
        pooled = self.apply_pooling(attn_output)
        
        return pooled

class CrossAttentionGeneVecModel(BaseTransformerModel):
    """
    Cross-Attention GeneVec Model for symmetric edge prediction.
    """
    def __init__(self, input_dim, region_pair_dataset, 
                 expression_bins=5, d_model=128, nhead=4, num_layers=4, 
                 deep_hidden_dims=[512, 256, 128], pooling_mode='mean',
                 use_alibi=False, transformer_dropout=0.1, dropout_rate=0.1, 
                 learning_rate=0.001, weight_decay=0.0, batch_size=512, epochs=100, 
                 aug_prob=0.0, num_workers=2, prefetch_factor=2, cosine_lr=False, aug_style='linear_decay',
                 lambda_sym=0.1, genevec_type='gene2vec', bidirectional=True):
        
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, 
                        num_workers, prefetch_factor, d_model, nhead, num_layers, 
                        deep_hidden_dims=deep_hidden_dims, aug_prob=aug_prob, aug_style='linear_decay')
        
        self.input_dim = input_dim // 2  # Split for two regions
        self.expression_bins = expression_bins
        self.pooling_mode = pooling_mode
        self.use_alibi = use_alibi
        self.cosine_lr = cosine_lr
        self.valid_genes = region_pair_dataset.valid_genes
        self.lambda_sym = lambda_sym
        self.bidirectional = bidirectional
        self.patience = 70 # usually 70
        
        # Setup symmetric loss if lambda_sym > 0
        if self.lambda_sym > 0:
            self.criterion = SymmetricLoss(nn.MSELoss(), lambda_sym=self.lambda_sym)
            print(f"Using SymmetricLoss with lambda_sym={self.lambda_sym}")
        
        # Bidirectional cross-attention encoder
        self.encoder = CrossAttentionGeneVecEncoder(
            valid_genes=self.valid_genes,
            expression_bins=expression_bins,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=transformer_dropout,
            use_alibi=use_alibi,
            pooling_mode=pooling_mode,
            genevec_type=genevec_type,
            device=self.device
        )
        self.encoder = torch.compile(self.encoder)
        
        # MLP prediction head
        if self.pooling_mode == 'linear':
            prev_dim = len(self.encoder.shared_genes)
        elif self.bidirectional:
            prev_dim = 2 * d_model
        else:
            prev_dim = d_model
        self.deep_layers, self.output_layer = self._set_mlp_head(
            prev_dim=prev_dim,
            deep_hidden_dims=deep_hidden_dims,
            dropout_rate=dropout_rate)
        
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in CrossAttentionGeneVec model: {num_params}")
        print(f"Expression bins: {expression_bins}")
        print(f"Pooling mode: {pooling_mode}")
        
        self._setup_optimizer_scheduler(learning_rate, weight_decay, use_cosine=cosine_lr)
    
    def forward(self, x, coords, idx):
        # Split input
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)  # Each: (B, num_genes)
        coords_i, coords_j = torch.chunk(coords, chunks=2, dim=1)  # Each: (B, 3)
        
        # Encode bidirectionally with cross-attention
        pooled_output_j = self.encoder(x_i, x_j, coords_i, coords_j)  # (B, d_model)
        if self.bidirectional:
            pooled_output_i = self.encoder(x_j, x_i, coords_j, coords_i)  # (B, d_model)
            pooled_output = torch.cat([pooled_output_i, pooled_output_j], dim=-1)  # (B, d_model*2)
        else: 
            pooled_output = pooled_output_j
            
        # Predict connectivity via MLP head
        y_pred = self.output_layer(self.deep_layers(pooled_output))
        
        return y_pred.squeeze()
    
    def fit(self, dataset, train_indices, test_indices, save_model=None, verbose=True, wandb_run=None):
        """Shared fit function for all transformer models"""
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, 
                                 pin_memory=True, num_workers=self.num_workers, 
                                 prefetch_factor=self.prefetch_factor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, 
                                pin_memory=True, num_workers=self.num_workers, 
                                prefetch_factor=self.prefetch_factor)
        
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, 
                            self.optimizer, self.patience, scheduler=self.scheduler, 
                            save_model=save_model, verbose=verbose, dataset=dataset, wandb_run=wandb_run)

    def get_attention_heads(self, avg_attn):
        """
        Simple helper to extract individual heads and global average from attention tensor
        
        Args:
            avg_attn: attention tensor from predict() method
            
        Returns:
            tuple: (individual_heads_list, global_average)
        """
        if avg_attn is None:
            return None, None
            
        # Convert to numpy if needed
        if hasattr(avg_attn, 'cpu'):
            avg_attn = avg_attn.cpu().numpy()
        
        # Individual heads as list
        individual_heads = [avg_attn[h] for h in range(avg_attn.shape[0])]
        
        # Global average
        global_average = avg_attn.mean(axis=0)
        
        return individual_heads, global_average


class MixedAttentionGeneVecEncoder(BaseGeneVecEncoder):
    """
    Mixed Self/Cross-Attention encoder with spatially-initialized region embeddings.
    
    Processes both regions in a single sequence allowing for both self-attention 
    (within regions) and cross-attention (between regions) in the same transformer block.
    
    Region embeddings can be initialized in two ways:
    1. 'spatial': From 3D coordinates of actual brain regions
    2. 'none': No region embedding (genes from both regions use same base embedding)
    
    Args:
        valid_genes: List of valid gene names
        expression_bins: Number of discrete expression bins (default: 5)
        d_model: Transformer hidden dimension (default: 32)
        nhead: Number of attention heads (default: 2)
        num_layers: Number of self-attention layers (default: 2)
        dropout: Dropout rate (default: 0.1)
        use_alibi: Use ALiBi positional biases (default: False)
        pooling_mode: Pooling method - 'mean', 'attention', or 'linear' (default: 'mean')
        genevec_type: Gene vector type - 'gene2vec', 'coexpression', or 'one_hot'
        region_emb_init: Region embedding initialization - 'spatial' or 'none'
        device: Device to place model on
    """
    def __init__(self, valid_genes, expression_bins=5, d_model=32, nhead=2, num_layers=2, 
                 dropout=0.1, use_alibi=False, pooling_mode='mean', 
                 genevec_type='gene2vec', region_emb_init='spatial', device=None):
        # Initialize base class with gene vector loading AND layer creation
        super().__init__(valid_genes, genevec_type, expression_bins, d_model, device)
        
        self.pooling_mode = pooling_mode
        self.region_emb_init = region_emb_init
        
        # Region embedding initialization
        if region_emb_init == 'spatial':
            # Project from 3D coordinates to d_model
            self.region_coord_projection = nn.Linear(3, d_model)
            # Initialize with small weights for stability
            nn.init.normal_(self.region_coord_projection.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.region_coord_projection.bias)
        elif region_emb_init == 'none':
            # No region embedding - genes from both regions use same base embedding
            pass
        
        # Self-attention layers (using FlashAttentionBlock from smt_utils)
        self.attn_layers = nn.ModuleList([
            FlashAttentionBlock(d_model, nhead, dropout, use_alibi=use_alibi)
            for _ in range(num_layers)
        ])
        
        # Pooling layers
        if self.pooling_mode == 'attention':
            self.attention_pooling = AttentionPooling(d_model, hidden_dim=32)
        elif self.pooling_mode == 'linear':
            self.linear_pooling = nn.Linear(d_model, 1)

    def get_attention_layers(self):
        """Return self-attention layers for attention collection."""
        return self.attn_layers
    
    def get_sequence_length(self):
        """Return sequence length for mixed attention (2 regions worth of genes)."""
        return len(self.shared_genes) * 2
    
    def setup_attention_collection(self):
        """Setup attention collection for self-attention."""
        collect_full_attention_heads(self.attn_layers)
    
    def accumulate_attention_weights(self, is_first_batch=False):
        """Accumulate self-attention weights."""
        accumulate_attention_weights(self.attn_layers, is_first_batch=is_first_batch)
    
    def process_attention_weights(self, total_batches, save_attn_path):
        """Process accumulated self-attention weights."""
        return process_full_attention_heads(self.attn_layers, total_batches, save_attn_path, self.get_sequence_length())

    def get_region_embeddings_from_coords(self, coords_i, coords_j):
        """
        Get region embeddings from coordinates directly.
        
        Args:
            coords_i: (B, 3) - coordinates of region i
            coords_j: (B, 3) - coordinates of region j
            
        Returns:
            region_emb_i: (B, d_model) - embeddings for region i
            region_emb_j: (B, d_model) - embeddings for region j
        """
        if self.region_emb_init == 'spatial':
            # Normalize coordinates for stability (optional but recommended)
            coords_i_norm = (coords_i - coords_i.mean(dim=0, keepdim=True)) / (coords_i.std(dim=0, keepdim=True) + 1e-6)
            coords_j_norm = (coords_j - coords_j.mean(dim=0, keepdim=True)) / (coords_j.std(dim=0, keepdim=True) + 1e-6)
            
            region_emb_i = self.region_coord_projection(coords_i_norm)  # (B, d_model)
            region_emb_j = self.region_coord_projection(coords_j_norm)  # (B, d_model)
            
            return region_emb_i, region_emb_j
        elif self.region_emb_init == 'none':
            # No region embedding - return zeros
            B = coords_i.size(0)
            zero_emb = torch.zeros(B, self.d_model, device=self.device)
            return zero_emb, zero_emb

    def encode_region_pair(self, gene_expr_i, gene_expr_j, coords_i, coords_j):
        """
        Encode both regions into a single concatenated sequence with region embeddings.
        
        Args:
            gene_expr_i: (B, num_genes) - region i expression
            gene_expr_j: (B, num_genes) - region j expression  
            coords_i: (B, 3) - coordinates of region i
            coords_j: (B, 3) - coordinates of region j
        """
        # Use base class encode_region method for both regions
        embeddings_i = self.encode_region(gene_expr_i)  # (B, num_shared_genes, d_model)
        embeddings_j = self.encode_region(gene_expr_j)  # (B, num_shared_genes, d_model)
        
        # Get region-specific embeddings based on actual brain region coordinates
        region_emb_i, region_emb_j = self.get_region_embeddings_from_coords(coords_i, coords_j)
        
        # Add region embeddings to all gene tokens in each region
        # Broadcast region embeddings across all genes in that region
        region_emb_i_expanded = region_emb_i.unsqueeze(1).expand(-1, len(self.shared_genes), -1)  # (B, num_shared_genes, d_model)
        region_emb_j_expanded = region_emb_j.unsqueeze(1).expand(-1, len(self.shared_genes), -1)  # (B, num_shared_genes, d_model)
        
        embeddings_i = embeddings_i + region_emb_i_expanded  # (B, num_shared_genes, d_model)
        embeddings_j = embeddings_j + region_emb_j_expanded  # (B, num_shared_genes, d_model)
        
        # Concatenate into single sequence: [region_i_genes, region_j_genes]
        combined_embeddings = torch.cat([embeddings_i, embeddings_j], dim=1)  # (B, 2*num_shared_genes, d_model)
        
        return combined_embeddings


    def forward(self, gene_expr_i, gene_expr_j, coords_i, coords_j):
        """
        Forward pass with mixed self/cross-attention and spatial region embeddings.
        
        Args:
            gene_expr_i: (B, num_genes) - region i expression
            gene_expr_j: (B, num_genes) - region j expression
            coords_i: (B, 3) - region i coordinates
            coords_j: (B, 3) - region j coordinates
        """
        B = gene_expr_i.size(0)
        
        # Encode both regions into single sequence with spatial region embeddings
        combined_embeddings = self.encode_region_pair(gene_expr_i, gene_expr_j, coords_i, coords_j)  # (B, 2*num_shared_genes, d_model)
        
        # Apply self-attention layers (allows both self and cross-region attention)
        x = combined_embeddings
        for layer in self.attn_layers:
            x = layer(x)  # Standard self-attention: Q, K, V all come from same sequence
        
        # Pool the output
        pooled = self.apply_pooling(x)
        
        return pooled


class MixedAttentionGeneVecModel(BaseTransformerModel):
    """
    Mixed Attention GeneVec Model for symmetric edge prediction.
    
    Uses a single sequence containing genes from both regions, allowing for
    both self-attention (within regions) and cross-attention (between regions)
    in the same transformer block. Region embeddings can be initialized from
    spatial coordinates or omitted entirely for better generalization.
    """
    def __init__(self, input_dim, region_pair_dataset, 
                 expression_bins=5, d_model=128, nhead=4, num_layers=4, 
                 deep_hidden_dims=[512, 256, 128], pooling_mode='mean',
                 use_alibi=False, transformer_dropout=0.1, dropout_rate=0.1, 
                 learning_rate=0.001, weight_decay=0.0, batch_size=512, epochs=100, 
                 aug_prob=0.0, num_workers=2, prefetch_factor=2, cosine_lr=False, aug_style='linear_decay',
                 lambda_sym=0.1, genevec_type='gene2vec', region_emb_init='spatial'):
        
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, 
                        num_workers, prefetch_factor, d_model, nhead, num_layers, 
                        deep_hidden_dims=deep_hidden_dims, aug_prob=aug_prob, aug_style='linear_decay')
        
        self.input_dim = input_dim // 2  # Split for two regions
        self.expression_bins = expression_bins
        self.pooling_mode = pooling_mode
        self.use_alibi = use_alibi
        self.cosine_lr = cosine_lr
        self.valid_genes = region_pair_dataset.valid_genes
        self.lambda_sym = lambda_sym
        self.region_emb_init = region_emb_init
        self.patience = 70 # usually 70
        
        # Setup symmetric loss if lambda_sym > 0
        if self.lambda_sym > 0:
            self.criterion = SymmetricLoss(nn.MSELoss(), lambda_sym=self.lambda_sym)
            print(f"Using SymmetricLoss with lambda_sym={self.lambda_sym}")
        
        # Mixed attention encoder with spatial region embeddings
        self.encoder = MixedAttentionGeneVecEncoder(
            valid_genes=self.valid_genes,
            expression_bins=expression_bins,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=transformer_dropout,
            use_alibi=use_alibi,
            pooling_mode=pooling_mode,
            genevec_type=genevec_type,
            region_emb_init=region_emb_init,
            device=self.device
        )
        self.encoder = torch.compile(self.encoder)
        
        # MLP prediction head - single representation (no bidirectional concatenation)
        if self.pooling_mode == 'linear':
            prev_dim = len(self.encoder.shared_genes) * 2  # 2 regions worth of genes
        else:
            prev_dim = d_model  # Single pooled representation
        self.deep_layers, self.output_layer = self._set_mlp_head(
            prev_dim=prev_dim,
            deep_hidden_dims=deep_hidden_dims,
            dropout_rate=dropout_rate)
        
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in MixedAttentionGeneVec model: {num_params}")
        print(f"Expression bins: {expression_bins}")
        print(f"Region embedding initialization: {region_emb_init}")
        
        self._setup_optimizer_scheduler(learning_rate, weight_decay, use_cosine=cosine_lr)
    
    def forward(self, x, coords, idx):
        # Split input
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)  # Each: (B, num_genes)
        coords_i, coords_j = torch.chunk(coords, chunks=2, dim=1)  # Each: (B, 3)
        
        # Single forward pass through mixed attention encoder with coordinates
        pooled_output = self.encoder(x_i, x_j, coords_i, coords_j)  # (B, d_model)
        
        # Predict connectivity via MLP head
        y_pred = self.output_layer(self.deep_layers(pooled_output))
        
        return y_pred.squeeze()
    
    def fit(self, dataset, train_indices, test_indices, save_model=None, verbose=True, wandb_run=None):
        """Shared fit function for all transformer models"""
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, 
                                 pin_memory=True, num_workers=self.num_workers, 
                                 prefetch_factor=self.prefetch_factor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, 
                                pin_memory=True, num_workers=self.num_workers, 
                                prefetch_factor=self.prefetch_factor)
        
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, 
                            self.optimizer, self.patience, scheduler=self.scheduler, 
                            save_model=save_model, verbose=verbose, dataset=dataset, wandb_run=wandb_run)

    def get_attention_heads(self, avg_attn):
        """
        Simple helper to extract individual heads and global average from attention tensor
        
        Args:
            avg_attn: attention tensor from predict() method
            
        Returns:
            tuple: (individual_heads_list, global_average)
        """
        if avg_attn is None:
            return None, None
            
        # Convert to numpy if needed
        if hasattr(avg_attn, 'cpu'):
            avg_attn = avg_attn.cpu().numpy()
        
        # Individual heads as list
        individual_heads = [avg_attn[h] for h in range(avg_attn.shape[0])]
        
        # Global average
        global_average = avg_attn.mean(axis=0)
        
        return individual_heads, global_average


class SelfAttentionGeneVecEncoder(BaseGeneVecEncoder):
    """
    Self-Attention encoder for individual ROI processing.
    
    Each ROI is processed independently with self-attention over its gene tokens,
    then pooled to create a fixed-size representation. The two ROI representations
    are then combined for final prediction.
    
    Args:
        valid_genes: List of valid gene names
        expression_bins: Number of discrete expression bins (default: 5)
        d_model: Transformer hidden dimension (default: 128)
        nhead: Number of attention heads (default: 4)
        num_layers: Number of self-attention layers (default: 4)
        dropout: Dropout rate (default: 0.1)
        use_alibi: Use ALiBi positional biases (default: False)
        pooling_mode: Pooling method - 'mean', 'attention', or 'linear' (default: 'mean')
        genevec_type: Gene vector type - 'gene2vec', 'coexpression', or 'one_hot'
        roi_combination: How to combine ROI representations - 'concat', 'add', 'subtract' (default: 'concat')
        device: Device to place model on
    """
    def __init__(self, valid_genes, expression_bins=5, d_model=128, nhead=4, num_layers=4, 
                 dropout=0.1, use_alibi=False, pooling_mode='mean', 
                 genevec_type='gene2vec', roi_combination='concat', device=None):
        # Initialize base class with gene vector loading AND layer creation
        super().__init__(valid_genes, genevec_type, expression_bins, d_model, device)
        
        self.pooling_mode = pooling_mode
        self.roi_combination = roi_combination
        
        # Self-attention layers (using FlashAttentionBlock from smt_utils)
        self.attn_layers = nn.ModuleList([
            FlashAttentionBlock(d_model, nhead, dropout, use_alibi=use_alibi)
            for _ in range(num_layers)
        ])
        
        # Pooling layers
        if self.pooling_mode == 'attention':
            self.attention_pooling = AttentionPooling(d_model, hidden_dim=32)
        elif self.pooling_mode == 'linear':
            self.linear_pooling = nn.Linear(d_model, 1)

    def get_attention_layers(self):
        """Return self-attention layers for attention collection."""
        return self.attn_layers
    
    def setup_attention_collection(self):
        """Setup attention collection for self-attention."""
        collect_full_attention_heads(self.attn_layers)
    
    def accumulate_attention_weights(self, is_first_batch=False):
        """Accumulate self-attention weights."""
        accumulate_attention_weights(self.attn_layers, is_first_batch=is_first_batch)
    
    def process_attention_weights(self, total_batches, save_attn_path):
        """Process accumulated self-attention weights."""
        return process_full_attention_heads(self.attn_layers, total_batches, save_attn_path, self.get_sequence_length())

    def forward(self, gene_expr_i, gene_expr_j, coords_i=None, coords_j=None):
        """
        Forward pass with independent ROI processing and pooling.

        Args:
            gene_expr_i: (B, num_genes) - region i expression
            gene_expr_j: (B, num_genes) - region j expression
            coords_i: (B, 3) - region i coordinates (unused in this model)
            coords_j: (B, 3) - region j coordinates (unused in this model)

        Returns:
            combined_repr: (B, combined_dim) - combined ROI representation
        """
        # --- Process ROI i ---
        # Encode region to gene token embeddings
        embeddings_i = self.encode_region(gene_expr_i)  # (B, num_shared_genes, d_model)

        # Apply self-attention layers
        x_i = embeddings_i
        for layer in self.attn_layers:
            x_i = layer(x_i)  # Self-attention: Q, K, V all from same sequence

        # Pool the output for ROI i
        roi_i_repr = self.apply_pooling(x_i)

        # --- Process ROI j ---
        embeddings_j = self.encode_region(gene_expr_j)  # (B, num_shared_genes, d_model)
        x_j = embeddings_j
        for layer in self.attn_layers:
            x_j = layer(x_j)  # Self-attention: Q, K, V all from same sequence

        # Pool the output for ROI j
        roi_j_repr = self.apply_pooling(x_j)

        # Combine ROI representations
        if self.roi_combination == 'concat':
            combined_repr = torch.cat([roi_i_repr, roi_j_repr], dim=-1)  # (B, 2*d_model)
        elif self.roi_combination == 'add':
            combined_repr = roi_i_repr + roi_j_repr  # (B, d_model)
        else:
            raise ValueError(f"Unknown ROI combination method: {self.roi_combination}")

        return combined_repr

class SelfAttentionGeneVecModel(BaseTransformerModel):
    """
    Self-Attention GeneVec Model for symmetric edge prediction.
    
    Each ROI is processed independently with self-attention over gene tokens,
    then pooled and combined for final connectivity prediction.
    """
    def __init__(self, input_dim, region_pair_dataset, 
                 expression_bins=5, d_model=128, nhead=4, num_layers=4, 
                 deep_hidden_dims=[512, 256, 128], pooling_mode='mean',
                 use_alibi=False, transformer_dropout=0.1, dropout_rate=0.1, 
                 learning_rate=0.001, weight_decay=0.0, batch_size=512, epochs=100, 
                 aug_prob=0.0, num_workers=2, prefetch_factor=2, cosine_lr=False, aug_style='linear_decay',
                 lambda_sym=0.1, genevec_type='gene2vec', roi_combination='concat'):
        
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, 
                        num_workers, prefetch_factor, d_model, nhead, num_layers, 
                        deep_hidden_dims=deep_hidden_dims, aug_prob=aug_prob, aug_style='linear_decay')
        
        self.input_dim = input_dim // 2  # Split for two regions
        self.expression_bins = expression_bins
        self.pooling_mode = pooling_mode
        self.use_alibi = use_alibi
        self.cosine_lr = cosine_lr
        self.valid_genes = region_pair_dataset.valid_genes
        self.lambda_sym = lambda_sym
        self.roi_combination = roi_combination
        self.patience = 70 # usually 70
        
        # Setup symmetric loss if lambda_sym > 0
        if self.lambda_sym > 0:
            self.criterion = SymmetricLoss(nn.MSELoss(), lambda_sym=self.lambda_sym)
            print(f"Using SymmetricLoss with lambda_sym={self.lambda_sym}")
        
        # Self-attention encoder for independent ROI processing
        self.encoder = SelfAttentionGeneVecEncoder(
            valid_genes=self.valid_genes,
            expression_bins=expression_bins,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=transformer_dropout,
            use_alibi=use_alibi,
            pooling_mode=pooling_mode,
            genevec_type=genevec_type,
            roi_combination=roi_combination,
            device=self.device
        )
        self.encoder = torch.compile(self.encoder)
        
        # MLP prediction head - dimension depends on ROI combination method
        if self.pooling_mode == 'linear':
            if roi_combination == 'concat':
                prev_dim = len(self.encoder.shared_genes) * 2  # 2 ROIs worth of tokens
            else:
                prev_dim = len(self.encoder.shared_genes)  # Combined tokens
        else:
            if roi_combination == 'concat':
                prev_dim = 2 * d_model  # 2 ROIs worth of d_model
            else:
                prev_dim = d_model  # Combined d_model
                
        self.deep_layers, self.output_layer = self._set_mlp_head(
            prev_dim=prev_dim,
            deep_hidden_dims=deep_hidden_dims,
            dropout_rate=dropout_rate)
        
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in SelfAttentionGeneVec model: {num_params}")
        print(f"Expression bins: {expression_bins}")
        print(f"Pooling mode: {pooling_mode}")
        print(f"ROI combination: {roi_combination}")
        
        self._setup_optimizer_scheduler(learning_rate, weight_decay, use_cosine=cosine_lr)
    
    def forward(self, x, coords, idx):
        # Split input
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)  # Each: (B, num_genes)
        coords_i, coords_j = torch.chunk(coords, chunks=2, dim=1)  # Each: (B, 3)
        
        # Process through self-attention encoder
        combined_repr = self.encoder(x_i, x_j, coords_i, coords_j)  # (B, combined_dim)
        
        # Predict connectivity via MLP head
        y_pred = self.output_layer(self.deep_layers(combined_repr))
        
        return y_pred.squeeze()
    
    def fit(self, dataset, train_indices, test_indices, save_model=None, verbose=True, wandb_run=None):
        """Shared fit function for all transformer models"""
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, 
                                 pin_memory=True, num_workers=self.num_workers, 
                                 prefetch_factor=self.prefetch_factor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, 
                                pin_memory=True, num_workers=self.num_workers, 
                                prefetch_factor=self.prefetch_factor)
        
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, 
                            self.optimizer, self.patience, scheduler=self.scheduler, 
                            save_model=save_model, verbose=verbose, dataset=dataset, wandb_run=wandb_run)

    def get_attention_heads(self, avg_attn):
        """
        Simple helper to extract individual heads and global average from attention tensor
        
        Args:
            avg_attn: attention tensor from predict() method
            
        Returns:
            tuple: (individual_heads_list, global_average)
        """
        if avg_attn is None:
            return None, None
            
        # Convert to numpy if needed
        if hasattr(avg_attn, 'cpu'):
            avg_attn = avg_attn.cpu().numpy()
        
        # Individual heads as list
        individual_heads = [avg_attn[h] for h in range(avg_attn.shape[0])]
        
        # Global average
        global_average = avg_attn.mean(axis=0)
        
        return individual_heads, global_average