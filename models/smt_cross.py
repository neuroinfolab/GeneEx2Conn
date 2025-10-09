"""
Cross-Attention Gene2Vec Model for Region-Pair Connectivity Prediction

This module implements a single-gene resolution model that uses:
1. Gene2Vec embeddings (200D) for semantic gene representations
2. Expression value binning to discretize expression levels
3. Cross-attention mechanism where CLS + region i genes (queries) attend to region j genes (keys/values)
4. FlashAttention for efficient computation

Architecture:
-----------
For each region pair (i, j):
    1. Each gene's expression value is binned (default: 5 bins in [0,1])
    2. Bin index is one-hot encoded and projected to 200D
    3. Projected bin embedding + Gene2Vec embedding (element-wise addition)
    4. Combined embeddings projected to d_model space (default: 128D)
    5. CLS token created (random or coordinate-based initialization)
    6. CLS token concatenated with region i gene embeddings to form queries
    7. Region j gene embeddings serve as keys/values
    8. Cross-attention: queries (CLS + region i) attend to keys/values (region j)
    9. Cross-attention applied over multiple layers
    10. Pool transformer output (mean/attention/cls) to create final representation
    11. Pooled representation passed through MLP to predict connectivity
"""

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


class SymmetricLoss(nn.Module):
    """
    Loss function with symmetry regularization.
    
    Encourages model predictions for (i,j) to be close to predictions for (j,i).
    Leverages the fact that swapping region i and j features gives the symmetric pair.
    
    Args:
        base_criterion: Base loss function (e.g., nn.MSELoss())
        lambda_sym: Weight for symmetry regularization term (default: 0.1)
        
    Example usage:
        criterion = SymmetricLoss(nn.MSELoss(), lambda_sym=0.1)
        loss = criterion(predictions, targets, model, batch_X, batch_coords, batch_idx)
    """
    def __init__(self, base_criterion, lambda_sym=0.1):
        super().__init__()
        self.base_criterion = base_criterion
        self.lambda_sym = lambda_sym
        
    def forward(self, predictions, targets, model=None, batch_X=None, batch_coords=None, batch_idx=None):
        """
        Compute loss with optional symmetry regularization.
        
        Args:
            predictions: Model predictions for current batch
            targets: Ground truth targets
            model: Model instance (required for symmetry loss)
            batch_X: Input features (required for symmetry loss)
            batch_coords: Coordinate features (required for symmetry loss)  
            batch_idx: Batch indices (required for symmetry loss)
            
        Returns:
            Total loss (base loss + symmetry regularization)
        """
        # Base loss
        base_loss = self.base_criterion(predictions, targets)
        
        # If no model or lambda is 0, return base loss only
        if model is None or self.lambda_sym == 0 or batch_X is None:
            return base_loss
        
        # Leverage the fact that (i,j) and (j,i) differ only by swapped features
        # Split batch_X into two halves and swap them to get symmetric pairs
        num_genes = batch_X.shape[1] // 2
        X_i, X_j = batch_X[:, :num_genes], batch_X[:, num_genes:]
        symmetric_X = torch.cat([X_j, X_i], dim=1)
        
        # Same for coordinates if provided
        if batch_coords is not None:
            coords_i, coords_j = batch_coords[:, :3], batch_coords[:, 3:]
            symmetric_coords = torch.cat([coords_j, coords_i], dim=1)
        else:
            symmetric_coords = None
        
        # Get symmetric indices (flip even/odd: idx XOR 1)
        if batch_idx is not None:
            symmetric_idx = batch_idx ^ 1
        else:
            symmetric_idx = None
        
        # Forward pass for symmetric pairs (with gradients)
        try:
            if symmetric_coords is not None and symmetric_idx is not None:
                symmetric_predictions = model(symmetric_X, symmetric_coords, symmetric_idx).squeeze()
            else:
                symmetric_predictions = model(symmetric_X).squeeze()
        except:
            # Fallback for models without coords/idx
            symmetric_predictions = model(symmetric_X).squeeze()
        
        # Compute symmetry loss: penalize difference between pred(i,j) and pred(j,i)
        symmetry_loss = torch.mean((predictions - symmetric_predictions) ** 2)
        
        # Total loss
        total_loss = base_loss + self.lambda_sym * symmetry_loss
        
        return total_loss

class CrossAttentionBlock(nn.Module):
    """Cross-attention block using FlashAttention"""
    def __init__(self, d_model, nhead, dropout=0.1, use_alibi=False):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.d_model = d_model
        
        # Separate projections for queries (from region i) and keys/values (from region j)
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, d_model * 2)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
        self.use_alibi = use_alibi
        if use_alibi:
            slopes = FlashAttentionBlock.build_alibi_slopes(nhead)
            self.register_buffer("alibi_slopes", slopes)
    
    def split_heads(self, x):
        return x.view(x.size(0), x.size(1), self.nhead, self.head_dim).transpose(1, 2)
    
    def merge_heads(self, x):
        return x.transpose(1, 2).reshape(x.size(0), x.size(2), -1)
    
    def forward(self, queries, keys_values):
        """
        Args:
            queries: (B, seq_len_q, d_model) - query tokens (could be CLS or genes)
            keys_values: (B, seq_len_kv, d_model) - key/value tokens (both regions)
        """
        with autocast(dtype=torch.bfloat16):
            residual = queries
            
            # Project queries
            q = self.q_proj(queries)  # (B, seq_len_q, d_model)
            
            # Project keys and values
            kv = self.kv_proj(keys_values)  # (B, seq_len_kv, 2*d_model)
            k, v = kv.chunk(2, dim=-1)  # Each: (B, seq_len_kv, d_model)
            
            # Reshape for multi-head attention
            q = self.split_heads(q)  # (B, nhead, seq_len_q, head_dim)
            k = self.split_heads(k)  # (B, nhead, seq_len_kv, head_dim)
            v = self.split_heads(v)  # (B, nhead, seq_len_kv, head_dim)
            
            # Flash attention (cross-attention)
            # Note: flash_attn_func expects (B, seqlen, nhead, head_dim) format
            q = q.transpose(1, 2)  # (B, seq_len_q, nhead, head_dim)
            k = k.transpose(1, 2)  # (B, seq_len_kv, nhead, head_dim)
            v = v.transpose(1, 2)  # (B, seq_len_kv, nhead, head_dim)
            
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=0.0,
                causal=False,
                # alibi_slopes=self.alibi_slopes if self.use_alibi else None  # Uncomment if FlashAttn supports cross-attn alibi
            )
            
            # Reshape back
            attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), -1)
            
            # Apply dropout and residual
            attn_output = self.attn_dropout(attn_output)
            x = self.attn_norm(residual + attn_output)
            
            # Feed-forward with residual
            residual = x
            x = self.ffn(x)
            x = self.ffn_norm(residual + x)
            
            return x

class CrossAttentionGene2VecEncoder(nn.Module):
    """
    Cross-attention encoder with Gene2Vec embeddings and value binning.
    
    Each gene is treated as a token with:
    - Gene2Vec semantic embedding (200D)
    - Expression bin embedding (learned projection from one-hot bin)
    - Combined via element-wise addition
    
    Architecture:
    - CLS token + region i genes form the query sequence
    - Region j genes form the key/value sequence
    - After cross-attention, pool to create final edge representation
    
    Args:
        valid_genes: List of valid gene names
        expression_bins: Number of discrete expression bins (default: 5)
        d_model: Transformer hidden dimension (default: 128)
        nhead: Number of attention heads (default: 4)
        num_layers: Number of cross-attention layers (default: 4)
        dropout: Dropout rate (default: 0.1)
        use_alibi: Use ALiBi positional biases (default: False)
        cls_init: CLS token initialization - 'random' or 'spatial' (default: 'random')
        pooling_mode: Pooling method - 'mean', 'attention', or 'cls' (default: 'mean')
        device: Device to place model on
    """
    def __init__(self, valid_genes, expression_bins=5, d_model=128, nhead=4, num_layers=4, 
                 dropout=0.1, use_alibi=False, cls_init='random', pooling_mode='mean', device=None):
        super().__init__()
        self.d_model = d_model
        self.valid_genes = valid_genes
        self.expression_bins = expression_bins
        self.cls_init = cls_init
        self.pooling_mode = pooling_mode
        self.device = device
        self.gene2vec_dim = 200
        
        # Load Gene2Vec embeddings
        self.gene2vec_path = '/scratch/asr655/neuroinformatics/GeneEx2Conn/data/gene_emb/gene2vec_dim_200_iter_9.txt'
        self._load_gene2vec_embeddings()
        
        # Expression bin embedding: project one-hot bin to gene2vec_dim (200)
        self.bin_projection = nn.Linear(expression_bins, self.gene2vec_dim)
        
        # Project combined embedding (gene2vec + bin) to d_model
        self.to_d_model = nn.Linear(self.gene2vec_dim, d_model)
        
        # CLS token initialization
        if cls_init == 'random':
            # Random initialization with 6 values, then project to d_model
            self.cls_token_proj = nn.Linear(6, d_model)
            # Initialize with normal distribution
            self.cls_token_init = nn.Parameter(torch.randn(6))
            nn.init.normal_(self.cls_token_init, mean=0.0, std=0.02)
        elif cls_init == 'spatial':
            # Coordinate-based: 3 coords from each region (6 total)
            self.cls_token_proj = nn.Linear(6, d_model)
        else:
            raise ValueError(f"Unknown cls_init: {cls_init}")
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dropout, use_alibi=use_alibi)
            for _ in range(num_layers)
        ])
        
        # Pooling layer (if attention pooling is used)
        if self.pooling_mode == 'attention':
            self.attention_pooling = AttentionPooling(d_model, hidden_dim=32)
        elif self.pooling_mode == 'linear': # simple linear combination of all tokens to d_model 
            self.linear_proj = nn.Linear(len(self.shared_genes), d_model)
        
    def _load_gene2vec_embeddings(self):
        """Load Gene2Vec embeddings and create lookup table aligned with valid_genes"""
        # Load Gene2Vec dataframe
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
        
        # Store as buffer (not trainable)
        self.register_buffer('gene2vec_embeddings', torch.FloatTensor(gene2vec_matrix))
        
        print(f"Loaded Gene2Vec embeddings: {len(self.shared_genes)} genes with 200-dimensional embeddings")
        print(f"Gene overlap: {len(self.shared_genes)}/{len(self.valid_genes)} valid genes have Gene2Vec embeddings")
    
    def bin_expression(self, expression):
        """
        Bin expression values into discrete bins and return one-hot encoding
        
        Args:
            expression: (B, num_shared_genes) - expression values
        
        Returns:
            one_hot: (B, num_shared_genes, expression_bins) - one-hot encoded bins
        """
        # Assume expression is normalized to [0, 1]
        # Clamp to ensure within range
        expression = torch.clamp(expression, 0.0, 1.0)
        
        # Determine bin indices (0 to expression_bins-1)
        bin_indices = (expression * self.expression_bins).long()
        bin_indices = torch.clamp(bin_indices, 0, self.expression_bins - 1)
        
        # One-hot encode
        one_hot = F.one_hot(bin_indices, num_classes=self.expression_bins).float()
        
        return one_hot
    
    def encode_region(self, gene_expression):
        """
        Encode a single region's gene expression
        
        Args:
            gene_expression: (B, num_genes) - gene expression values
        
        Returns:
            embeddings: (B, num_shared_genes, d_model) - encoded gene tokens
        """
        B, num_genes = gene_expression.shape
        
        # Subset to shared genes
        gene_expression_subset = gene_expression[:, self.shared_gene_indices]  # (B, num_shared_genes)
        
        # Bin expression values
        binned_expr = self.bin_expression(gene_expression_subset)  # (B, num_shared_genes, expression_bins)
        
        # Project bins to gene2vec_dim
        bin_embeddings = self.bin_projection(binned_expr)  # (B, num_shared_genes, 200)
        
        # Get gene2vec embeddings and expand to batch
        gene2vec_emb = self.gene2vec_embeddings.unsqueeze(0).expand(B, -1, -1)  # (B, num_shared_genes, 200)
        
        # Element-wise addition
        combined_emb = bin_embeddings + gene2vec_emb  # (B, num_shared_genes, 200)
        
        # Project to d_model
        embeddings = self.to_d_model(combined_emb)  # (B, num_shared_genes, d_model)
        
        return embeddings
    
    def create_cls_token(self, coords_i, coords_j, batch_size):
        """
        Create CLS token for cross-attention
        
        Args:
            coords_i: (B, 3) - coordinates of region i
            coords_j: (B, 3) - coordinates of region j
            batch_size: int
        
        Returns:
            cls_token: (B, 1, d_model)
        """
        if self.cls_init == 'random':
            # Use random initialization, broadcast to batch
            cls_init = self.cls_token_init.unsqueeze(0).expand(batch_size, -1)  # (B, 6)
            cls_token = self.cls_token_proj(cls_init).unsqueeze(1)  # (B, 1, d_model)
        elif self.cls_init == 'spatial':
            # Concatenate coordinates
            coords_combined = torch.cat([coords_i, coords_j], dim=-1)  # (B, 6)
            # Scale coordinates to match random init range (normalize then scale to 0.02 std)
            coords_mean = coords_combined.mean(dim=0, keepdim=True)
            coords_std = coords_combined.std(dim=0, keepdim=True) + 1e-6  # Add epsilon for stability
            coords_normalized = (coords_combined - coords_mean) / coords_std
            coords_scaled = coords_normalized * 0.02
            cls_token = self.cls_token_proj(coords_scaled).unsqueeze(1)  # (B, 1, d_model)
        
        return cls_token
    
    def forward(self, gene_expr_i, gene_expr_j, coords_i, coords_j):
        """
        Forward pass with cross-attention
        
        Args:
            gene_expr_i: (B, num_genes) - region i expression
            gene_expr_j: (B, num_genes) - region j expression
            coords_i: (B, 3) - region i coordinates
            coords_j: (B, 3) - region j coordinates
        
        Returns:
            output: (B, d_model) - Pooled representation after cross-attention
        """
        B = gene_expr_i.size(0)
        
        # Encode both regions
        embeddings_i = self.encode_region(gene_expr_i)  # (B, num_shared_genes, d_model)
        embeddings_j = self.encode_region(gene_expr_j)  # (B, num_shared_genes, d_model)
        
        # Create CLS token
        cls_token = self.create_cls_token(coords_i, coords_j, B)  # (B, 1, d_model)
        
        # Concatenate CLS token with region i genes as queries
        #queries = torch.cat([cls_token, embeddings_i], dim=1)  # (B, num_shared_genes + 1, d_model)

        # temporarily omit CLS token entirely
        queries = torch.cat([embeddings_i], dim=1)  # (B, num_shared_genes + 1, d_model)
        
        # Region j genes as keys/values
        keys_values = embeddings_j  # (B, num_shared_genes, d_model)
        
        # Apply cross-attention layers - queries (CLS + region i) attend to keys/values (region j)
        for layer in self.cross_attn_layers:
            queries_out = layer(queries, keys_values)  # (B, num_shared_genes + 1, d_model)
        
        # Pool the output based on pooling_mode
        if self.pooling_mode == 'cls':
            # Use only CLS token (first position)
            output = queries_out[:, 0, :]  # (B, d_model)
        elif self.pooling_mode == 'mean':
            # Mean pool over all tokens (CLS + genes)
            output = queries_out.mean(dim=1)  # (B, d_model)
        elif self.pooling_mode == 'attention':
            # Attention pooling over all tokens
            output = self.attention_pooling(queries_out)  # (B, d_model)
        elif self.pooling_mode == 'linear':
            # Linear pooling over all tokens
            output = self.linear_proj(queries_out.transpose(1, 2))  # (B, d_model)
        else:
            raise ValueError(f"Unknown pooling_mode: {self.pooling_mode}")
        
        return output

class CrossAttentionGene2VecModel(BaseTransformerModel):
    """
    Cross-Attention Gene2Vec Model for predicting region-pair connectivity.
    
    Uses single-gene resolution with Gene2Vec embeddings, value binning, and 
    cross-attention where CLS + region i genes attend to region j genes.
    
    Args:
        input_dim: Total input dimension (2 * num_genes)
        region_pair_dataset: Dataset containing gene information
        expression_bins: Number of discrete expression bins (default: 5)
        d_model: Transformer embedding dimension (default: 128)
        nhead: Number of attention heads (default: 4)
        num_layers: Number of cross-attention layers (default: 4)
        deep_hidden_dims: MLP head hidden dimensions (default: [512, 256, 128])
        cls_init: CLS token initialization - 'random' or 'spatial' (default: 'random')
        pooling_mode: Pooling method - 'mean', 'attention', or 'cls' (default: 'mean')
        use_alibi: Use ALiBi positional biases (default: False)
        transformer_dropout: Dropout in attention blocks (default: 0.1)
        dropout_rate: Dropout in MLP head (default: 0.1)
        learning_rate: Learning rate (default: 0.001)
        weight_decay: Weight decay (default: 0.0)
        batch_size: Training batch size (default: 512)
        epochs: Number of training epochs (default: 100)
        aug_prob: Data augmentation probability (default: 0.0)
        num_workers: DataLoader workers (default: 2)
        prefetch_factor: DataLoader prefetch factor (default: 2)
        cosine_lr: Use cosine learning rate schedule (default: False)
        lambda_sym: Weight for symmetry regularization (default: 0.1)
    """
    def __init__(self, input_dim, region_pair_dataset, 
                 expression_bins=5, d_model=128, nhead=4, num_layers=4, 
                 deep_hidden_dims=[512, 256, 128], cls_init='random', pooling_mode='mean',
                 use_alibi=False, transformer_dropout=0.1, dropout_rate=0.1, 
                 learning_rate=0.001, weight_decay=0.0, batch_size=512, epochs=100, 
                 aug_prob=0.0, num_workers=2, prefetch_factor=2, cosine_lr=False, aug_style='linear_decay',
                 lambda_sym=0.1):
        
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, 
                        num_workers, prefetch_factor, d_model, nhead, num_layers, 
                        deep_hidden_dims=deep_hidden_dims, aug_prob=aug_prob, aug_style='linear_decay')
        
        self.input_dim = input_dim // 2  # Split for two regions
        self.expression_bins = expression_bins
        self.cls_init = cls_init
        self.pooling_mode = pooling_mode
        self.use_alibi = use_alibi
        self.cosine_lr = cosine_lr
        self.valid_genes = region_pair_dataset.valid_genes
        self.lambda_sym = lambda_sym
        # self.include_coords = True

        # Setup symmetric loss if lambda_sym > 0
        if self.lambda_sym > 0:
            self.criterion = SymmetricLoss(nn.MSELoss(), lambda_sym=self.lambda_sym)
            print(f"Using SymmetricLoss with lambda_sym={self.lambda_sym}")
        
        # Cross-attention encoder
        self.encoder = CrossAttentionGene2VecEncoder(
            valid_genes=self.valid_genes,
            expression_bins=expression_bins,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=transformer_dropout,
            use_alibi=use_alibi,
            cls_init=cls_init,
            pooling_mode=pooling_mode,
            device=self.device
        )
        self.encoder = torch.compile(self.encoder)
        
        # MLP prediction head
        prev_dim = d_model
        self.deep_layers, self.output_layer = self._set_mlp_head(
            prev_dim=prev_dim,
            deep_hidden_dims=deep_hidden_dims,
            dropout_rate=dropout_rate
        )
        
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in CrossAttentionGene2Vec model: {num_params}")
        print(f"Number of genes after Gene2Vec intersection: {len(self.encoder.shared_gene_indices)}")
        print(f"Expression bins: {expression_bins}")
        print(f"CLS token initialization: {cls_init}")
        print(f"Pooling mode: {pooling_mode}")
        
        self._setup_optimizer_scheduler(learning_rate, weight_decay, use_cosine=cosine_lr)
    
    def forward(self, x, coords, idx):
        # Split input
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)  # Each: (B, num_genes)
        coords_i, coords_j = torch.chunk(coords, chunks=2, dim=1)  # Each: (B, 3)
        
        # Encode with cross-attention
        pooled_output = self.encoder(x_i, x_j, coords_i, coords_j)  # (B, d_model)
        
        # Predict connectivity
        y_pred = self.output_layer(self.deep_layers(pooled_output))
        
        return y_pred.squeeze()
    
    def fit(self, dataset, train_indices, test_indices, save_model=None, verbose=True):
        """Shared fit function for all transformer models"""
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, 
                                 pin_memory=True, num_workers=self.num_workers, 
                                 prefetch_factor=self.prefetch_factor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, 
                                pin_memory=True, num_workers=self.num_workers, 
                                prefetch_factor=self.prefetch_factor)
        self.patience = 200
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, 
                          self.optimizer, self.patience, scheduler=None, 
                          train_scheduler=self.scheduler if self.cosine_lr else None, 
                          save_model=save_model, verbose=verbose, dataset=dataset)


'''
class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, d_model * 2)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2* d_model, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def split_heads(self, x):
        return x.view(x.size(0), x.size(1), self.nhead, self.head_dim).transpose(1, 2)

    def merge_heads(self, x):
        return x.transpose(1, 2).reshape(x.size(0), x.size(2), -1)

    def forward(self, q_input, kv_input):
        with autocast(dtype=torch.bfloat16):
            residual = q_input
            q = self.q_proj(q_input)
            kv = self.kv_proj(kv_input)
            k, v = kv.chunk(2, dim=-1)

            q = self.split_heads(q).transpose(1, 2)
            k = self.split_heads(k).transpose(1, 2)
            v = self.split_heads(v).transpose(1, 2)
            attn_output = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
            attn_output = attn_output.transpose(1, 2)        
            attn_output = self.merge_heads(attn_output)
            attn_output = self.attn_dropout(attn_output)

            x = self.attn_norm(residual + attn_output)
            residual = x
            x = self.ffn(x)
            x = self.ffn_norm(residual + x)
            return x
'''
class CrossAttentionEncoder(nn.Module):
    def __init__(self, token_encoder_dim, d_model, output_dim, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(token_encoder_dim, d_model)
        self.transformer_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(self, x_i, x_j):
        batch_size = x_i.shape[0]
        L_i = x_i.shape[1] // self.input_projection.in_features
        L_j = x_j.shape[1] // self.input_projection.in_features

        x_i = x_i.view(batch_size, L_i, -1)
        x_j = x_j.view(batch_size, L_j, -1)

        q = self.input_projection(x_i)
        kv = self.input_projection(x_j)

        for layer in self.transformer_layers:
            q = layer(q, kv)

        out = self.output_projection(q)
        out = out.flatten(start_dim=1)
        return out

class CrossAttentionModel(BaseTransformerModel):
    def __init__(self, input_dim, token_encoder_dim=20, d_model=128, encoder_output_dim=10, nhead=2, num_layers=2, deep_hidden_dims=[256, 128],
                 transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0,
                 batch_size=128, epochs=100):
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs)

        self.input_dim = input_dim // 2
        self.token_encoder_dim = token_encoder_dim
        self.d_model = d_model
        self.encoder_output_dim = encoder_output_dim

        self.encoder = CrossAttentionEncoder(
            token_encoder_dim=self.token_encoder_dim,
            d_model=self.d_model,
            output_dim=self.encoder_output_dim,
            nhead=nhead,
            num_layers=num_layers,
            dropout=transformer_dropout
        )
        self.encoder = torch.compile(self.encoder)

        prev_dim = (self.input_dim // self.token_encoder_dim * self.encoder_output_dim)
        self.deep_layers, self.output_layer = self._set_mlp_head(
            prev_dim=prev_dim,
            deep_hidden_dims=deep_hidden_dims,
            dropout_rate=dropout_rate
        )

        self._setup_optimizer_scheduler(learning_rate, weight_decay)

    def forward(self, x):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        encoded = self.encoder(x_i, x_j)
        return self.deep_layers(encoded).squeeze()
