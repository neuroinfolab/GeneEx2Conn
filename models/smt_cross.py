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
    Cross-Attention block using FlashAttention.
    
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
        
        self.use_alibi = use_alibi
        if use_alibi:
            slopes = FlashAttentionBlock.build_alibi_slopes(nhead)
            self.register_buffer("alibi_slopes", slopes)
    
    def split_heads(self, x):
        """Split heads for queries, keys and values"""
        return x.view(x.size(0), x.size(1), self.nhead, self.head_dim)
    
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
            #residual = (queries + keys) / 2 # residual can be mean of both regions
            residual = queries

            # Project queries, keys, and values
            q = self.q_proj(queries)  # (B, seq_len_q, d_model)
            k = self.k_proj(keys)  # (B, seq_len_kv, d_model)
            v = self.v_proj(values)  # (B, seq_len_kv, d_model*2)
            
            # Reshape for multi-head attention
            # flash_attn_func expects (B, seqlen, nhead, head_dim) format
            q = self.split_heads(q)  # (B, seq_len_q, nhead, head_dim)
            k = self.split_heads(k)  # (B, seq_len_kv, nhead, head_dim)
            v = self.split_heads(v)  # (B, seq_len_kv, nhead, head_dim*2)
            
            # Flash attention (cross-attention)
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout,
                causal=False,
            )  # (B, seq_len_q, nhead, head_dim)
            
            # Reshape back: (B, seq_len_q, d_model*2)
            attn_output = attn_output.reshape(attn_output.size(0), attn_output.size(1), self.d_model)        
            # Apply dropout and residual
            attn_output = self.attn_dropout(attn_output)
            
            x = self.attn_norm(residual + attn_output)
            
            # Feed-forward with residual
            residual = x
            x = self.ffn(x)
            x = self.ffn_norm(residual + x)
            
            return x

class CrossAttentionGeneVecEncoder(nn.Module):
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
        cls_init: CLS token initialization - 'random' or 'spatial' (default: 'random')
        pooling_mode: Pooling method - 'mean', 'attention', or 'cls' (default: 'mean')
        device: Device to place model on
    """
    def __init__(self, valid_genes, expression_bins=5, d_model=32, nhead=2, num_layers=2, 
                 dropout=0.1, use_alibi=False, cls_init='random', pooling_mode='mean', genevec_type='gene2vec', device=None):
        super().__init__()
        self.d_model = d_model
        self.valid_genes = valid_genes
        self.expression_bins = expression_bins
        self.cls_init = cls_init
        self.pooling_mode = pooling_mode
        self.genevec_type = genevec_type
        self.device = device
        
        self._load_genevec_embeddings(genevec_type)
        
        # project genevec to d_model
        self.genevec_projection = nn.Linear(self.genevec_dim, d_model)
        
        # project one-hot bin to d_model
        self.bin_embedding = nn.Embedding(expression_bins, d_model)

        # project scalar expression to d_model
        self.scalar_projection = nn.Linear(1, d_model)

        # project combined genevec and bin to d_model
        self.combined_projection = nn.Linear(2*d_model, d_model)
            
        if cls_init == 'random':
            # Random initialization with 6 values, then project to d_model
            self.cls_token_proj = nn.Linear(6, d_model)
            self.cls_token_init = nn.Parameter(torch.randn(6))
            nn.init.normal_(self.cls_token_init, mean=0.0, std=0.02)
        elif cls_init == 'spatial':
            self.cls_token_proj = nn.Linear(6, d_model)        
    
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dropout, use_alibi=use_alibi)
            for _ in range(num_layers)
        ])
        
        if self.pooling_mode == 'attention':
            self.attention_pooling = AttentionPooling(d_model, hidden_dim=32)
        elif self.pooling_mode == 'linear':
            self.linear_pooling = nn.Linear(d_model, 1)
        
    def _load_genevec_embeddings(self, genevec_type='gene2vec'):
        """Load gene vector embeddings and create lookup table aligned with valid_genes
        Args:
            type: 'gene2vec' or 'coexpression'
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
            # indices of valid_genes that can be kept
            self.shared_gene_indices = [self.valid_genes.index(gene) for gene in shared_genes if gene in self.valid_genes]
            # list of shared genes to use to create the lookup matrix
            self.shared_genes = [self.valid_genes[i] for i in self.shared_gene_indices]
            
            # Create Gene2Vec lookup matrix in the order of shared genes
            gene2vec_matrix = np.zeros((len(self.shared_genes), 200))
            for i, gene in enumerate(self.shared_genes):
                if gene in gene2vec_df.index: # use the gene name from shared genes to create the lookup matrix
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
            # can toggle between cov and corr in the path
            self.coexpression_path = '/scratch/asr655/neuroinformatics/GeneEx2Conn/data/gene_emb/genevec_cov.txt'
            coexpression_df = pd.read_csv(self.coexpression_path, sep=' ', skiprows=1, index_col=0)
            
            # Use all valid genes since coexpression matrix contains all genes
            self.shared_genes = self.valid_genes
            self.shared_gene_indices = list(range(len(self.valid_genes)))
            
            # Create coexpression lookup matrix in order of valid genes
            # Each gene has correlation vector of length 7380 (from 0.2 gene list)
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
        Encode a single region's gene expression
        """
        B, num_genes = gene_expression.shape
        
        # Subset to genes with genevec embeddings
        gene_expression_subset = gene_expression[:, self.shared_gene_indices]  # (B, num_shared_genes)
        
        # Get genevec embeddings, expand to batch, and project to d_model
        genevec_emb = self.genevec_embeddings.unsqueeze(0).expand(B, -1, -1)
        genevec_emb = self.genevec_projection(genevec_emb)  # (B, num_shared_genes, d_model)
        
        # Bin expression values
        # bin_indices = (gene_expression_subset * self.expression_bins).long()
        # bin_indices = torch.clamp(bin_indices, 0, self.expression_bins - 1)
        # bin_emb = self.bin_embedding(bin_indices)
        
        # Project scalar expression to d_model
        scalar_emb = self.scalar_projection(gene_expression_subset.unsqueeze(-1))  # (B, num_shared_genes, d_model)
        
        # Element-wise addition
        #embeddings = genevec_emb + scalar_emb # quick ablation: genevec_emb + bin_emb  # (B, num_shared_genes, d_model)
        
        # Concatenate embeddings
        embeddings = torch.cat([genevec_emb, scalar_emb], dim=-1)  # (B, num_shared_genes, 2*d_model)
        embeddings = self.combined_projection(embeddings)  # Project back to d_model dimension

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
            cls_init = self.cls_token_init.unsqueeze(0).expand(batch_size, -1)  # (B, 6)
            cls_token = self.cls_token_proj(cls_init).unsqueeze(1)  # (B, 1, d_model)
        elif self.cls_init == 'spatial':
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
        
        # Optional: Create CLS token (only used if pooling_mode='cls')
        if self.pooling_mode == 'cls':
            cls_token = self.create_cls_token(coords_i, coords_j, B)  # (B, 1, d_model)
            # Concatenate CLS with embeddings
            embeddings_i = torch.cat([cls_token, embeddings_i], dim=1)  # (B, num_shared_genes + 1, d_model)
            embeddings_j = torch.cat([cls_token, embeddings_j], dim=1)  # (B, num_shared_genes + 1, d_model)
        
        for layer in self.cross_attn_layers:
            # Q, K, V
            attn_output = layer(embeddings_i, embeddings_j, embeddings_j)  # (B, seq_len, d_model)
        
        # Pool the output
        if self.pooling_mode == 'cls': # Use only CLS token (first position
            # this technically only captures information about how CLS from region i attends to region j
            pooled = attn_output[:, 0, :]  # (B, d_model)
        elif self.pooling_mode == 'mean': # Mean pool over all attn_output tokens
            pooled = attn_output.mean(dim=1)  # (B, d_model)
        elif self.pooling_mode == 'attention': # Attention pooling over all attn_output tokens
            pooled = self.attention_pooling(attn_output)  # (B, d_model)
        elif self.pooling_mode == 'linear': # Mean pool over all attn_output tokens
            # Reshape from (B, num_tokens, d_model) to (B, d_model, num_tokens) for linear projection
            pooled = self.linear_pooling(attn_output).squeeze(-1)  # (B, num_tokens)
        
        return pooled

class CrossAttentionGeneVecModel(BaseTransformerModel):
    """
    Cross-Attention GeneVec Model for symmetric edge prediction.
    """
    def __init__(self, input_dim, region_pair_dataset, 
                 expression_bins=5, d_model=128, nhead=4, num_layers=4, 
                 deep_hidden_dims=[512, 256, 128], cls_init='random', pooling_mode='mean',
                 use_alibi=False, transformer_dropout=0.1, dropout_rate=0.1, 
                 learning_rate=0.001, weight_decay=0.0, batch_size=512, epochs=100, 
                 aug_prob=0.0, num_workers=2, prefetch_factor=2, cosine_lr=False, aug_style='linear_decay',
                 lambda_sym=0.1, genevec_type='gene2vec'):
        
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
        self.patience = 70

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
            cls_init=cls_init,
            pooling_mode=pooling_mode,
            genevec_type=genevec_type,
            device=self.device
        )
        self.encoder = torch.compile(self.encoder)
        
        # MLP prediction head
        if self.pooling_mode == 'linear':
            prev_dim = len(self.encoder.shared_genes)
        else:
            prev_dim = d_model
        self.deep_layers, self.output_layer = self._set_mlp_head(
            prev_dim=prev_dim,
            deep_hidden_dims=deep_hidden_dims,
            dropout_rate=dropout_rate)
        
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in CrossAttentionGeneVec model: {num_params}")
        print(f"Expression bins: {expression_bins}")
        print(f"CLS token initialization: {cls_init}")
        
        self._setup_optimizer_scheduler(learning_rate, weight_decay, use_cosine=cosine_lr)
    
    def forward(self, x, coords, idx):
        # Split input
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)  # Each: (B, num_genes)
        coords_i, coords_j = torch.chunk(coords, chunks=2, dim=1)  # Each: (B, 3)
        
        # Encode with cross-attention
        pooled_output = self.encoder(x_i, x_j, coords_i, coords_j)  # (B, d_model)
        
        # Predict connectivity via MLP head
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
'''