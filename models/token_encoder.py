import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TokenEncoder(nn.Module):
    """Base class for token encoders."""
    def __init__(self, token_dim, d_model):
        super().__init__()
        self.token_dim = token_dim
        self.d_model = d_model

    def forward(self, x):
        # x has shape (batch_size, num_tokens, token_dim)
        raise NotImplementedError

class LinearTokenEncoder(TokenEncoder):
    """Encodes tokens using a single linear layer."""
    def __init__(self, token_dim, d_model):
        super().__init__(token_dim, d_model)
        self.projection = nn.Linear(token_dim, d_model)

    def forward(self, x):
        return self.projection(x)

class MLPTokenEncoder(TokenEncoder):
    """Encodes tokens using a small MLP."""
    def __init__(self, token_dim, d_model, hidden_dim_multiplier=2):
        super().__init__(token_dim, d_model)
        hidden_dim = d_model * hidden_dim_multiplier
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x):
        return self.mlp(x)

class Conv1DTokenEncoder(TokenEncoder):
    """
    Encodes tokens using a 1D convolution within each token.
    This captures local relationships among genes within a token.
    """
    def __init__(self, token_dim, d_model, kernel_size=5):
        super().__init__(token_dim, d_model)
        # Each token is a sequence of `token_dim` genes.
        # We'll treat channels as 1 and sequence length as `token_dim`.
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=d_model, # We'll treat the output channels as the feature dim
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        # After convolution, the length will be `token_dim`. We need to pool it to a single vector.
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x has shape (batch_size, num_tokens, token_dim)
        # Reshape for Conv1D: (batch_size * num_tokens, 1, token_dim)
        batch_size, num_tokens, token_dim = x.shape
        x_reshaped = x.contiguous().view(batch_size * num_tokens, 1, token_dim)
        
        x_conv = self.conv(x_reshaped) # -> (B * L, d_model, token_dim)
        x_pooled = self.pool(x_conv) # -> (B * L, d_model, 1)
        x_squeezed = x_pooled.squeeze(-1) # -> (B * L, d_model)
        
        # Reshape back to (batch_size, num_tokens, d_model)
        output = x_squeezed.contiguous().view(batch_size, num_tokens, self.d_model)
        return output

class SetTransformerTokenEncoder(TokenEncoder):
    """
    Uses Set Transformer approach - treats genes within a token as a set.
    More appropriate for gene expression where order doesn't matter.
    """
    def __init__(self, token_dim, d_model, num_heads=4, num_layers=2):
        super().__init__(token_dim, d_model)
        self.gene_embedding = nn.Linear(1, d_model)  # Each gene expression -> d_model
        
        # Multi-head attention layers
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Global pooling to get fixed-size representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x shape: (batch_size, num_tokens, token_dim)
        batch_size, num_tokens, token_dim = x.shape
        
        # Process tokens one by one to reduce memory usage
        outputs = []
        for i in range(num_tokens):
            # Get single token: (batch_size, token_dim)
            token = x[:, i, :]
            
            # Reshape for gene embedding: (batch_size, token_dim, 1)
            token_reshaped = token.unsqueeze(-1)
            
            # Embed each gene expression value
            token_embedded = self.gene_embedding(token_reshaped)  # -> (batch_size, token_dim, d_model)
            
            # Apply self-attention layers (treating genes as set elements)
            for layer, layer_norm in zip(self.layers, self.layer_norms):
                attn_out, _ = layer(token_embedded, token_embedded, token_embedded)
                token_embedded = layer_norm(token_embedded + attn_out)
            
            # Global average pooling across genes
            token_pooled = token_embedded.mean(dim=1)  # -> (batch_size, d_model)
            outputs.append(token_pooled)
        
        # Stack all token outputs
        output = torch.stack(outputs, dim=1)  # -> (batch_size, num_tokens, d_model)
        return output

class GeneInteractionTokenEncoder(TokenEncoder):
    """
    Captures gene-gene interactions within each token using pairwise attention.
    Biologically motivated for gene co-expression patterns.
    """
    def __init__(self, token_dim, d_model):
        super().__init__(token_dim, d_model)
        self.gene_proj = nn.Linear(1, d_model // 2)
        self.interaction_layer = nn.Linear(token_dim * (token_dim - 1) // 2, d_model // 2)
        self.final_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x shape: (batch_size, num_tokens, token_dim)
        batch_size, num_tokens, token_dim = x.shape
        
        # Direct gene representations
        x_reshaped = x.contiguous().view(batch_size * num_tokens, token_dim, 1)
        gene_features = self.gene_proj(x_reshaped).mean(dim=1)  # -> (B * num_tokens, d_model//2)
        
        # Pairwise gene interactions
        x_flat = x.contiguous().view(batch_size * num_tokens, token_dim)
        interactions = []
        for i in range(token_dim):
            for j in range(i + 1, token_dim):
                interactions.append((x_flat[:, i] * x_flat[:, j]).unsqueeze(1))
        
        if interactions:
            interaction_features = torch.cat(interactions, dim=1)  # -> (B * num_tokens, token_dim*(token_dim-1)/2)
            interaction_features = self.interaction_layer(interaction_features)  # -> (B * num_tokens, d_model//2)
        else:
            interaction_features = torch.zeros(batch_size * num_tokens, self.d_model // 2, device=x.device)
        
        # Combine gene and interaction features
        combined = torch.cat([gene_features, interaction_features], dim=1)  # -> (B * num_tokens, d_model)
        output = self.final_proj(combined)
        
        # Reshape back to token format
        return output.view(batch_size, num_tokens, self.d_model)

class StatisticalTokenEncoder(TokenEncoder):
    """
    Computes statistical features of gene expression within each token.
    Captures distribution properties relevant for connectivity.
    """
    def __init__(self, token_dim, d_model):
        super().__init__(token_dim, d_model)
        # Statistical features: mean, std, min, max, skew, kurtosis approximation
        self.stat_proj = nn.Linear(6, d_model // 2)
        self.gene_proj = nn.Linear(token_dim, d_model // 2)
        self.final_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x shape: (batch_size, num_tokens, token_dim)
        batch_size, num_tokens, token_dim = x.shape
        
        # Process in smaller chunks to avoid memory issues
        outputs = []
        chunk_size = min(32, num_tokens)  # Process 32 tokens at a time
        
        for i in range(0, num_tokens, chunk_size):
            end_idx = min(i + chunk_size, num_tokens)
            x_chunk = x[:, i:end_idx, :]  # (batch_size, chunk_tokens, token_dim)
            chunk_tokens = end_idx - i
            
            x_flat = x_chunk.contiguous().view(batch_size * chunk_tokens, token_dim)
            
            # Compute statistical features with numerical stability
            mean_vals = x_flat.mean(dim=1, keepdim=True)
            std_vals = x_flat.std(dim=1, keepdim=True, unbiased=False) + 1e-8
            min_vals, _ = x_flat.min(dim=1, keepdim=True)
            max_vals, _ = x_flat.max(dim=1, keepdim=True)
            
            # Simple skewness and kurtosis approximations with better numerical stability
            centered = x_flat - mean_vals
            skew_approx = (centered ** 3).mean(dim=1, keepdim=True) / (std_vals ** 3)
            kurt_approx = (centered ** 4).mean(dim=1, keepdim=True) / (std_vals ** 4)
            
            # Clamp extreme values to prevent numerical issues
            skew_approx = torch.clamp(skew_approx, -10, 10)
            kurt_approx = torch.clamp(kurt_approx, -10, 10)
            
            stats = torch.cat([mean_vals, std_vals, min_vals, max_vals, skew_approx, kurt_approx], dim=1)
            stat_features = self.stat_proj(stats)  # -> (B * chunk_tokens, d_model//2)
            
            # Also include direct gene representations
            gene_features = self.gene_proj(x_flat)  # -> (B * chunk_tokens, d_model//2)
            
            # Combine features
            combined = torch.cat([stat_features, gene_features], dim=1)
            chunk_output = self.final_proj(combined)
            
            # Reshape chunk back to token format
            chunk_output = chunk_output.view(batch_size, chunk_tokens, self.d_model)
            outputs.append(chunk_output)
        
        # Concatenate all chunks
        output = torch.cat(outputs, dim=1)
        return output

class HierarchicalTokenEncoder(TokenEncoder):
    """
    Uses hierarchical grouping - first local groups, then global aggregation.
    Better for larger token dimensions.
    """
    def __init__(self, token_dim, d_model, group_size=4):
        super().__init__(token_dim, d_model)
        self.group_size = group_size
        self.num_groups = (token_dim + group_size - 1) // group_size  # Ceiling division
        
        # Local processing within each group
        self.local_encoder = nn.Sequential(
            nn.Linear(group_size, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 2)
        )
        
        # Global processing across groups
        self.global_encoder = nn.Sequential(
            nn.Linear(self.num_groups * (d_model // 2), d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Fallback encoder for edge cases
        self.fallback_encoder = nn.Sequential(
            nn.Linear(token_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x):
        # x shape: (batch_size, num_tokens, token_dim)
        batch_size, num_tokens, token_dim = x.shape
        
        # Process in chunks to reduce memory usage
        outputs = []
        chunk_size = min(16, num_tokens)  # Smaller chunks for hierarchical processing
        
        for i in range(0, num_tokens, chunk_size):
            end_idx = min(i + chunk_size, num_tokens)
            x_chunk = x[:, i:end_idx, :]  # (batch_size, chunk_tokens, token_dim)
            chunk_tokens = end_idx - i
            
            x_flat = x_chunk.contiguous().view(batch_size * chunk_tokens, token_dim)
            
            # Process local groups
            local_features = []
            for j in range(self.num_groups):
                start_idx = j * self.group_size
                end_idx_group = min(start_idx + self.group_size, token_dim)
                
                if start_idx < token_dim:
                    # Handle partial groups at the end
                    if end_idx_group > token_dim:
                        # Pad the group if it's smaller than group_size
                        group_data = x_flat[:, start_idx:token_dim]
                        padding_size = self.group_size - (token_dim - start_idx)
                        padding = torch.zeros(group_data.shape[0], padding_size, 
                                            device=group_data.device, dtype=group_data.dtype)
                        group_data = torch.cat([group_data, padding], dim=1)
                    else:
                        group_data = x_flat[:, start_idx:end_idx_group]
                    
                    group_features = self.local_encoder(group_data)
                    local_features.append(group_features)
            
            if local_features:
                local_concat = torch.cat(local_features, dim=1)
                chunk_output = self.global_encoder(local_concat)
            else:
                # Fallback: use dedicated fallback encoder
                chunk_output = self.fallback_encoder(x_flat)
            
            # Reshape chunk back to token format
            chunk_output = chunk_output.view(batch_size, chunk_tokens, self.d_model)
            outputs.append(chunk_output)
        
        # Concatenate all chunks
        output = torch.cat(outputs, dim=1)
        return output

class TokenEncoderFactory:
    """Factory to create token encoders."""
    @staticmethod
    def create(encoder_type, **kwargs):
        if encoder_type == 'linear':
            return LinearTokenEncoder(kwargs['token_dim'], kwargs['d_model'])
        elif encoder_type == 'mlp':
            return MLPTokenEncoder(kwargs['token_dim'], kwargs['d_model'])
        elif encoder_type == 'conv1d':
            return Conv1DTokenEncoder(kwargs['token_dim'], kwargs['d_model'])
        elif encoder_type == 'set_transformer':
            return SetTransformerTokenEncoder(kwargs['token_dim'], kwargs['d_model'])
        elif encoder_type == 'gene_interaction':
            return GeneInteractionTokenEncoder(kwargs['token_dim'], kwargs['d_model'])
        elif encoder_type == 'statistical':
            return StatisticalTokenEncoder(kwargs['token_dim'], kwargs['d_model'])
        elif encoder_type == 'hierarchical':
            return HierarchicalTokenEncoder(kwargs['token_dim'], kwargs['d_model'])
        elif encoder_type == 'scbert':
            return scBERTTokenEncoder(kwargs['token_dim'], kwargs['d_model'])
        elif encoder_type == 'geneformer':
            return GeneformerTokenEncoder(kwargs['token_dim'], kwargs['d_model'])
        else:
            raise ValueError(f"Unknown token encoder type: {encoder_type}") 

class scBERTTokenEncoder(TokenEncoder):
    """
    Uses a pre-trained scBERT model to encode gene expression tokens.
    This allows for fine-tuning of scBERT for the connectome prediction task.
    """
    def __init__(self, token_dim, d_model, scbert_model_path='/scratch/sg8603/scBERT/', scbert_params=None):
        super().__init__(token_dim, d_model)

        # Add scBERT module path to sys.path to allow for dynamic import
        import sys
        if scbert_model_path not in sys.path:
            sys.path.append(scbert_model_path)

        from performer_pytorch.performer_pytorch import PerformerLM
    
        # Simple vocabulary class for gene expression binning
        class SimpleVocab:
            def __init__(self):
                self.pad_token_id = 0
                self.cls_token_id = 1
                self.sep_token_id = 2
                self.mask_token_id = 3
                self.unk_token_id = 4
                # Gene expression bins start after special tokens
                self.gene_start_id = 5

        # Original scBERT parameters to fully utilize pretrained weights
        if scbert_params is None:
            scbert_params = {
                'num_tokens': 7,  # Number of bins for expression values
                'dim': 200,      # Original pretrained dimension
                'depth': 6,      # Original pretrained depth
                'heads': 10,     # Original pretrained heads
                'max_seq_len': min(token_dim, 2048)  # Cap sequence length for memory
            }

        # Load vocabulary
        self.vocab = SimpleVocab()

        # Instantiate the scBERT model with memory optimizations
        self.scbert = PerformerLM(
            num_tokens=scbert_params['num_tokens'],
            max_seq_len=scbert_params['max_seq_len'],
            dim=scbert_params['dim'],
            depth=scbert_params['depth'],
            heads=scbert_params['heads'],
            causal=False,  # Not a causal model for this task
            local_attn_heads=0,
            g2v_position_emb=False, # Set to False as we are not using gene2vec
            reversible=True,  # Use reversible layers to save memory
            ff_chunks=8  # Process feed-forward in chunks
        )

        # Load pre-trained weights
        try:
            state_dict = torch.load(f"{scbert_model_path}/panglao_pretrain.pth", map_location='cpu')
            # Filter out unnecessary keys from the state_dict
            model_keys = self.scbert.state_dict().keys()
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
            self.scbert.load_state_dict(filtered_state_dict, strict=False)
        except FileNotFoundError:
            print("WARNING: scBERT pre-trained model not found. Initializing with random weights.")
        except Exception as e:
            print(f"An error occurred while loading scBERT weights: {e}")

        # Projection layer to match the main model's dimension (d_model)
        self.projection = nn.Linear(scbert_params['dim'], d_model)
        self.d_model = d_model

    def forward(self, x):
        # x shape: (batch_size, num_tokens, token_dim)
        batch_size, num_tokens, token_dim = x.shape
        x_flat = x.contiguous().view(batch_size * num_tokens, token_dim)

        # --- Preprocessing for scBERT ---
        # 1. Binning: Convert continuous expression values to discrete integer tokens.
        # This is a simplified binning strategy. scBERT's original preprocessing might be more complex.
        # We'll use 5 bins (excluding padding and unknown tokens).
        # Bins: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%
        percentiles = torch.tensor([0.2, 0.4, 0.6, 0.8], device=x.device)
        bin_edges = torch.quantile(x_flat[x_flat > 0], percentiles, dim=0, keepdim=True).T
        
        # Handle cases where all values are zero
        if bin_edges.shape[0] != x_flat.shape[1]:
            bin_edges = torch.zeros(x_flat.shape[1], 4, device=x.device)

        # Add a small value to the last edge to include max value
        bin_edges_plus = torch.cat([bin_edges, torch.full((x_flat.shape[1], 1), float('inf'), device=x.device)], dim=1)

        # Convert to tokens
        x_binned = torch.sum(x_flat.unsqueeze(-1) > bin_edges_plus, dim=-1)
        x_binned = x_binned + self.vocab.cls_token_id # Offset by special tokens

        # --- Pass through scBERT with chunked processing ---
        # Process in smaller chunks to reduce memory usage
        chunk_size = 512  # Process 512 samples at a time
        total_samples = x_binned.shape[0]
        
        embeddings_list = []
        for i in range(0, total_samples, chunk_size):
            end_idx = min(i + chunk_size, total_samples)
            chunk = x_binned[i:end_idx]
            
            # Process chunk through scBERT
            chunk_embeddings = self.scbert(chunk, return_encodings=True)
            
            # Mean pooling over sequence for this chunk
            chunk_pooled = chunk_embeddings.mean(dim=1)
            embeddings_list.append(chunk_pooled)
            
            # Clear cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all chunk results
        pooled_embedding = torch.cat(embeddings_list, dim=0)

        # Project to the model's expected dimension
        projected_embedding = self.projection(pooled_embedding)

        # Reshape back to (batch_size, num_tokens, d_model)
        return projected_embedding.view(batch_size, num_tokens, self.d_model)


class GeneformerTokenEncoder(TokenEncoder):
    """
    Token encoder using pretrained Geneformer (Theodoris et al. 2023).
    Geneformer is a BERT-based transformer pretrained on ~30M single-cell transcriptomes.
    """
    
    def __init__(self, token_dim, d_model, model_name='ctheodoris/Geneformer'):
        super().__init__(token_dim, d_model)
        
        try:
            from transformers import BertModel, BertTokenizer
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("transformers library is required for GeneformerTokenEncoder. Install with: pip install transformers")
        
        # Load pretrained Geneformer model and tokenizer
        self.model_name = model_name
        self.geneformer = BertModel.from_pretrained(model_name, trust_remote_code=True)
        
        # Freeze the pretrained model to use as feature extractor
        for param in self.geneformer.parameters():
            param.requires_grad = False
        
        # Get the hidden size from the model config
        self.geneformer_dim = self.geneformer.config.hidden_size
        
        # Projection layer to match the main model's dimension
        self.projection = nn.Linear(self.geneformer_dim, d_model)
        self.d_model = d_model
        
        # Gene ranking approach: rank genes by expression level
        # This mimics Geneformer's tokenization strategy
        self.max_genes = min(token_dim, 2048)  # Limit to model's max sequence length
    
    def forward(self, x):
        # x shape: (batch_size, num_tokens, token_dim)
        batch_size, num_tokens, token_dim = x.shape
        x_flat = x.contiguous().view(batch_size * num_tokens, token_dim)
        
        # --- Geneformer-style tokenization ---
        # Rank genes by expression level (Geneformer's approach)
        # For each sample, select top-k expressed genes
        
        embeddings_list = []
        chunk_size = 32  # Process in smaller chunks for memory efficiency
        
        for i in range(0, x_flat.shape[0], chunk_size):
            end_idx = min(i + chunk_size, x_flat.shape[0])
            chunk = x_flat[i:end_idx]
            
            # Rank genes by expression for each sample in chunk
            chunk_embeddings = []
            for sample in chunk:
                # Get top-k genes by expression level
                nonzero_mask = sample > 0
                if nonzero_mask.sum() == 0:
                    # Handle case with no expressed genes
                    gene_indices = torch.zeros(min(100, self.max_genes), dtype=torch.long, device=sample.device)
                else:
                    # Get indices of top expressed genes
                    _, top_indices = torch.topk(sample, min(nonzero_mask.sum().item(), self.max_genes))
                    gene_indices = top_indices
                
                # Create input_ids (gene indices + 1 to avoid 0 which is padding)
                input_ids = gene_indices + 1
                
                # Pad to consistent length
                if len(input_ids) < self.max_genes:
                    padding = torch.zeros(self.max_genes - len(input_ids), dtype=torch.long, device=sample.device)
                    input_ids = torch.cat([input_ids, padding])
                else:
                    input_ids = input_ids[:self.max_genes]
                
                chunk_embeddings.append(input_ids)
            
            # Stack chunk inputs
            chunk_input_ids = torch.stack(chunk_embeddings)  # (chunk_size, max_genes)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = (chunk_input_ids > 0).long()
            
            # Pass through Geneformer
            with torch.no_grad():
                outputs = self.geneformer(input_ids=chunk_input_ids, attention_mask=attention_mask)
                # Use [CLS] token embedding (first token)
                chunk_pooled = outputs.last_hidden_state[:, 0, :]  # (chunk_size, geneformer_dim)
            
            embeddings_list.append(chunk_pooled)
            
            # Clear cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all chunk results
        pooled_embedding = torch.cat(embeddings_list, dim=0)  # (batch_size * num_tokens, geneformer_dim)
        
        # Project to the model's expected dimension
        projected_embedding = self.projection(pooled_embedding)
        
        # Reshape back to (batch_size, num_tokens, d_model)
        return projected_embedding.view(batch_size, num_tokens, self.d_model)