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
        
        # Reshape to treat each gene separately: (B * num_tokens, token_dim, 1)
        x_reshaped = x.view(batch_size * num_tokens, token_dim, 1)
        
        # Embed each gene expression value
        x_embedded = self.gene_embedding(x_reshaped)  # -> (B * num_tokens, token_dim, d_model)
        
        # Apply self-attention layers (treating genes as set elements)
        for layer, layer_norm in zip(self.layers, self.layer_norms):
            attn_out, _ = layer(x_embedded, x_embedded, x_embedded)
            x_embedded = layer_norm(x_embedded + attn_out)
        
        # Global average pooling across genes
        x_pooled = x_embedded.mean(dim=1)  # -> (B * num_tokens, d_model)
        
        # Reshape back to token format
        output = x_pooled.view(batch_size, num_tokens, self.d_model)
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
        x_reshaped = x.view(batch_size * num_tokens, token_dim, 1)
        gene_features = self.gene_proj(x_reshaped).mean(dim=1)  # -> (B * num_tokens, d_model//2)
        
        # Pairwise gene interactions
        x_flat = x.view(batch_size * num_tokens, token_dim)
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
        x_flat = x.view(batch_size * num_tokens, token_dim)
        
        # Compute statistical features
        mean_vals = x_flat.mean(dim=1, keepdim=True)
        std_vals = x_flat.std(dim=1, keepdim=True)
        min_vals, _ = x_flat.min(dim=1, keepdim=True)
        max_vals, _ = x_flat.max(dim=1, keepdim=True)
        
        # Simple skewness and kurtosis approximations
        centered = x_flat - mean_vals
        skew_approx = (centered ** 3).mean(dim=1, keepdim=True) / (std_vals ** 3 + 1e-8)
        kurt_approx = (centered ** 4).mean(dim=1, keepdim=True) / (std_vals ** 4 + 1e-8)
        
        stats = torch.cat([mean_vals, std_vals, min_vals, max_vals, skew_approx, kurt_approx], dim=1)
        stat_features = self.stat_proj(stats)  # -> (B * num_tokens, d_model//2)
        
        # Also include direct gene representations
        gene_features = self.gene_proj(x_flat)  # -> (B * num_tokens, d_model//2)
        
        # Combine features
        combined = torch.cat([stat_features, gene_features], dim=1)
        output = self.final_proj(combined)
        
        # Reshape back to token format
        return output.view(batch_size, num_tokens, self.d_model)

class HierarchicalTokenEncoder(TokenEncoder):
    """
    Uses hierarchical grouping - first local groups, then global aggregation.
    Better for larger token dimensions.
    """
    def __init__(self, token_dim, d_model, group_size=4):
        super().__init__(token_dim, d_model)
        self.group_size = group_size
        self.num_groups = token_dim // group_size
        
        # Local processing within groups
        self.local_encoder = nn.Linear(group_size, d_model // 2)
        
        # Global processing across groups
        self.global_encoder = nn.Sequential(
            nn.Linear(self.num_groups * (d_model // 2), d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x):
        # x shape: (batch_size, num_tokens, token_dim)
        batch_size, num_tokens, token_dim = x.shape
        x_flat = x.view(batch_size * num_tokens, token_dim)
        
        # Process local groups
        local_features = []
        for i in range(self.num_groups):
            start_idx = i * self.group_size
            end_idx = start_idx + self.group_size
            if end_idx <= token_dim:
                group_data = x_flat[:, start_idx:end_idx]
                group_features = self.local_encoder(group_data)
                local_features.append(group_features)
        
        if local_features:
            local_concat = torch.cat(local_features, dim=1)
            output = self.global_encoder(local_concat)
        else:
            output = torch.zeros(batch_size * num_tokens, self.d_model, device=x.device)
        
        # Reshape back to token format
        return output.view(batch_size, num_tokens, self.d_model)

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
        else:
            raise ValueError(f"Unknown token encoder type: {encoder_type}") 