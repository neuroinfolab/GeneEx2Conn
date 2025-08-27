from env.imports import *
from data.data_utils import create_data_loader
from models.train_val import train_model
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
from torch.cuda.amp import autocast
import os

def scaled_dot_product_attention_with_weights(query, key, value, dropout_p=0.0, is_causal=False, scale=None, apply_dropout=False):
    '''
    Helper function to compute attention output and weights at inference
    '''
    # similar to https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention for inference
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    weights = torch.softmax(scores, dim=-1)
    weights = F.dropout(weights, p=dropout_p, training=apply_dropout)
    output = torch.matmul(weights, value)
    return output, weights

def plot_avg_attention(avg_attn):
    '''
    Helper function to plot average attention weights
    '''
    nhead = avg_attn.shape[0]
    for h in range(nhead):
        plt.figure(figsize=(6, 5))
        vmin, vmax = avg_attn[h].min(), avg_attn[h].max()
        plt.imshow(avg_attn[h], cmap="viridis", vmin=vmin, vmax=vmax)
        plt.colorbar(label=f"Attention Weight [{vmin:.2f}, {vmax:.2f}]")
        plt.title(f"Average Attention Head {h}")
        plt.xlabel("Key")
        plt.ylabel("Query") 
        plt.show()

# === ATTENTION COLLECTION HELPER FUNCTIONS === #
"""
This module provides two types of attention weight collection:

1. Attention Pooling Weights:
   - Used when use_attention_pooling=True in transformer models
   - Collects the attention weights from the AttentionPooling mechanism
   - Returns average attention pattern across all samples and both encoders
   - Saved as .npy files with tokens array showing overall attention importance

2. Full Attention Head Weights:
   - Used when use_attention_pooling=False in transformer models
   - Collects the full multi-head self-attention weights from transformer layers
   - Returns averaged attention weights across all samples and batches
   - Saved as .npy files with the averaged attention matrix
   - Automatically plots the attention heads for visualization

Usage:
    # For attention pooling weights
    model = SharedSelfAttentionModel(..., use_attention_pooling=True)
    preds, targets = model.predict(loader, collect_attn=True, save_attn_path="attn_pooling.npy")
    
    # For full attention head weights  
    model = SharedSelfAttentionModel(..., use_attention_pooling=False)
    preds, targets = model.predict(loader, collect_attn=True, save_attn_path="attn_heads.npy")
"""
def collect_attention_pooling_weights(all_attn, save_attn_path=None):
    """
    Helper function to process and save attention pooling weights
    
    Args:
        all_attn: List of tuples (attn_i, attn_j), each attn is (B, L) per encoder
        save_attn_path: Optional path to save attention weights
    
    Returns:
        avg_attn_arr: numpy array of shape (tokens,) - averaged across all samples and both encoders
    """
    all_attn_list = []
    
    # Collect all attention weights into a single list
    for attn_i, attn_j in all_attn:
        # Append both encoder attention weights to the same list
        all_attn_list.append(attn_i.cpu().numpy())  # (B, L)
        all_attn_list.append(attn_j.cpu().numpy())  # (B, L)
    
    # Concatenate all attention weights into a single matrix
    all_attn_matrix = np.concatenate(all_attn_list, axis=0)  # (total_samples, tokens)
    
    # Average across all samples (both encoders and all batches)
    avg_attn_arr = np.mean(all_attn_matrix, axis=0)  # (tokens,)
    
    # Optionally save
    if save_attn_path is not None:
        np.save(save_attn_path, avg_attn_arr)
    
    return avg_attn_arr

def collect_full_attention_heads(encoder_layers, save_attn_path=None):
    """
    Helper function to collect and process full attention head weights
    
    Args:
        encoder_layers: List of transformer layers with stored attention weights
        save_attn_path: Optional path to save attention weights
    
    Returns:
        avg_attn: Average attention weights across all batches
    """
    # Enable attention collection for all layers
    for layer in encoder_layers:
        layer.store_attn = True
    
    return None  # Will be filled during inference

def process_full_attention_heads(encoder_layers, total_batches, save_attn_path=None):
    """
    Helper function to process collected full attention head weights
    
    Args:
        encoder_layers: List of transformer layers with stored attention weights
        total_batches: Number of batches processed
        save_attn_path: Optional path to save attention weights
    
    Returns:
        avg_attn: Average attention weights from the last layer
    """
    # Get attention weights from the last layer
    last_layer = encoder_layers[-1]
    avg_attn = getattr(last_layer, '_accumulated_attn', None)
    
    if avg_attn is not None and total_batches > 0:
        avg_attn = avg_attn / total_batches
        plot_avg_attention(avg_attn.cpu())
        
        if save_attn_path is not None:
            np.save(save_attn_path, avg_attn.cpu().numpy())
    
    # Clean up
    for layer in encoder_layers:
        layer.store_attn = False
        if hasattr(layer, '_accumulated_attn'):
            delattr(layer, '_accumulated_attn')
    
    return avg_attn

def accumulate_attention_weights(encoder_layers, is_first_batch=False):
    """
    Helper function to accumulate attention weights during inference
    
    Args:
        encoder_layers: List of transformer layers
        is_first_batch: Whether this is the first batch being processed
    """
    last_layer = encoder_layers[-1]
    attn_weights = last_layer.last_attn_weights
    
    if attn_weights is not None:
        batch_avg = attn_weights.mean(dim=0)
        
        if is_first_batch or not hasattr(last_layer, '_accumulated_attn'):
            last_layer._accumulated_attn = batch_avg
        else:
            last_layer._accumulated_attn += batch_avg

# === ATTENTION POOLING IMPLEMENTATION FROM CELLSPLICENET === #
class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.theta1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.theta2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_attn=False):
        B, L, D = x.shape
        # (B, L, D) -> (B, L, hidden_dim) -> (B, hidden_dim, L) -> BN -> (B, L, hidden_dim)
        scores = self.theta2(F.gelu(self.bn(self.theta1(x).transpose(1, 2)).transpose(1, 2)))  # (B, L, 1)
        # Normalize attention scores into probabilities that sum to 1 across sequence length dimension
        attn_weights = torch.softmax(scores, dim=1)  # (B, L, 1)
       # Batched matmul: (B, L, 1)^T @ (B, L, D) -> (B, 1, D) -> (B, D)
        pooled = torch.bmm(attn_weights.transpose(1, 2), x).squeeze(1)
        
        if return_attn:
            return pooled, attn_weights.squeeze(-1)  # (B, D), (B, L)
        
        return pooled
        
# === FLASH ATTENTION IMPLEMENTATION === #
class FlashAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, use_alibi=True):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        # Store attention weights
        self.store_attn = False
        self.last_attn_weights = None

        self.use_alibi = use_alibi
        if use_alibi:
            slopes = self.build_alibi_slopes(nhead)
            self.register_buffer("alibi_slopes", slopes)

    def split_heads(self, x):
        return x.view(x.size(0), x.size(1), self.nhead, self.head_dim).transpose(1, 2)

    def merge_heads(self, x):
        return x.transpose(1, 2).reshape(x.size(0), x.size(2), -1)

    @staticmethod
    def build_alibi_slopes(n_heads):
        slopes = []
        base = 2.0
        for i in range(n_heads):
            power = i // (n_heads // base)
            slopes.append(1.0 / (base ** power))
        return torch.tensor(slopes).float()

    def forward(self, x):
        with autocast(dtype=torch.bfloat16):
            residual = x

            # Project QKV jointly
            qkv = self.qkv_proj(x)  # (B, L, 3 * d_model)
            if self.store_attn:
                q, k, v = qkv.chunk(3, dim=-1)
                q = self.split_heads(q)  # (B, nhead, L, head_dim)
                k = self.split_heads(k)
                v = self.split_heads(v)
                # Manual scaled dot-product attention
                attn_output, attn_weights = scaled_dot_product_attention_with_weights(q, k, v, dropout_p=0.0, is_causal=False)
                self.last_attn_weights = attn_weights.detach().cpu()
                attn_output = self.merge_heads(attn_output)
            else:
                qkv = qkv.view(x.size(0), x.size(1), 3, self.nhead, self.head_dim)  # (B, L, 3, nhead, head_dim)
                attn_output = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=0.0,
                    causal=False,
                    # window_size=(128, 128),
                    alibi_slopes=self.alibi_slopes if self.use_alibi else None
                )
                # ALiBi indicates positionality in the transformer by simply modifying the attention mechanism,
                # biasing the attention mechanism to have words that are away from each other interact less than words that are nearby.
                attn_output = attn_output.transpose(1, 2)  # (B, nhead, L, head_dim)
                attn_output = self.merge_heads(attn_output)
            
            attn_output = self.attn_dropout(attn_output)

            x = self.attn_norm(residual + attn_output)
            residual = x

            x = self.ffn(x)
            x = self.ffn_norm(residual + x)

            return x

class FlashAttentionEncoder(nn.Module):
    def __init__(self, token_encoder_dim, d_model, output_dim, nhead=4, num_layers=4, dropout=0.1, use_alibi=False, num_tokens=None, use_attention_pooling=True):
        super().__init__()
        
        self.input_projection = nn.Linear(token_encoder_dim, d_model)

        self.layers = nn.ModuleList([
            FlashAttentionBlock(d_model, nhead, dropout, use_alibi=use_alibi)
            for _ in range(num_layers)
        ])
        
        self.use_attention_pooling = use_attention_pooling
        if self.use_attention_pooling:
            self.pooling = AttentionPooling(d_model, hidden_dim=32)
        else:
            self.output_projection = nn.Linear(d_model, output_dim)
        
        self.num_tokens = num_tokens

    def forward(self, x, return_attn=False):
        B, T = x.shape
        x = x.view(B, -1, self.input_projection.in_features)
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x)
        if self.use_attention_pooling:
            return self.pooling(x, return_attn=return_attn)
        else:
            x = self.output_projection(x)  # shape: (B, L, encoder_output_dim)
            x = x.flatten(start_dim=1)     # shape: (B, L * encoder_output_dim)
            if return_attn:
                return x, None
            return x

class SharedSelfAttentionModel(nn.Module):
    def __init__(self, input_dim, token_encoder_dim=20, d_model=128, encoder_output_dim=10, nhead=2, num_layers=2, deep_hidden_dims=[256, 128],
                 use_alibi=False, transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0,
                 batch_size=256, aug_prob=0.0, epochs=100, num_workers=2, prefetch_factor=2, use_attention_pooling=False):
        super().__init__()
        
        self.input_dim = input_dim // 2
        self.token_encoder_dim = token_encoder_dim
        self.d_model = d_model
        self.encoder_output_dim = encoder_output_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.aug_prob = aug_prob
        self.use_alibi = use_alibi
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_tokens = self.input_dim // self.token_encoder_dim
        self.use_attention_pooling = use_attention_pooling

        self.encoder = FlashAttentionEncoder(
            token_encoder_dim=self.token_encoder_dim,
            d_model=self.d_model,
            output_dim=self.encoder_output_dim,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=transformer_dropout,
            use_alibi=self.use_alibi,
            num_tokens=self.num_tokens,
            use_attention_pooling=self.use_attention_pooling,
        )
        self.encoder = torch.compile(self.encoder) # doubles training speed

        if not self.use_attention_pooling:
            prev_dim = (self.encoder_output_dim * self.num_tokens * 2)
        else:
            prev_dim = self.d_model * 2
        
        deep_layers = []
        for hidden_dim in deep_hidden_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim, dtype=torch.float32),
                nn.Dropout(dropout_rate)])
            prev_dim = hidden_dim
        self.deep_layers = nn.Sequential(*deep_layers)
        self.output_layer = nn.Linear(prev_dim, 1)

        # Calculate and display total number of learnable parameters
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in SMT model: {num_params}")
        
        self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.patience = 45
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.3,
            patience=20,
            threshold=0.1,
            cooldown=1,
            min_lr=1e-6,
            verbose=True
        )
        self.store_attn = False

    def forward(self, x):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        if self.store_attn:
            x_i, attn_beta_i = self.encoder(x_i, return_attn=True)
            x_j, attn_beta_j = self.encoder(x_j, return_attn=True)
        else:
            x_i = self.encoder(x_i)
            x_j = self.encoder(x_j)

        x = torch.cat([x_i, x_j], dim=1)
        y_pred = self.output_layer(self.deep_layers(x))
        y_pred = torch.clamp(y_pred, min=-1.0, max=1.0)
        
        if self.store_attn and self.use_attention_pooling:
            return {"output": y_pred.squeeze(), "attn_beta_i": attn_beta_i, "attn_beta_j": attn_beta_j}
        
        return y_pred.squeeze()

    def predict(self, loader, collect_attn=False, save_attn_path=None):
        """
        Make predictions on a data loader with optional attention weight collection
        
        Args:
            loader: DataLoader for prediction
            collect_attn: Whether to collect attention weights
            save_attn_path: Optional path to save attention weights
            
        Returns:
            (predictions, targets) - consistent format regardless of attention collection mode
            Attention weights are automatically saved to save_attn_path if provided
        """
        self.eval()
        predictions = []
        targets = []
        
        # Initialize attention collection based on pooling type
        all_attn = []  # For attention pooling weights
        total_batches = 0  # For full attention head weights
        
        # Set up attention collection
        self.store_attn = collect_attn
        if collect_attn and not self.use_attention_pooling:
            # Enable full attention head collection
            collect_full_attention_heads(self.encoder.layers)
        
        with torch.no_grad():
            for batch_idx, (batch_X, batch_y, batch_coords, _) in enumerate(loader):
                batch_X = batch_X.to(self.device)
                
                if collect_attn:
                    out = self(batch_X)
                    
                    if self.use_attention_pooling:
                        # Extract attention pooling weights
                        batch_preds = out["output"].cpu().numpy()
                        attns = (out["attn_beta_i"], out["attn_beta_j"])
                        all_attn.append(attns)
                    else:
                        # Accumulate full attention head weights
                        batch_preds = out.cpu().numpy()
                        accumulate_attention_weights(self.encoder.layers, is_first_batch=(batch_idx == 0))
                        total_batches += 1
                else:
                    batch_preds = self(batch_X).cpu().numpy()
                
                predictions.append(batch_preds)
                targets.append(batch_y.numpy())
        
        # Concatenate results
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        # Process and save attention weights if collected
        self.store_attn = False
        
        if collect_attn:
            if self.use_attention_pooling:
                # Process attention pooling weights - returns (tokens,) array averaged across all samples
                avg_attn_arr = collect_attention_pooling_weights(all_attn, save_attn_path)
            else:
                # Process full attention head weights
                avg_attn = process_full_attention_heads(self.encoder.layers, total_batches, save_attn_path)
        
        return predictions, targets
        
    def fit(self, dataset, train_indices, test_indices, verbose=True):
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        
        # print(f"Using {self.num_workers} workers and {self.prefetch_factor} prefetch factor")
        # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers, persistent_workers=True, prefetch_factor=self.prefetch_factor)
        # test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers, persistent_workers=True, prefetch_factor=self.prefetch_factor)
       
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, self.scheduler, verbose=verbose, dataset=dataset)


# === SelfAttentionCLSEncoder using FlashAttention, and CLS token === # 
class SelfAttentionCLSEncoder(nn.Module):    
    def __init__(self, token_encoder_dim, d_model, output_dim, nhead=4, num_layers=4, dropout=0.1, cls_init='spatial_learned', use_alibi=False, num_tokens=None, use_attention_pooling=True):
        super().__init__()
        self.token_encoder_dim = token_encoder_dim
        self.d_model = d_model
        self.cls_init = cls_init
        self.use_alibi = use_alibi
        self.num_tokens = num_tokens
        self.use_attention_pooling = use_attention_pooling
        
        # Token projection
        self.input_projection = nn.Linear(token_encoder_dim, d_model)
        self.coord_to_cls = nn.Linear(3, d_model)
        
        @staticmethod
        def build_coord_remap_table(coord_tensor, seed=42, scale=100.0):
            torch.manual_seed(seed)
            remap = (torch.rand_like(coord_tensor) * 2 - 1) * scale  # [-100, 100]
            return remap
        
        # CLS token randomization/projection freeze
        if self.cls_init == 'random_learned':
            df = pd.read_csv('./data/UKBB/atlas-4S456Parcels_dseg_reformatted.csv')
            region_coords = torch.tensor(df.iloc[:, -3:].values, dtype=torch.float32)
            remap_tensor = self.build_coord_remap_table(region_coords)
            self.register_buffer("coord_ref_table", region_coords)
            self.register_buffer("coord_remap_table", remap_tensor)
        elif self.cls_init == 'spatial_fixed':
            self.coord_to_cls.weight.requires_grad = False
            self.coord_to_cls.bias.requires_grad = False            

        # Transformer layers
        self.layers = nn.ModuleList([
            FlashAttentionBlock(d_model, nhead, dropout, use_alibi=self.use_alibi)
            for _ in range(num_layers)
        ])

        # Register per-head ALiBi slopes once if use_alibi is True
        if self.use_alibi:
            slopes = FlashAttentionBlock.build_alibi_slopes(nhead)
            self.register_buffer("alibi_slopes", slopes)
            for layer in self.layers:
                layer.alibi_slopes = self.alibi_slopes

        if self.use_attention_pooling:
            self.pooling = AttentionPooling(d_model, hidden_dim=32)
        else:
            self.output_projection = nn.Linear(d_model, output_dim)
    
    def replace_coords_with_remapped(self, coords):
        B, _ = coords.shape
        match = (coords[:, None, :] == self.coord_ref_table[None, :, :]).all(dim=-1)
        idx = match.float().argmax(dim=1)
        remapped = self.coord_remap_table[idx]
        return remapped  # shape (B, 3)
    
    def forward(self, gene_exp, coords, return_attn=False):
        B, T = gene_exp.shape
        x = gene_exp.view(B, -1, self.token_encoder_dim)
        x = self.input_projection(x)
        if self.cls_init == 'random_learned':
            coords = self.replace_coords_with_remapped(coords)
        cls_token = self.coord_to_cls(coords).unsqueeze(1)
        x = torch.cat([cls_token, x], dim=1)
        for layer in self.layers:
            x = layer(x)
        if self.use_attention_pooling:
            return self.pooling(x, return_attn=return_attn)
        else:
            x = self.output_projection(x)  # shape: (B, L, encoder_output_dim)
            x = x.flatten(start_dim=1)     # shape: (B, L * encoder_output_dim)
            if return_attn:
                return x, None
            return x


class SharedSelfAttentionCLSModel(nn.Module):
    def __init__(self, input_dim, token_encoder_dim=20, d_model=128, encoder_output_dim=10, nhead=2, num_layers=2, deep_hidden_dims=[256, 128], 
                 cls_init='spatial_learned', use_alibi=False, transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0, 
                 batch_size=128, epochs=100, aug_prob=0.0, num_workers=2, prefetch_factor=2, use_attention_pooling=False):
        super().__init__()
        self.include_coords = True
        self.cls_init = cls_init
        self.aug_prob = aug_prob
        self.use_alibi = use_alibi
        self.input_dim = input_dim // 2
        self.token_encoder_dim = token_encoder_dim 
        self.d_model = d_model
        self.encoder_output_dim = encoder_output_dim
        self.transformer_dropout = transformer_dropout
        self.nhead = nhead
        self.num_layers = num_layers
        self.deep_hidden_dims = deep_hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers=num_workers
        self.prefetch_factor=prefetch_factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_tokens = self.input_dim // self.token_encoder_dim
        self.use_attention_pooling = use_attention_pooling

        # Create self-attention encoder
        self.encoder = SelfAttentionCLSEncoder(
            token_encoder_dim=self.token_encoder_dim,
            d_model=self.d_model,
            output_dim=self.encoder_output_dim, 
            nhead=self.nhead, 
            num_layers=self.num_layers,
            dropout=self.transformer_dropout,
            cls_init=self.cls_init,
            use_alibi=self.use_alibi,
            num_tokens=self.num_tokens,
            use_attention_pooling=self.use_attention_pooling,
        )
        self.encoder = torch.compile(self.encoder)

        if not self.use_attention_pooling:
            prev_dim = (self.encoder_output_dim * (self.num_tokens + 1) * 2)
        else:
            prev_dim = self.d_model * 2
        deep_layers = []
        for hidden_dim in self.deep_hidden_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim, dtype=torch.float32), 
                nn.Dropout(dropout_rate)])
            prev_dim = hidden_dim
        self.deep_layers = nn.Sequential(*deep_layers)
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # Calculate and display total number of learnable parameters
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in SMT w/ CLS model: {num_params}")

        self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.patience = 45
        self.scheduler = ReduceLROnPlateau( 
            self.optimizer, 
            mode='min', 
            factor=0.3,
            patience=20,
            threshold=0.1,
            cooldown=1,
            min_lr=1e-6,
            verbose=True
        )
        self.store_attn = False

    def forward(self, x, coords):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        coords_i, coords_j = torch.chunk(coords, chunks=2, dim=1)
        if self.store_attn:
            x_i, attn_beta_i = self.encoder(x_i, coords_i, return_attn=True)
            x_j, attn_beta_j = self.encoder(x_j, coords_j, return_attn=True)
        else:
            x_i = self.encoder(x_i, coords_i)
            x_j = self.encoder(x_j, coords_j)
        x = torch.cat([x_i, x_j], dim=1)
        y_pred = self.output_layer(self.deep_layers(x))
        y_pred = torch.clamp(y_pred, min=-0.8, max=1.0)
        if self.store_attn and self.use_attention_pooling:
            return {"output": y_pred.squeeze(), "attn_beta_i": attn_beta_i, "attn_beta_j": attn_beta_j}
        return y_pred.squeeze()
        
    def predict(self, loader, collect_attn=False, save_attn_path=None):
        """
        Make predictions on a data loader with optional attention weight collection
        
        Args:
            loader: DataLoader for prediction
            collect_attn: Whether to collect attention weights
            save_attn_path: Optional path to save attention weights
            
        Returns:
            (predictions, targets) - consistent format regardless of attention collection mode
            Attention weights are automatically saved to save_attn_path if provided
        """
        self.eval()
        predictions = []
        targets = []
        
        # Initialize attention collection based on pooling type
        all_attn = []  # For attention pooling weights
        total_batches = 0  # For full attention head weights
        
        # Set up attention collection
        self.store_attn = collect_attn
        if collect_attn and not self.use_attention_pooling:
            # Enable full attention head collection
            collect_full_attention_heads(self.encoder.layers)
        
        with torch.no_grad():
            for batch_idx, (batch_X, batch_y, batch_coords, _) in enumerate(loader):
                batch_X = batch_X.to(self.device)
                batch_coords = batch_coords.to(self.device)
                
                if collect_attn:
                    out = self(batch_X, batch_coords)
                    
                    if self.use_attention_pooling:
                        # Extract attention pooling weights
                        batch_preds = out["output"].cpu().numpy()
                        attns = (out["attn_beta_i"], out["attn_beta_j"])
                        all_attn.append(attns)
                    else:
                        # Accumulate full attention head weights
                        batch_preds = out.cpu().numpy()
                        accumulate_attention_weights(self.encoder.layers, is_first_batch=(batch_idx == 0))
                        total_batches += 1
                else:
                    batch_preds = self(batch_X, batch_coords).cpu().numpy()
                
                predictions.append(batch_preds)
                targets.append(batch_y.numpy())
        
        # Concatenate results
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        # Process and save attention weights if collected
        self.store_attn = False
        
        if collect_attn:
            if self.use_attention_pooling:
                # Process attention pooling weights - returns (tokens,) array averaged across all samples
                avg_attn_arr = collect_attention_pooling_weights(all_attn, save_attn_path)
            else:
                # Process full attention head weights
                avg_attn = process_full_attention_heads(self.encoder.layers, total_batches, save_attn_path)
        
        return predictions, targets
    
    def fit(self, dataset, train_indices, test_indices, verbose=True):
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        
        # parallel loaders
        # print(f"Using {self.num_workers} workers and {self.prefetch_factor} prefetch factor")
        # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers, persistent_workers=True, prefetch_factor=self.prefetch_factor)
        # test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers, persistent_workers=True, prefetch_factor=self.prefetch_factor)
       
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, self.scheduler, verbose=verbose, dataset=dataset)


# === CrossAttentionEncoder ===
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

            # Pytorch SDPA implementation - will use flash attention backend if available
            '''
            q = self.split_heads(q)
            k = self.split_heads(k)
            v = self.split_heads(v)
            attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            attn_output = self.merge_heads(attn_output)
            attn_output = self.attn_dropout(attn_output)
            '''

            # FlashAttention implementation - expects (B, L, nhead, head_dim)
            q = self.split_heads(q).transpose(1, 2)  # (B, L, nhead, head_dim)
            k = self.split_heads(k).transpose(1, 2)
            v = self.split_heads(v).transpose(1, 2)
            attn_output = flash_attn_func(q, k, v, dropout_p=0.0, causal=False) # , window_size=(128, 128)) # experiment with window size
            attn_output = attn_output.transpose(1, 2)  # (B, nhead, L, head_dim)        
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
        self.layers = nn.ModuleList([
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

        for layer in self.layers:
            q = layer(q, kv)

        out = self.output_projection(q)
        out = out.flatten(start_dim=1)
        return out

# Update CrossAttentionModel to pass num_layers to encoder
class CrossAttentionModel(nn.Module):
    def __init__(self, input_dim, token_encoder_dim=20, d_model=128, encoder_output_dim=10, nhead=2, num_layers=2, deep_hidden_dims=[256, 128],
                 transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0,
                 batch_size=128, epochs=100):

        super().__init__()

        self.input_dim = input_dim // 2
        self.token_encoder_dim = token_encoder_dim
        self.d_model = d_model
        self.encoder_output_dim = encoder_output_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.deep_layers = nn.Sequential(
            nn.Linear(prev_dim, deep_hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(deep_hidden_dims[0]),
            nn.Dropout(dropout_rate),
            nn.Linear(deep_hidden_dims[0], 1)
        )

        self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.patience = 40
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.3,
            patience=20,
            threshold=0.1,
            cooldown=1,
            min_lr=1e-6,
            verbose=True
        )

    def forward(self, x):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        encoded = self.encoder(x_i, x_j)
        return self.deep_layers(encoded).squeeze()

    def predict(self, loader):
        self.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for batch_X, batch_y, _, _ in loader:
                batch_X = batch_X.to(self.device)
                batch_preds = self(batch_X).cpu().numpy()
                predictions.append(batch_preds)
                targets.append(batch_y.numpy())
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        return ((predictions > 0.5).astype(int) if self.binarize else predictions), targets

    def fit(self, dataset, train_indices, test_indices, verbose=True):
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, self.scheduler, verbose=verbose, dataset=dataset)

from torch.nn import RMSNorm

# === FlashAttentionBlockConv ===
class FlashAttentionBlockConv(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, use_alibi=False):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Conv1d QKV projection
        self.qkv_proj = nn.Conv1d(in_channels=d_model, out_channels=d_model * 3, kernel_size=3, padding=1)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = RMSNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = RMSNorm(d_model)

        # Store attention weights
        self.store_attn = False
        self.last_attn_weights = None

        self.use_alibi = use_alibi
        if use_alibi:
            slopes = self.build_alibi_slopes(nhead)
            self.register_buffer("alibi_slopes", slopes)

    def split_heads(self, x):
        return x.view(x.size(0), x.size(1), self.nhead, self.head_dim).transpose(1, 2)

    def merge_heads(self, x):
        return x.transpose(1, 2).reshape(x.size(0), x.size(2), -1)

    @staticmethod
    def build_alibi_slopes(n_heads):
        slopes = []
        base = 2.0
        for i in range(n_heads):
            power = i // (n_heads // base)
            slopes.append(1.0 / (base ** power))
        return torch.tensor(slopes).float()

    def forward(self, x):
        with autocast(dtype=torch.bfloat16):
            residual = x
            # x: (B, L, d_model)
            x_conv = x.transpose(1, 2)  # (B, d_model, L)
            qkv = self.qkv_proj(x_conv).transpose(1, 2)  # (B, L, 3*d_model)
            if self.store_attn:
                q, k, v = qkv.chunk(3, dim=-1)
                q = self.split_heads(q)  # (B, nhead, L, head_dim)
                k = self.split_heads(k)
                v = self.split_heads(v)
                attn_output, attn_weights = scaled_dot_product_attention_with_weights(q, k, v, dropout_p=0.0, is_causal=False)
                self.last_attn_weights = attn_weights.detach().cpu()
                attn_output = self.merge_heads(attn_output)
            else:
                qkv = qkv.view(x.size(0), x.size(1), 3, self.nhead, self.head_dim)  # (B, L, 3, nhead, head_dim)
                attn_output = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=0.0,
                    causal=False,
                    alibi_slopes=self.alibi_slopes if self.use_alibi else None
                )
                attn_output = attn_output.transpose(1, 2)  # (B, nhead, L, head_dim)
                attn_output = self.merge_heads(attn_output)
            attn_output = self.attn_dropout(attn_output)

            x = self.attn_norm(residual + attn_output)
            residual = x

            x = self.ffn(x)
            x = self.ffn_norm(residual + x)

            return x

# === SelfAttentionCLSEncoderConv using FlashAttentionBlockConv, FlashAttention, and CLS token === #
class SelfAttentionCLSEncoderConv(nn.Module):
    def __init__(self, token_encoder_dim, d_model, output_dim, nhead=4, num_layers=4, dropout=0.1, cls_init='spatial_learned', use_alibi=False):
        super().__init__()
        self.token_encoder_dim = token_encoder_dim
        self.d_model = d_model
        self.cls_init = cls_init
        self.use_alibi = use_alibi

        # Token projection: Conv1d over gene feature dimension
        self.input_projection = nn.Conv1d(in_channels=token_encoder_dim, out_channels=d_model, kernel_size=3, padding=1)
        self.coord_to_cls = nn.Linear(3, d_model)

        @staticmethod
        def build_coord_remap_table(coord_tensor, seed=42, scale=100.0):
            torch.manual_seed(seed)
            remap = (torch.rand_like(coord_tensor) * 2 - 1) * scale
            return remap

        # CLS token randomization/projection freeze
        if self.cls_init == 'random_learned':
            df = pd.read_csv('./data/UKBB/atlas-4S456Parcels_dseg_reformatted.csv')
            region_coords = torch.tensor(df.iloc[:, -3:].values, dtype=torch.float32)
            remap_tensor = self.build_coord_remap_table(region_coords)
            self.register_buffer("coord_ref_table", region_coords)
            self.register_buffer("coord_remap_table", remap_tensor)
        elif self.cls_init == 'spatial_fixed':
            self.coord_to_cls.weight.requires_grad = False
            self.coord_to_cls.bias.requires_grad = False

        # Transformer layers
        self.layers = nn.ModuleList([
            FlashAttentionBlockConv(d_model, nhead, dropout, use_alibi=self.use_alibi)
            for _ in range(num_layers)
        ])

        # Register per-head ALiBi slopes once if use_alibi is True
        if self.use_alibi:
            slopes = FlashAttentionBlockConv.build_alibi_slopes(nhead)
            self.register_buffer("alibi_slopes", slopes)
            for layer in self.layers:
                layer.alibi_slopes = self.alibi_slopes

        self.output_projection = nn.Linear(d_model, output_dim)

    def replace_coords_with_remapped(self, coords):
        B, _ = coords.shape
        match = (coords[:, None, :] == self.coord_ref_table[None, :, :]).all(dim=-1)
        idx = match.float().argmax(dim=1)
        remapped = self.coord_remap_table[idx]
        return remapped

    def forward(self, gene_exp, coords):
        batch_size, total_features = gene_exp.shape
        L = total_features // self.token_encoder_dim

        gene_exp = gene_exp.view(batch_size, L, self.token_encoder_dim).transpose(1, 2)  # (B, token_encoder_dim, L)
        x_proj = self.input_projection(gene_exp).transpose(1, 2)  # (B, L, d_model)

        if self.cls_init == 'random_learned':
            coords = self.replace_coords_with_remapped(coords)

        cls_token = self.coord_to_cls(coords).unsqueeze(1)  # (B, 1, d_model)
        x = torch.cat([cls_token, x_proj], dim=1)  # (B, L+1, d_model)

        for layer in self.layers:
            x = layer(x)

        x = self.output_projection(x)
        x = x.reshape(batch_size, -1)
        return x

# === SharedSelfAttentionConvCLSModel === #
class SharedSelfAttentionConvCLSModel(nn.Module):
    def __init__(self, input_dim, token_encoder_dim=20, d_model=128, encoder_output_dim=10, nhead=2, num_layers=2, deep_hidden_dims=[256, 128],
                 cls_init='spatial_learned', use_alibi=False, transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0,
                 batch_size=128, epochs=100, aug_prob=0.0, num_workers=2, prefetch_factor=2):
        super().__init__()

        self.include_coords = True
        self.cls_init = cls_init
        self.aug_prob = aug_prob
        self.use_alibi = use_alibi
        self.input_dim = input_dim // 2
        self.token_encoder_dim = token_encoder_dim
        self.d_model = d_model
        self.encoder_output_dim = encoder_output_dim
        self.transformer_dropout = transformer_dropout
        self.nhead = nhead
        self.num_layers = num_layers
        self.deep_hidden_dims = deep_hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoder with Conv1d QKV and RMSNorm
        self.encoder = SelfAttentionCLSEncoderConv(
            token_encoder_dim=self.token_encoder_dim,
            d_model=self.d_model,
            output_dim=self.encoder_output_dim,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.transformer_dropout,
            cls_init=self.cls_init,
            use_alibi=self.use_alibi
        )
        self.encoder = torch.compile(self.encoder)

        prev_dim = (self.input_dim // self.token_encoder_dim * self.encoder_output_dim) * 2 + 2 * self.encoder_output_dim
        deep_layers = []
        for hidden_dim in self.deep_hidden_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim, dtype=torch.float32),
                nn.Dropout(dropout_rate)])
            prev_dim = hidden_dim
        self.deep_layers = nn.Sequential(*deep_layers)
        self.output_layer = nn.Linear(prev_dim, 1)

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in SMT Conv1D w/ CLS model: {num_params}")

        self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.patience = 45
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.3,
            patience=20,
            threshold=0.1,
            cooldown=1,
            min_lr=1e-6,
            verbose=True
        )

    def forward(self, x, coords):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        coords_i, coords_j = torch.chunk(coords, chunks=2, dim=1)

        encoded_i = self.encoder(x_i, coords_i)
        encoded_j = self.encoder(x_j, coords_j)

        pairwise_embedding = torch.cat((encoded_i, encoded_j), dim=1)
        deep_output = self.deep_layers(pairwise_embedding)
        output = self.output_layer(deep_output)
        pred = torch.clamp(output, min=-0.8, max=1.0)  # Clip output to [-1, 1]
        return pred.squeeze()

    def predict(self, loader, collect_attn=False, save_attn_path=None):
        """
        Make predictions on a data loader with optional attention weight collection
        
        Args:
            loader: DataLoader for prediction
            collect_attn: Whether to collect attention weights
            save_attn_path: Optional path to save attention weights
            
        Returns:
            (predictions, targets)
        
        Note: This model only supports full attention head collection, not attention pooling
        """
        self.eval()
        predictions = []
        targets = []
        
        # Initialize attention collection
        total_batches = 0
        
        # Set up attention collection if requested
        if collect_attn:
            collect_full_attention_heads(self.encoder.layers)
        
        with torch.no_grad():
            for batch_idx, (batch_X, batch_y, batch_coords, _) in enumerate(loader):
                batch_X = batch_X.to(self.device)
                batch_coords = batch_coords.to(self.device)
                batch_preds = self(batch_X, batch_coords).cpu().numpy()
                predictions.append(batch_preds)
                targets.append(batch_y.numpy())
                
                if collect_attn:
                    # Accumulate full attention head weights
                    accumulate_attention_weights(self.encoder.layers, is_first_batch=(batch_idx == 0))
                    total_batches += 1
        
        # Concatenate results
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        # Process attention weights if collected
        if collect_attn:
            avg_attn = process_full_attention_heads(self.encoder.layers, total_batches, save_attn_path)
        
        return predictions, targets

    def fit(self, dataset, train_indices, test_indices, verbose=True):
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        print(f"Using {self.num_workers} workers and {self.prefetch_factor} prefetch factor")
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers, persistent_workers=True, prefetch_factor=self.prefetch_factor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers, persistent_workers=True, prefetch_factor=self.prefetch_factor)
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, self.scheduler, verbose=verbose, dataset=dataset)