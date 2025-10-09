import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
from torch.cuda.amp import autocast
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.stats as stats
from scipy.stats import spearmanr

# === CORE TRANSFORMER CLASSES === #
class AttentionPooling(nn.Module):
    # CellSpliceNet implementation
    def __init__(self, input_dim, hidden_dim, use_residual=True):
        super().__init__()
        self.theta1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.theta2 = nn.Linear(hidden_dim, 1)
        # Dropout layers (automatically disabled in eval mode)
        self.dropout_gelu = nn.Dropout(0.1)
        self.dropout_theta2 = nn.Dropout(0.1)
        
        # FiLM modulation parameters (for CLS token based conditioning)
        self.gamma_proj = nn.Linear(input_dim, input_dim)
        self.beta_proj = nn.Linear(input_dim, input_dim)
        self.use_residual = use_residual

    def forward(self, x, return_attn=False, cls_token=False):
        B, L, D = x.shape
        
        if cls_token:
            # Split CLS from rest so doesn't participate in attention pooling
            cls_token, x_rest = x[:, 0:1, :], x[:, 1:, :]

            # Project features and apply dropout after GELU
            x1 = self.theta1(x_rest).transpose(1, 2)                # (B, hidden_dim, L-1)
            x1 = self.bn(x1).transpose(1, 2)                        # (B, L-1, hidden_dim)
            x1 = F.gelu(x1)
            x1 = self.dropout_gelu(x1)

            # Project to scalar scores and apply dropout again before softmax
            scores = self.theta2(x1)                                # (B, L-1, 1)
            scores = self.dropout_theta2(scores)
            attn_weights = torch.softmax(scores, dim=1)             # (B, L-1, 1)

            x_gene = torch.bmm(attn_weights.transpose(1, 2), x_rest).squeeze(1)  # (B, D)

            # === FiLM conditioning from CLS token ===
            gamma = self.gamma_proj(cls_token).squeeze(1)        # (B, D)
            beta = self.beta_proj(cls_token).squeeze(1)          # (B, D)
            modulated = gamma * x_gene + beta                    # (B, D) - affine transformation to gene expression data

            # Residual connection
            pooled = x_gene + modulated if self.use_residual else modulated
        else:
            # Project features and apply dropout after GELU
            x1 = self.theta1(x).transpose(1, 2)                     # (B, hidden_dim, L)
            x1 = self.bn(x1).transpose(1, 2)                        # (B, L, hidden_dim)
            x1 = F.gelu(x1)
            x1 = self.dropout_gelu(x1)

            # Project to scalar scores and apply dropout again before softmax
            scores = self.theta2(x1)                                # (B, L, 1)
            scores = self.dropout_theta2(scores)
            attn_weights = torch.softmax(scores, dim=1)             # (B, L, 1)

            # Weighted sum over tokens
            pooled = torch.bmm(attn_weights.transpose(1, 2), x).squeeze(1)  # (B, D)
        
        if return_attn:
            return pooled, attn_weights.squeeze(-1)                 # (B, D), (B, L)
        return pooled

class FlashAttentionBlock(nn.Module):
    # https://github.com/Dao-AILab/flash-attention
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
        # geometric sequence approach from Press, 2021 https://openreview.net/pdf?id=R8sQPpGCv0
        slopes = []
        base = 2.0
        for i in range(n_heads):
            power = i // (n_heads // base)
            slopes.append(1.0 / (base ** power))
        return torch.tensor(slopes).float()

    def forward(self, x):
        with autocast(dtype=torch.bfloat16):
            residual = x
            qkv = self.qkv_proj(x)
            
            if self.store_attn:
                q, k, v = qkv.chunk(3, dim=-1)
                q = self.split_heads(q)
                k = self.split_heads(k)
                v = self.split_heads(v)
                attn_output, attn_weights = scaled_dot_product_attention_with_weights(q, k, v, dropout_p=0.0, is_causal=False)
                self.last_attn_weights = attn_weights.detach().cpu()
                attn_output = self.merge_heads(attn_output)
            else:
                qkv = qkv.view(x.size(0), x.size(1), 3, self.nhead, self.head_dim)
            
                attn_output = flash_attn_qkvpacked_func(
                    qkv, dropout_p=0.0, causal=False,
                    alibi_slopes=self.alibi_slopes if self.use_alibi else None,
                )
                attn_output = attn_output.transpose(1, 2)
                attn_output = self.merge_heads(attn_output)
            
            attn_output = self.attn_dropout(attn_output)
            x = self.attn_norm(residual + attn_output)
            residual = x
            x = self.ffn(x)
            x = self.ffn_norm(residual + x)
            return x


class SymmetricLoss(nn.Module):
    """
    Loss function with symmetry regularization.
    Encourages model predictions for (i,j) to be close to predictions for (j,i).
    
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
        
    def forward(self, predictions, targets, model=None, batch_X=None, batch_coords=None, batch_idx=None, dataset=None):
        """
        Compute loss with optional symmetry regularization.
        
        Args:
            predictions: Model predictions for current batch
            targets: Ground truth targets
            model: Model instance (required for symmetry loss)
            batch_X: Input features (required for symmetry loss)
            batch_coords: Coordinate features (required for symmetry loss)  
            batch_idx: Batch indices (required for symmetry loss)
            dataset: Dataset instance (required for computing symmetric index)
        Returns:
            Total loss (base loss + symmetry regularization)
        """
        # Base loss
        base_loss = self.base_criterion(predictions, targets)
        
        # If no model or lambda is 0, return base loss only
        if model is None or self.lambda_sym == 0 or batch_X is None:
            return base_loss
        
        # Switch gene expression
        num_genes = batch_X.shape[1] // 2
        X_i, X_j = batch_X[:, :num_genes], batch_X[:, num_genes:]
        symmetric_X = torch.cat([X_j, X_i], dim=1)
        
        # Switch coords
        if batch_coords is not None:
            coords_i, coords_j = batch_coords[:, :3], batch_coords[:, 3:]
            symmetric_coords = torch.cat([coords_j, coords_i], dim=1)
        else:
            symmetric_coords = None
        
        # Symmetric indices
        if batch_idx is not None:
            symmetric_idx = batch_idx ^ 1 # see data_utils.expand_X_symmetric logic
        else:
            symmetric_idx = None
        
        if symmetric_coords is not None and symmetric_idx is not None:
            symmetric_predictions = model(symmetric_X, symmetric_coords, symmetric_idx).squeeze()
        else: # this case is for non-SMT models
            symmetric_predictions = model(symmetric_X).squeeze()
        
        # Compute symmetry loss: penalize difference between pred(i,j) and pred(j,i)
        symmetry_loss = torch.mean((predictions - symmetric_predictions) ** 2)
        total_loss = base_loss + self.lambda_sym * symmetry_loss
        
        return total_loss

# === CORE TRANSFORMER HELPER FUNCS === #
def scaled_dot_product_attention_with_weights(query, key, value, dropout_p=0.0, is_causal=False, scale=None, apply_dropout=False):
    """Helper function to compute attention output and weights at inference"""
    # similar to https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention for inference
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    weights = torch.softmax(scores, dim=-1)
    weights = F.dropout(weights, p=dropout_p, training=apply_dropout)
    output = torch.matmul(weights, value)
    return output, weights

def create_gene_tokens(token_chunk_size=20, verbose=True):
    """
    Create gene tokens and chromosome switch points based on token chunk size

    Args:
        token_chunk_size (int): Number of genes per token
        
    Returns:
        gene_tokens (list): List of dictionaries containing token info
        chrom_switches (list): List of indices where chromosomes switch
    """
    # Get valid genes from load_transcriptome
    from data.data_load import load_transcriptome

    X, valid_genes = load_transcriptome(parcellation='S456', gene_list='0.2', dataset='AHBA', 
                                      run_PCA=False, omit_subcortical=False, hemisphere='both',
                                      impute_strategy='mirror_interpolate', sort_genes='refgenome', 
                                      return_valid_genes=True, null_model='none', random_seed=42)

    # Get reference genome info for valid genes
    refgenome = pd.read_csv('./data/enigma/gene_lists/human_refgenome_ordered.csv')
    valid_refgenome = refgenome[refgenome['gene_id'].isin(valid_genes)].drop_duplicates(subset='gene_id', keep='first')
    
    # Create gene groups/tokens
    num_tokens = len(valid_refgenome) // token_chunk_size
    
    gene_tokens = []

    for i in range(num_tokens):
        start_idx = i * token_chunk_size
        end_idx = start_idx + token_chunk_size
        token_genes = valid_refgenome.iloc[start_idx:end_idx]
        
        # Get unique chromosome for this token
        chrom = token_genes['chromosome'].iloc[0]
        
        token = {
            'token_id': i,
            'genes': token_genes['gene_id'].tolist(),
            'chromosome': chrom
        }
        gene_tokens.append(token)

    # Handle remaining genes if any
    remaining_genes = len(valid_refgenome) % token_chunk_size
    if remaining_genes > 0:
        start_idx = num_tokens * token_chunk_size
        token_genes = valid_refgenome.iloc[start_idx:]
        token = {
            'token_id': num_tokens,
            'genes': token_genes['gene_id'].tolist(), 
            'chromosome': token_genes['chromosome'].iloc[0]
        }
        gene_tokens.append(token)
        
    # Get chromosome switch points
    chrom_switches = []
    for i in range(1, len(gene_tokens)):
        if gene_tokens[i]['chromosome'] != gene_tokens[i-1]['chromosome']:
            chrom_switches.append(i)
    
    if verbose:
        print(f"Number of tokens: {len(gene_tokens)}")
        print(f"Number of chromosome switch points: {len(chrom_switches)}")
        print("Example token:")
        print(gene_tokens[0])
        print("Chromosome switch points:")
        for i in chrom_switches:
            print(f"Switch at token {i}: {gene_tokens[i-1]['chromosome']} -> {gene_tokens[i]['chromosome']}")
        print('\n')
        
    return gene_tokens, chrom_switches

def plot_avg_attention(avg_attn, token_encoder_dim=None, use_chrom_labels=True):
    """Helper function to plot average attention weights"""
    nhead = avg_attn.shape[0]
    
    # Create subplot grid with n+1 plots (n heads + global average)
    fig, axes = plt.subplots(1, nhead+1, figsize=((nhead+1)*6, 5))
    
    # Get chromosome switch points based on token_encoder_dim
    if token_encoder_dim is not None:
        _, chrom_switches = create_gene_tokens(token_chunk_size=token_encoder_dim, verbose=False)
        # Convert chromosome numbers to readable format
        chrom_labels = []
        for i in range(len(chrom_switches)):
            chrom_num = i + 2  # First switch is to chromosome 2
            if chrom_num <= 23:
                chrom_labels.append(str(chrom_num))

    # Plot individual attention heads
    for h in range(nhead):
        vmin, vmax = avg_attn[h].min(), avg_attn[h].max()
        im = axes[h].imshow(avg_attn[h], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[h].set_title(f"Attention Head {h}")
        
        if token_encoder_dim is not None:
            # Add chromosome switch ticks
            axes[h].set_xticks(chrom_switches)
            axes[h].set_yticks(chrom_switches)
            
            axes[h].set_xlabel("TSS Key Position")
            axes[h].set_ylabel("TSS Query Position")
            
            if use_chrom_labels:
                axes[h].set_xticklabels(chrom_labels, rotation=45, fontsize=8)
                axes[h].set_yticklabels(chrom_labels, fontsize=8)
            else:
                axes[h].set_xticklabels(chrom_switches, rotation=45, fontsize=8)
                axes[h].set_yticklabels(chrom_switches, fontsize=8)

        plt.colorbar(im, ax=axes[h], label=f"Weight [{vmin:.2f}, {vmax:.2f}]")
    
    # Plot global average attention
    global_avg = avg_attn.mean(axis=0)
    vmin, vmax = global_avg.min(), global_avg.max()
    im = axes[-1].imshow(global_avg, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[-1].set_title("Global Average")
    
    if token_encoder_dim is not None:
        # Add chromosome switch ticks to global average plot
        axes[-1].set_xticks(chrom_switches)
        axes[-1].set_yticks(chrom_switches)
        if use_chrom_labels:
            axes[-1].set_xticklabels(chrom_labels, rotation=45, fontsize=8)
            axes[-1].set_yticklabels(chrom_labels, fontsize=8)
        else:
            axes[-1].set_xticklabels(chrom_switches, rotation=45, fontsize=8)
            axes[-1].set_yticklabels(chrom_switches, fontsize=8)
    else:
        axes[-1].set_xlabel("TSS Key Position")
        axes[-1].set_ylabel("TSS Query Position")
        
    plt.colorbar(im, ax=axes[-1], label=f"Weight [{vmin:.2f}, {vmax:.2f}]")
    
    plt.tight_layout()
    plt.show()

def collect_full_attention_heads(encoder_layers, save_attn_path=None):
    """Enable full attention head collection"""
    for layer in encoder_layers:
        layer.store_attn = True
    return None

def accumulate_attention_weights(encoder_layers, is_first_batch=False):
    """Accumulate attention weights during inference"""
    last_layer = encoder_layers[-1]
    attn_weights = last_layer.last_attn_weights # trained weights
    
    if attn_weights is not None:
        batch_avg = attn_weights.mean(dim=0)
        if is_first_batch or not hasattr(last_layer, '_accumulated_attn'):
            last_layer._accumulated_attn = batch_avg
        else:
            last_layer._accumulated_attn += batch_avg

def process_full_attention_heads(encoder_layers, total_batches, save_attn_path=None, token_encoder_dim=None):
    """Process collected full attention head weights"""
    # Extract last layer of transformer and accumulated average attention over all batches
    last_layer = encoder_layers[-1]
    avg_attn = getattr(last_layer, '_accumulated_attn', None)
    
    if avg_attn is not None and total_batches > 0:
        avg_attn = avg_attn / total_batches
        plot_avg_attention(avg_attn.cpu(), token_encoder_dim=token_encoder_dim)
        
        if save_attn_path is not None:
            np.save('./models/saved_models/saved_heads/'+save_attn_path, avg_attn.cpu().numpy())
    
    # Clean up
    for layer in encoder_layers:
        layer.store_attn = False
        if hasattr(layer, '_accumulated_attn'):
            delattr(layer, '_accumulated_attn')
    
    return avg_attn

def collect_attention_pooling_weights(all_attn, save_attn_path=None):
    """Process and save attention pooling weights"""
    all_attn_list = []
    
    for attn_i, attn_j in all_attn:
        all_attn_list.append(attn_i.cpu().numpy())
        all_attn_list.append(attn_j.cpu().numpy())
    
    all_attn_matrix = np.concatenate(all_attn_list, axis=0)
    avg_attn_arr = np.mean(all_attn_matrix, axis=0)
    
    if save_attn_path is not None:
        np.save(save_attn_path, avg_attn_arr)
    
    return avg_attn_arr

# SPECIFIC ATTENTION PLOTTING MODALITIES
def plot_global_head_attention(avg_attn_dict, use_chrom_labels=False, token_encoder_dim=60, cls_token=False):
    """Plot global average attention for each head across all subnetworks"""
    # Get dimensions from first attention tensor
    first_attn = next(iter(avg_attn_dict.values()))
    n_heads = first_attn.shape[0]
    
    # Average attention across all subnetworks
    avg_attn = np.zeros_like(first_attn)
    attn_arrays = [attn for attn in avg_attn_dict.values()]
    avg_attn = np.mean(np.stack(attn_arrays, axis=0), axis=0)
    
    # Create subplot grid with n+1 plots (n heads + global average)
    fig, axes = plt.subplots(1, n_heads+1, figsize=((n_heads+1)*6, 5))
    
    # Get chromosome switch points
    _, chrom_switches = create_gene_tokens(token_chunk_size=token_encoder_dim, verbose=False)
    # Shift switch points if using cls token
    if cls_token:
        chrom_switches = [x + 1 for x in chrom_switches]
    
    # Convert chromosome numbers to readable format
    chrom_labels = []
    for i in range(len(chrom_switches)):
        chrom_num = i + 2  # First switch is to chromosome 2
        if chrom_num <= 23:
            chrom_labels.append(str(chrom_num))
            
    # Plot individual attention heads
    for h in range(n_heads):
        vmin, vmax = avg_attn[h].min(), avg_attn[h].max()
        im = axes[h].imshow(avg_attn[h], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[h].set_title(f"Attention Head {h}")
        
        # Add chromosome switch ticks
        if cls_token:
            axes[h].set_xticks([0] + chrom_switches)
            axes[h].set_yticks([0] + chrom_switches)
        else:
            axes[h].set_xticks(chrom_switches)
            axes[h].set_yticks(chrom_switches)
        
        axes[h].set_xlabel("TSS Key Position") 
        axes[h].set_ylabel("TSS Query Position")
        
        if use_chrom_labels:
            if cls_token:
                axes[h].set_xticklabels(['CLS'] + chrom_labels, rotation=45, fontsize=8)
                axes[h].set_yticklabels(['CLS'] + chrom_labels, fontsize=8)
            else:
                axes[h].set_xticklabels(chrom_labels, rotation=45, fontsize=8)
                axes[h].set_yticklabels(chrom_labels, fontsize=8)
        else:
            if cls_token:
                axes[h].set_xticklabels(['CLS'] + [str(x) for x in chrom_switches], rotation=45, fontsize=8)
                axes[h].set_yticklabels(['CLS'] + [str(x) for x in chrom_switches], fontsize=8)
            else:
                axes[h].set_xticklabels(chrom_switches, rotation=45, fontsize=8)
                axes[h].set_yticklabels(chrom_switches, fontsize=8)

        plt.colorbar(im, ax=axes[h], label=f"Weight [{vmin:.2f}, {vmax:.2f}]")
    
    # Plot global average attention
    global_avg = avg_attn.mean(axis=0)
    vmin, vmax = global_avg.min(), global_avg.max()
    im = axes[-1].imshow(global_avg, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[-1].set_title("Global Average")
    
    # Add chromosome switch ticks to global average plot
    if cls_token:
        axes[-1].set_xticks([0] + chrom_switches)
        axes[-1].set_yticks([0] + chrom_switches)
    else:
        axes[-1].set_xticks(chrom_switches)
        axes[-1].set_yticks(chrom_switches)
        
    if use_chrom_labels:
        if cls_token:
            axes[-1].set_xticklabels(['CLS'] + chrom_labels, rotation=45, fontsize=8)
            axes[-1].set_yticklabels(['CLS'] + chrom_labels, fontsize=8)
        else:
            axes[-1].set_xticklabels(chrom_labels, rotation=45, fontsize=8)
            axes[-1].set_yticklabels(chrom_labels, fontsize=8)
    else:
        if cls_token:
            axes[-1].set_xticklabels(['CLS'] + [str(x) for x in chrom_switches], rotation=45, fontsize=8)
            axes[-1].set_yticklabels(['CLS'] + [str(x) for x in chrom_switches], fontsize=8)
        else:
            axes[-1].set_xticklabels(chrom_switches, rotation=45, fontsize=8)
            axes[-1].set_yticklabels(chrom_switches, fontsize=8)
    
    axes[-1].set_xlabel("TSS Key Position")
    axes[-1].set_ylabel("TSS Query Position")
        
    plt.colorbar(im, ax=axes[-1], label=f"Weight [{vmin:.2f}, {vmax:.2f}]")
    
    plt.tight_layout()
    plt.show()

def plot_subnetwork_token_attention(avg_attn_dict, use_chrom_labels=False, token_encoder_dim=60, cls_token=False):
    """Plot per-token attention for each subnetwork"""
    # Get dimensions
    n_networks = len(avg_attn_dict)
    n_tokens = next(iter(avg_attn_dict.values())).shape[1]
    
    # Initialize matrix for token attention vectors
    token_attention = np.zeros((n_networks, n_tokens))
    
    # Compute average attention per token for each subnetwork
    for i, (network_name, attn) in enumerate(avg_attn_dict.items()):
        # Average across heads then sum columns
        token_attention[i] = attn.mean(axis=0).sum(axis=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot heatmap
    vmin, vmax = token_attention.min(), token_attention.max()
    im = ax.imshow(token_attention, cmap="viridis", aspect='auto', vmin=vmin, vmax=vmax)
    
    # Add network labels on y-axis
    ax.set_yticks(range(n_networks))
    ax.set_yticklabels(list(avg_attn_dict.keys()))
    
    # Add chromosome markers on x-axis
    _, chrom_switches = create_gene_tokens(token_chunk_size=token_encoder_dim, verbose=False)
    # Shift switch points if using cls token
    if cls_token:
        chrom_switches = [x + 1 for x in chrom_switches]
        
    # Convert chromosome numbers to readable format
    chrom_labels = []
    for i in range(len(chrom_switches)):
        chrom_num = i + 2  # First switch is to chromosome 2
        if chrom_num <= 23:
            chrom_labels.append(str(chrom_num))
            
    if cls_token:
        ax.set_xticks([0] + chrom_switches)
        if use_chrom_labels:
            ax.set_xticklabels(['CLS'] + chrom_labels, rotation=45, fontsize=14)
            ax.set_xlabel("Chromosome")
        else:
            ax.set_xticklabels(['CLS'] + [str(x) for x in chrom_switches], rotation=45, fontsize=14)
            ax.set_xlabel("Token Position")
    else:
        ax.set_xticks(chrom_switches)
        if use_chrom_labels:
            ax.set_xticklabels(chrom_labels, rotation=45, fontsize=14)
            ax.set_xlabel("Chromosome")
        else:
            ax.set_xticklabels(chrom_switches, rotation=45, fontsize=14)
            ax.set_xlabel("Token Position")
    
    ax.set_title("Average Attention per Token Across Subnetworks")
    
    # Add colorbar
    plt.colorbar(im, label=f"Relative Attention [{vmin:.2f}, {vmax:.2f}]")
    
    plt.tight_layout()
    plt.show()
    
    # Create correlation matrix between subnetworks
    network_names = list(avg_attn_dict.keys())
    n_networks = len(network_names)
    corr_matrix = np.zeros((n_networks, n_networks))
    
    for i in range(n_networks):
        for j in range(n_networks):
            corr, _ = stats.spearmanr(token_attention[i], token_attention[j])
            corr_matrix[i,j] = corr
    
    # Create figure for correlation matrix
    plt.figure(figsize=(8,7))
    
    # Plot heatmap
    im = plt.imshow(corr_matrix, 
                    cmap='RdBu_r',
                    vmin=-1, vmax=1,
                    aspect='equal')
    
    # Add colorbar
    plt.colorbar(im)
    
    # Add text annotations
    for i in range(n_networks):
        for j in range(n_networks):
            text = f'{corr_matrix[i,j]:.2f}'.lstrip('0')
            plt.text(j, i, text,
                    ha='center', va='center', 
                    color='black' if abs(corr_matrix[i,j]) < 0.5 else 'white',
                    fontsize=10)
    
    # Set ticks and labels
    plt.xticks(range(n_networks), network_names, rotation=45, ha='right')
    plt.yticks(range(n_networks), network_names)
    
    plt.title('Spearman Correlation between Network Attention Patterns', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    return token_attention

def plot_subnetwork_expression_correlation(network_expr, use_chrom_labels=False, token_encoder_dim=60):
    """Plot per-token expression for each subnetwork and their correlations"""
    # Get dimensions
    n_networks = len(network_expr)
    n_genes = next(iter(network_expr.values())).shape[1]
    
    # Calculate number of tokens based on gene chunks
    n_tokens = (n_genes + token_encoder_dim - 1) // token_encoder_dim  # Ceiling division
    
    # Initialize matrix for binned expression values
    token_expression = np.zeros((n_networks, n_tokens))
    
    # Compute average expression per token for each subnetwork
    for i, (network_name, expr) in enumerate(network_expr.items()):
        # Average expression across nodes in network
        network_avg = expr.mean(axis=0)
        
        # Bin the genes into token-sized chunks
        for j in range(n_tokens):
            start_idx = j * token_encoder_dim
            end_idx = min(start_idx + token_encoder_dim, n_genes)
            token_expression[i,j] = network_avg[start_idx:end_idx].mean()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot heatmap
    vmin, vmax = token_expression.min(), token_expression.max()
    im = ax.imshow(token_expression, cmap="viridis", aspect='auto', vmin=vmin, vmax=vmax)
    
    # Add network labels on y-axis
    ax.set_yticks(range(n_networks))
    ax.set_yticklabels(list(network_expr.keys()))
    
    # Add chromosome markers on x-axis
    _, chrom_switches = create_gene_tokens(token_chunk_size=token_encoder_dim, verbose=False)
    # Convert chromosome numbers to readable format
    chrom_labels = []
    for i in range(len(chrom_switches)):
        chrom_num = i + 2  # First switch is to chromosome 2
        if chrom_num <= 23:
            chrom_labels.append(str(chrom_num))
            
    ax.set_xticks(chrom_switches)
    if use_chrom_labels:
        ax.set_xticklabels(chrom_labels, rotation=45, fontsize=14)
        ax.set_xlabel("Chromosome")
    else:
        ax.set_xticklabels(chrom_switches, rotation=45, fontsize=14)
        ax.set_xlabel("Token Position")
    
    ax.set_title("Average Expression per Token Across Subnetworks")
    
    # Add colorbar
    plt.colorbar(im, label=f"Average Expression [{vmin:.2f}, {vmax:.2f}]")
    
    plt.tight_layout()
    plt.show()
    
    # Create correlation matrix between subnetworks
    network_names = list(network_expr.keys())
    n_networks = len(network_names)
    corr_matrix = np.zeros((n_networks, n_networks))
    
    for i in range(n_networks):
        for j in range(n_networks):
            corr, _ = stats.spearmanr(token_expression[i], token_expression[j])
            corr_matrix[i,j] = corr
    
    # Create figure for correlation matrix
    plt.figure(figsize=(8,7))
    
    # Plot heatmap
    im = plt.imshow(corr_matrix, 
                    cmap='RdBu_r',
                    vmin=-1, vmax=1,
                    aspect='equal')
    
    # Add colorbar
    plt.colorbar(im)
    
    # Add text annotations
    for i in range(n_networks):
        for j in range(n_networks):
            text = f'{corr_matrix[i,j]:.2f}'.lstrip('0')
            plt.text(j, i, text,
                    ha='center', va='center', 
                    color='black' if abs(corr_matrix[i,j]) < 0.5 else 'white',
                    fontsize=10)
    
    # Set ticks and labels
    plt.xticks(range(n_networks), network_names, rotation=45, ha='right')
    plt.yticks(range(n_networks), network_names)
    
    plt.title('Spearman Correlation between Network Expression Patterns', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    return token_expression