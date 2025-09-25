import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
from torch.cuda.amp import autocast
import numpy as np
import matplotlib.pyplot as plt
import math

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

def plot_avg_attention(avg_attn):
    """Helper function to plot average attention weights"""
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

def collect_full_attention_heads(encoder_layers, save_attn_path=None):
    """Enable full attention head collection"""
    for layer in encoder_layers:
        layer.store_attn = True
    return None

def process_full_attention_heads(encoder_layers, total_batches, save_attn_path=None):
    """Process collected full attention head weights"""
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
    """Accumulate attention weights during inference"""
    last_layer = encoder_layers[-1]
    attn_weights = last_layer.last_attn_weights
    
    if attn_weights is not None:
        batch_avg = attn_weights.mean(dim=0)
        
        if is_first_batch or not hasattr(last_layer, '_accumulated_attn'):
            last_layer._accumulated_attn = batch_avg
        else:
            last_layer._accumulated_attn += batch_avg
