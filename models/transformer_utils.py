import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

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
