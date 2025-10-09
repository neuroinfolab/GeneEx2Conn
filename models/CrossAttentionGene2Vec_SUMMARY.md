# CrossAttentionGene2VecModel - Implementation Summary

## Overview

A novel transformer-based model for predicting brain region connectivity using single-gene resolution with Gene2Vec embeddings and cross-attention mechanism.

## Key Features

1. **Single Gene Resolution**: Each gene is treated as an individual token
2. **Gene2Vec Embeddings**: 200D semantic gene representations
3. **Expression Value Binning**: Discretizes continuous expression values into learned bins
4. **Cross-Attention Architecture**: CLS token + region i genes attend to region j genes
5. **FlashAttention**: Efficient attention computation

## Architecture Details

### Input Processing

For each brain region pair (i, j):

1. **Gene Expression Encoding** (per region):
   - Filter to genes with Gene2Vec embeddings (~shared subset of valid genes)
   - Bin expression values into discrete categories (default: 5 bins in [0,1])
   - One-hot encode bin indices
   - Project one-hot bins to 200D via learned linear layer
   - Element-wise addition: `gene2vec_embedding + bin_embedding`
   - Project combined 200D embedding to `d_model` space (default: 128D)

2. **CLS Token Initialization**:
   - **Random mode**: 6 random values projected to `d_model`
   - **Spatial mode**: Concatenate coords of regions i and j (6 values), normalize, then project to `d_model`

3. **Cross-Attention**:
   ```
   Queries = [CLS_token, gene_tokens_region_i]  # Shape: (B, num_genes + 1, d_model)
   Keys/Values = gene_tokens_region_j           # Shape: (B, num_genes, d_model)
   
   → Multi-layer cross-attention (FlashAttention)
   → Extract CLS token output: queries[:, 0, :]  # Shape: (B, d_model)
   ```

4. **Prediction Head**:
   - CLS token embedding → MLP (configurable hidden dims) → scalar connectivity prediction

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `expression_bins` | 5 | Number of discrete expression bins |
| `d_model` | 128 | Transformer embedding dimension |
| `nhead` | 4 | Number of attention heads |
| `num_layers` | 4 | Number of cross-attention layers |
| `deep_hidden_dims` | [512, 256, 128] | MLP head architecture |
| `cls_init` | 'random' | CLS initialization: 'random' or 'spatial' |
| `pooling_mode` | 'mean' | Pooling method: 'mean', 'attention', or 'cls' |
| `use_alibi` | False | Use ALiBi positional biases |
| `transformer_dropout` | 0.1 | Dropout in attention blocks |
| `dropout_rate` | 0.1 | Dropout in MLP head |
| `learning_rate` | 0.001 | Optimizer learning rate |
| `batch_size` | 512 | Training batch size |
| `cosine_lr` | False | Use cosine learning rate schedule |

## Usage Example

```python
from models.smt_singlegene import CrossAttentionGene2VecModel
from data.data_utils import RegionPairDataset

# Create dataset
dataset = RegionPairDataset(
    X=gene_expression,           # (num_regions, num_genes)
    Y=connectivity_matrix,       # (num_regions, num_regions)
    coords=coordinates,          # (num_regions, 3)
    valid2true_mapping=mapping,
    dataset='UKBB',
    parcellation='S456',
    valid_genes=valid_genes
)

# Initialize model
model = CrossAttentionGene2VecModel(
    input_dim=gene_expression.shape[1] * 2,  # 2 * num_genes
    region_pair_dataset=dataset,
    expression_bins=5,
    d_model=128,
    nhead=4,
    num_layers=4,
    deep_hidden_dims=[512, 256, 128],
    cls_init='spatial',  # or 'random'
    learning_rate=0.001,
    batch_size=512,
    epochs=100
)

# Train model
history = model.fit(
    dataset=dataset,
    train_indices=train_indices,
    test_indices=test_indices,
    save_model='path/to/save/model.pt'
)

# Make predictions
predictions, targets = model.predict(test_loader)
```

## Model Architecture Diagram

```
Region i genes:        Region j genes:
[g1, g2, ..., gN]     [g1, g2, ..., gN]
       ↓                      ↓
  Gene2Vec + Bins        Gene2Vec + Bins
       ↓                      ↓
  Project to d_model    Project to d_model
       ↓                      ↓
  
  CLS Token (random/spatial init)
       ↓
  [CLS, g1_i, g2_i, ..., gN_i]  →  Cross-Attention  ←  [g1_j, g2_j, ..., gN_j]
                                   (Queries)              (Keys/Values)
       ↓
  Extract CLS token (position 0)
       ↓
  MLP Head [512 → 256 → 128 → 1]
       ↓
  Connectivity Prediction
```

## Key Implementation Files

- **models/smt_singlegene.py**: Main model implementation
  - `CrossAttentionBlock`: Cross-attention with FlashAttention
  - `CrossAttentionGene2VecEncoder`: Gene encoding and cross-attention
  - `CrossAttentionGene2VecModel`: Main model class
  
- **models/test_smt_singlegene.py**: Testing/validation script

## Design Considerations

1. **Gene2Vec Integration**: Only genes with pre-trained Gene2Vec embeddings are used (typically ~600-700 genes from valid gene set)

2. **Expression Binning**: Provides discrete representation of continuous expression values, creating an effective vocabulary size of `num_genes × expression_bins`

3. **CLS Token Design**: 
   - Acts as a learned aggregator of cross-region information
   - Can incorporate spatial information via coordinate-based initialization
   - Participates in attention alongside gene tokens from region i

4. **Cross-Attention Choice**: Asymmetric attention allows the model to learn how region i's expression pattern (queries) relates to region j's pattern (keys/values) in the context of their connectivity

5. **ALiBi Support**: Optional positional biases useful when genes are sorted (e.g., by chromosome position or co-expression patterns)

## Comparison to Other Models

| Model | Token Type | Attention | Pooling |
|-------|-----------|-----------|---------|
| SharedSelfAttentionModel | Gene chunks | Self (per region) | Flatten |
| SharedSelfAttentionPoolingModel | Gene chunks | Self (per region) | Attention pooling |
| SharedSelfAttentionGene2VecModel | Single genes | Self (per region) | Attention pooling |
| **CrossAttentionGene2VecModel** | Single genes | **Cross (between regions)** | **CLS token** |

## Future Extensions

1. **Bidirectional cross-attention**: Let region j also attend to region i
2. **Multiple CLS tokens**: Separate CLS tokens for different aspects (spatial, functional, etc.)
3. **Dynamic binning**: Learn bin boundaries rather than using fixed equal-width bins
4. **Attention visualization**: Extract and analyze cross-attention patterns

