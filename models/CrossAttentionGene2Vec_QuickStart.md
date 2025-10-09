# CrossAttentionGene2VecModel - Quick Start Guide

## Basic Usage

```python
from models.smt_singlegene import CrossAttentionGene2VecModel
from data.data_load import load_transcriptome
from data.data_utils import RegionPairDataset
import numpy as np

# 1. Load data
X, coords, valid2true_mapping, valid_genes = load_transcriptome(
    parcellation='S456',
    gene_list='0.2',
    dataset='AHBA',
    return_valid_genes=True,
    return_coords=True,
    return_mapping=True,
    sort_genes='refgenome'  # or 'coexpression' for ALiBi
)

# Load or create connectivity matrix Y
Y = np.load('path/to/connectivity_matrix.npy')  # (num_regions, num_regions)

# 2. Create dataset
dataset = RegionPairDataset(
    X=X, 
    Y=Y, 
    coords=coords,
    valid2true_mapping=valid2true_mapping,
    dataset='UKBB',  # or 'HCP', 'BHA2'
    parcellation='S456',
    valid_genes=valid_genes
)

# 3. Split data
from sklearn.model_selection import train_test_split
train_indices, test_indices = train_test_split(
    np.arange(len(dataset)), 
    test_size=0.2, 
    random_state=42
)

# 4. Initialize model
model = CrossAttentionGene2VecModel(
    input_dim=X.shape[1] * 2,
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

# 5. Train
history = model.fit(
    dataset=dataset,
    train_indices=train_indices,
    test_indices=test_indices,
    save_model='./saved_models/cross_attn_g2v.pt'
)

# 6. Evaluate
from torch.utils.data import DataLoader, Subset
test_dataset = Subset(dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

predictions, targets = model.predict(test_loader)

from scipy.stats import pearsonr
corr, pval = pearsonr(predictions.flatten(), targets.flatten())
print(f"Test Correlation: {corr:.4f} (p={pval:.2e})")
```

## Configuration Presets

### Lightweight (Fast Training)
```python
config_lightweight = {
    'expression_bins': 3,
    'd_model': 64,
    'nhead': 2,
    'num_layers': 2,
    'deep_hidden_dims': [256, 128],
    'pooling_mode': 'mean',
    'batch_size': 1024,
    'learning_rate': 0.001
}
```

### Standard (Balanced)
```python
config_standard = {
    'expression_bins': 5,
    'd_model': 128,
    'nhead': 4,
    'num_layers': 4,
    'deep_hidden_dims': [512, 256, 128],
    'pooling_mode': 'mean',
    'batch_size': 512,
    'learning_rate': 0.001
}
```

### Heavy (Maximum Capacity)
```python
config_heavy = {
    'expression_bins': 10,
    'd_model': 256,
    'nhead': 8,
    'num_layers': 6,
    'deep_hidden_dims': [1024, 512, 256, 128],
    'pooling_mode': 'attention',  # Use attention pooling for heavy model
    'batch_size': 256,
    'learning_rate': 0.0005
}
```

### With Cosine LR Schedule
```python
config_cosine = {
    'expression_bins': 5,
    'd_model': 128,
    'nhead': 4,
    'num_layers': 4,
    'deep_hidden_dims': [512, 256, 128],
    'batch_size': 512,
    'learning_rate': 0.001,
    'cosine_lr': True  # Enable cosine learning rate
}
```

## Hyperparameter Tuning Guide

### Expression Bins
- **3 bins**: Coarse discretization (low/medium/high)
- **5 bins**: Standard, good balance
- **10 bins**: Fine-grained, more parameters
- **Trade-off**: More bins = more capacity but also more parameters to learn

### d_model (Embedding Dimension)
- **64**: Fast, less capacity
- **128**: Standard
- **256**: High capacity, slower
- **Rule**: Should be divisible by nhead

### Number of Heads
- **2**: Minimal
- **4**: Standard
- **8**: High capacity
- **Rule**: d_model % nhead == 0

### Number of Layers
- **2**: Fast, shallow interactions
- **4**: Standard
- **6**: Deep, complex interactions
- **Trade-off**: More layers = better modeling but slower training

### CLS Initialization
- **'random'**: Fully learnable, no prior assumptions
- **'spatial'**: Incorporates coordinate information
- **Recommendation**: Try both, spatial often works better for brain connectivity

### Pooling Mode
- **'mean'**: Simple average over all tokens (CLS + genes) - **DEFAULT**
- **'attention'**: Learned attention weights to pool tokens
- **'cls'**: Use only CLS token (original design)
- **Recommendation**: Start with 'mean' (fastest), try 'attention' for more capacity

## Common Issues and Solutions

### Issue: Out of Memory
**Solution**:
```python
# Reduce batch size
model = CrossAttentionGene2VecModel(..., batch_size=128)

# Or reduce model size
model = CrossAttentionGene2VecModel(
    d_model=64,
    num_layers=2,
    deep_hidden_dims=[256, 128]
)
```

### Issue: Slow Training
**Solution**:
```python
# Reduce number of workers if I/O bound
model = CrossAttentionGene2VecModel(..., num_workers=1)

# Use smaller model
model = CrossAttentionGene2VecModel(
    expression_bins=3,
    d_model=64,
    num_layers=2
)
```

### Issue: Underfitting (Low Training Performance)
**Solution**:
```python
# Increase model capacity
model = CrossAttentionGene2VecModel(
    expression_bins=10,
    d_model=256,
    nhead=8,
    num_layers=6,
    deep_hidden_dims=[1024, 512, 256, 128]
)

# Decrease regularization
model = CrossAttentionGene2VecModel(
    transformer_dropout=0.05,
    dropout_rate=0.05,
    weight_decay=0.0
)
```

### Issue: Overfitting (Train-Test Gap)
**Solution**:
```python
# Increase regularization
model = CrossAttentionGene2VecModel(
    transformer_dropout=0.2,
    dropout_rate=0.2,
    weight_decay=0.01
)

# Use data augmentation (if available in dataset)
model = CrossAttentionGene2VecModel(..., aug_prob=0.3)

# Reduce model capacity
model = CrossAttentionGene2VecModel(
    d_model=64,
    num_layers=2
)
```

## Integration with Existing Pipelines

### Using with Different Gene Lists
```python
# Ensure Gene2Vec compatibility
# Model automatically filters to genes with Gene2Vec embeddings
model = CrossAttentionGene2VecModel(
    input_dim=X.shape[1] * 2,
    region_pair_dataset=dataset,
    # ... other params
)
# Check which genes are used:
print(f"Using {len(model.encoder.shared_gene_indices)} genes")
print(f"Gene overlap: {len(model.encoder.shared_genes)}")
```

### Using with Different Parcellations
```python
# Works with any parcellation, just update dataset creation
dataset = RegionPairDataset(
    X=X, Y=Y, coords=coords,
    valid2true_mapping=valid2true_mapping,
    dataset='UKBB',
    parcellation='S200',  # or 'S400', 'DK', etc.
    valid_genes=valid_genes
)
```

### Saving and Loading Models
```python
# Save during training
history = model.fit(
    dataset=dataset,
    train_indices=train_indices,
    test_indices=test_indices,
    save_model='./models/saved_models/my_model.pt'
)

# Load for inference
import torch
model.load_state_dict(torch.load('./models/saved_models/my_model.pt'))
model.eval()
```

## Comparison to Other Models

| Model | When to Use |
|-------|-------------|
| `SharedSelfAttentionModel` | Need chunk-based tokens, faster training |
| `SharedSelfAttentionPoolingModel` | Want attention pooling, similar speed to above |
| `SharedSelfAttentionGene2VecModel` | Want Gene2Vec but self-attention |
| **`CrossAttentionGene2VecModel`** | **Want explicit cross-region interaction, have Gene2Vec embeddings** |
| `SharedSelfAttentionCelltypeModel` | Have cell type data |
| `SharedSelfAttentionPCAModel` | Want dimensionality reduction first |

## Advanced: Custom Expression Binning

```python
# The default uses equal-width bins in [0, 1]
# To customize, you would need to modify bin_expression() method

# Example: Log-scale binning (requires model modification)
# Or use quantile-based bins (requires pre-processing)

import numpy as np

# Quantile-based preprocessing
def create_quantile_bins(X, n_bins=5):
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(X, quantiles)
    return bin_edges

# This would require modifying the model's bin_expression method
# to use custom bin edges instead of equal-width
```

## Performance Benchmarks

Approximate training speeds (on V100 GPU):

| Configuration | Params | Speed (samples/sec) | Memory |
|---------------|--------|---------------------|--------|
| Lightweight | ~500K | ~8000 | ~4GB |
| Standard | ~2M | ~4000 | ~8GB |
| Heavy | ~8M | ~1500 | ~16GB |

*Note: Actual performance depends on hardware, data size, and num_workers*

