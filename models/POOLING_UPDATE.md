# Pooling Options Update - CrossAttentionGene2VecModel

## Summary of Changes

Added flexible pooling options to the `CrossAttentionGene2VecModel` instead of only using the CLS token directly. Users can now choose between three pooling strategies to create the final edge representation from the transformer output.

## Pooling Modes

### 1. Mean Pooling (DEFAULT)
```python
pooling_mode='mean'
```
- **How it works**: Simple average over all tokens (CLS + gene tokens)
- **Formula**: `output = queries.mean(dim=1)`
- **Pros**: 
  - Fast and simple
  - No additional parameters
  - Utilizes all token information equally
- **Cons**: 
  - Equal weight to all tokens (no learned importance)
- **When to use**: Default choice, works well in most cases

### 2. Attention Pooling
```python
pooling_mode='attention'
```
- **How it works**: Learned attention weights to pool tokens
- **Implementation**: Uses `AttentionPooling` layer from `smt_utils.py`
- **Pros**:
  - Learns which tokens are most important
  - More expressive than mean pooling
  - Adaptive weighting based on context
- **Cons**:
  - Additional parameters (~few thousand)
  - Slightly slower than mean pooling
- **When to use**: When you want the model to learn token importance

### 3. CLS-only
```python
pooling_mode='cls'
```
- **How it works**: Uses only the CLS token (position 0)
- **Formula**: `output = queries[:, 0, :]`
- **Pros**:
  - Aligns with original BERT/ViT design
  - CLS acts as learned aggregator
- **Cons**:
  - Discards explicit gene token information (relies on residual enrichment)
- **When to use**: For comparison with standard transformer architectures

## Usage Examples

### Basic Usage
```python
from models.smt_singlegene import CrossAttentionGene2VecModel

# Default: mean pooling
model = CrossAttentionGene2VecModel(
    input_dim=input_dim,
    region_pair_dataset=dataset,
    pooling_mode='mean'  # DEFAULT
)

# Attention pooling
model = CrossAttentionGene2VecModel(
    input_dim=input_dim,
    region_pair_dataset=dataset,
    pooling_mode='attention'
)

# CLS-only (original behavior)
model = CrossAttentionGene2VecModel(
    input_dim=input_dim,
    region_pair_dataset=dataset,
    pooling_mode='cls'
)
```

### Comparing Pooling Strategies
```python
# Train with different pooling modes
results = {}

for pooling in ['mean', 'attention', 'cls']:
    model = CrossAttentionGene2VecModel(
        input_dim=input_dim,
        region_pair_dataset=dataset,
        pooling_mode=pooling,
        d_model=128,
        nhead=4,
        num_layers=4
    )
    
    history = model.fit(dataset, train_indices, test_indices)
    results[pooling] = history
    
# Compare performance
for pooling, hist in results.items():
    print(f"{pooling}: Test R = {hist['test_corr'][-1]:.4f}")
```

## Architecture Changes

### Before (CLS-only)
```
Cross-Attention Output: (B, num_shared_genes + 1, d_model)
         ↓
Extract CLS[0]: (B, d_model)
         ↓
MLP → Prediction
```

### After (Flexible Pooling)
```
Cross-Attention Output: (B, num_shared_genes + 1, d_model)
         ↓
Apply Pooling:
  - mean: Average all tokens
  - attention: Weighted sum with learned weights
  - cls: Extract position 0
         ↓
Pooled Output: (B, d_model)
         ↓
MLP → Prediction
```

## Performance Considerations

| Pooling Mode | Speed | Parameters | Expressiveness |
|--------------|-------|------------|----------------|
| mean | Fastest | 0 additional | Medium |
| attention | Medium | ~few thousand | High |
| cls | Fastest | 0 additional | Medium |

## Implementation Details

### Code Changes

1. **Added `pooling_mode` parameter** to `CrossAttentionGene2VecEncoder`:
   ```python
   def __init__(self, ..., pooling_mode='mean', ...):
       self.pooling_mode = pooling_mode
       if pooling_mode == 'attention':
           self.attention_pooling = AttentionPooling(d_model, hidden_dim=32)
   ```

2. **Modified forward pass**:
   ```python
   def forward(self, gene_expr_i, gene_expr_j, coords_i, coords_j):
       # ... cross-attention ...
       
       if self.pooling_mode == 'cls':
           output = queries[:, 0, :]
       elif self.pooling_mode == 'mean':
           output = queries.mean(dim=1)
       elif self.pooling_mode == 'attention':
           output = self.attention_pooling(queries)
       
       return output
   ```

3. **Updated model class** to expose parameter:
   ```python
   class CrossAttentionGene2VecModel(BaseTransformerModel):
       def __init__(self, ..., pooling_mode='mean', ...):
           # ...
           self.encoder = CrossAttentionGene2VecEncoder(
               ...,
               pooling_mode=pooling_mode,
               ...
           )
   ```

## Testing

Updated test script to verify all three pooling modes:

```python
configs = [
    {
        'name': 'Mean pooling',
        'pooling_mode': 'mean',
        ...
    },
    {
        'name': 'Attention pooling',
        'pooling_mode': 'attention',
        ...
    },
    {
        'name': 'CLS-only pooling',
        'pooling_mode': 'cls',
        ...
    }
]
```

## Recommendations

1. **Start with 'mean'** (default): Fast, no extra parameters, works well
2. **Try 'attention'**: If you have sufficient data and want more expressiveness
3. **Use 'cls'**: For comparison with standard transformer architectures or ablation studies

## Backward Compatibility

- Default is `pooling_mode='mean'` (not 'cls')
- This is a slight behavior change from the original implementation
- To get the original behavior, explicitly set `pooling_mode='cls'`

## Files Updated

1. `models/smt_singlegene.py` - Core implementation
2. `models/test_smt_singlegene.py` - Added pooling mode tests
3. `models/CrossAttentionGene2Vec_SUMMARY.md` - Added pooling_mode to hyperparameters
4. `models/CrossAttentionGene2Vec_Architecture.md` - Updated architecture diagrams
5. `models/CrossAttentionGene2Vec_QuickStart.md` - Added pooling examples and guidance
6. `models/POOLING_UPDATE.md` - This file

## Date
2025-10-08



