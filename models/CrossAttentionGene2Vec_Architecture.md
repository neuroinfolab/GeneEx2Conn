# CrossAttentionGene2VecModel - Detailed Architecture

## Data Flow

### Step 1: Gene Embedding Creation

For each region (i and j independently):

```
Input: gene_expression (B, num_genes)
         ↓
Subset to Gene2Vec genes: (B, num_shared_genes)
         ↓
┌─────────────────────────────────────────────────┐
│ For each gene:                                  │
│                                                 │
│  Expression Value (scalar)                      │
│         ↓                                       │
│  Bin to [0, expression_bins-1]                  │
│         ↓                                       │
│  One-hot encode → (expression_bins,)            │
│         ↓                                       │
│  Linear projection → (200,)                     │
│         ↓                                       │
│  bin_embedding (200,)                           │
│         +                                       │
│  gene2vec_embedding (200,)  [fixed, loaded]     │
│         ↓                                       │
│  combined_embedding (200,)                      │
│         ↓                                       │
│  Linear projection → (d_model,)                 │
└─────────────────────────────────────────────────┘
         ↓
Output: gene_embeddings (B, num_shared_genes, d_model)
```

### Step 2: CLS Token Creation

```
Option A - Random Initialization:
    random_vector (6,) [learnable parameter]
         ↓
    Linear projection
         ↓
    CLS token (1, d_model)

Option B - Spatial Initialization:
    coords_i (3,) + coords_j (3,) → combined (6,)
         ↓
    Normalize (zero mean, unit std)
         ↓
    Scale to match random init (× 0.02)
         ↓
    Linear projection
         ↓
    CLS token (1, d_model)
```

### Step 3: Cross-Attention Setup

```
Queries Side (Region i + CLS):
    CLS token: (B, 1, d_model)
    Region i embeddings: (B, num_shared_genes, d_model)
         ↓
    Concatenate along sequence dimension
         ↓
    queries: (B, num_shared_genes + 1, d_model)
                    ↑
                Position 0 = CLS token
                Positions 1 to num_shared_genes = gene tokens

Keys/Values Side (Region j):
    Region j embeddings: (B, num_shared_genes, d_model)
         ↓
    keys_values: (B, num_shared_genes, d_model)
```

### Step 4: Cross-Attention Mechanism

```
For each attention layer:

    Queries: (B, num_shared_genes + 1, d_model)
         ↓
    Q_proj → Q: (B, num_shared_genes + 1, d_model)
    
    Keys/Values: (B, num_shared_genes, d_model)
         ↓
    KV_proj → K, V: each (B, num_shared_genes, d_model)
    
    Split into heads:
         Q: (B, nhead, num_shared_genes + 1, head_dim)
         K: (B, nhead, num_shared_genes, head_dim)
         V: (B, nhead, num_shared_genes, head_dim)
    
    FlashAttention:
         Attention(Q, K, V) → output: (B, num_shared_genes + 1, d_model)
    
    Residual + LayerNorm → FFN → Residual + LayerNorm
         ↓
    Updated queries: (B, num_shared_genes + 1, d_model)

Repeat for num_layers
```

### Step 5: Pooling

```
After all cross-attention layers:
    
    queries: (B, num_shared_genes + 1, d_model)
         ↓
    Apply pooling based on pooling_mode:
    
    Option A - Mean Pooling (DEFAULT):
        output = queries.mean(dim=1) → (B, d_model)
    
    Option B - Attention Pooling:
        output = AttentionPooling(queries) → (B, d_model)
        (Learned attention weights over all tokens)
    
    Option C - CLS-only:
        output = queries[:, 0, :] → (B, d_model)
        (Use only first token)
```

### Step 6: Connectivity Prediction

```
CLS output: (B, d_model)
         ↓
┌─────────────────────────────────┐
│ MLP Head (example):             │
│                                 │
│  Linear(d_model → 512)          │
│       ↓                         │
│  ReLU + BatchNorm + Dropout     │
│       ↓                         │
│  Linear(512 → 256)              │
│       ↓                         │
│  ReLU + BatchNorm + Dropout     │
│       ↓                         │
│  Linear(256 → 128)              │
│       ↓                         │
│  ReLU + BatchNorm + Dropout     │
│       ↓                         │
│  Linear(128 → 1)                │
└─────────────────────────────────┘
         ↓
Predicted connectivity: (B,)
```

## Cross-Attention Pattern Interpretation

The cross-attention pattern allows the model to learn:

1. **CLS token (position 0 in queries)** attends to all genes in region j
   - Learns a global representation of how region i relates to region j
   - Weighted aggregation based on relevance of region j genes

2. **Region i gene tokens (positions 1+)** attend to region j genes
   - Each gene in region i learns context from region j
   - All tokens contribute to final representation via pooling

3. **Pooling aggregates information**:
   - **Mean**: Equal weight to all tokens (CLS + genes)
   - **Attention**: Learned importance weights for each token
   - **CLS-only**: Uses only the CLS token, gene tokens enrich via residuals

## Key Design Decisions

### Why Cross-Attention Instead of Self-Attention?

- **Self-attention** (previous models): Each region independently, then concatenate
  - Processes regions in isolation
  - Interaction happens only at MLP head

- **Cross-attention** (this model): Direct interaction during encoding
  - CLS token learns to attend to relevant genes in opposite region
  - More direct modeling of region-pair relationships
  - Asymmetric: captures directional relationships

### Why CLS Token at Position 0?

- Established convention (BERT, ViT, etc.)
- Clear semantic meaning: aggregator token
- Easy to extract after attention layers
- Can be initialized with domain knowledge (coordinates)

### Expression Binning Benefits

1. **Discretization**: Reduces continuous space to manageable vocabulary
2. **Robustness**: Less sensitive to exact expression values
3. **Learnable**: Bin embeddings adapt during training
4. **Compatible**: Works with Gene2Vec's discrete token paradigm

## Tensor Shape Summary

| Stage | Tensor | Shape |
|-------|--------|-------|
| Input | gene_expr_i | (B, num_genes) |
| Input | gene_expr_j | (B, num_genes) |
| Input | coords_i, coords_j | (B, 3) each |
| After subset | expr_subset | (B, num_shared_genes) |
| After binning | binned_expr | (B, num_shared_genes, expression_bins) |
| Bin projection | bin_emb | (B, num_shared_genes, 200) |
| Gene2Vec | g2v_emb | (num_shared_genes, 200) |
| Combined | combined_emb | (B, num_shared_genes, 200) |
| To d_model | embeddings | (B, num_shared_genes, d_model) |
| CLS token | cls_token | (B, 1, d_model) |
| Queries | queries | (B, num_shared_genes + 1, d_model) |
| Keys/Values | keys_values | (B, num_shared_genes, d_model) |
| After attention | queries_out | (B, num_shared_genes + 1, d_model) |
| Pooling (mean) | pooled_output | (B, d_model) |
| Pooling (attention) | pooled_output | (B, d_model) |
| Pooling (cls) | pooled_output | (B, d_model) |
| Final output | prediction | (B,) |

## Mathematical Formulation

Given region pair (i, j):

1. **Gene embedding**: 
   ```
   e_g = Linear(OneHot(Bin(x_g))) + Gene2Vec(g)
   h_g = Linear(e_g)
   ```

2. **CLS initialization**:
   ```
   CLS₀ = Linear(Normalize([coord_i; coord_j]))  # spatial
   CLS₀ = Linear(θ_random)                        # random
   ```

3. **Cross-attention**:
   ```
   Q = [CLS₀; h_{i,1}; ...; h_{i,N}]  # queries
   K = [h_{j,1}; ...; h_{j,N}]         # keys
   V = [h_{j,1}; ...; h_{j,N}]         # values
   
   Q_out = CrossAttention(Q, K, V)
   ```

4. **Pooling**:
   ```
   # Mean pooling (default)
   h_final = Mean(Q_out) = 1/(N+1) Σ Q_out[i]
   
   # Attention pooling
   h_final = AttentionPool(Q_out)
   
   # CLS-only
   h_final = Q_out[0]
   ```

5. **Prediction**:
   ```
   y_pred = MLP(h_final)
   ```

