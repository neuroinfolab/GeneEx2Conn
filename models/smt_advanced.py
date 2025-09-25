from env.imports import *
from models.train_val import train_model
from models.smt_utils import *
from models.pls import PLSEncoder
from models.smt import BaseTransformerModel

import torch
import torch.nn.functional as F
from torch.nn import RMSNorm
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
from torch.cuda.amp import autocast
from transformers import get_cosine_schedule_with_warmup
from collections import OrderedDict


class PCAEncoder(nn.Module):
    def __init__(self, train_indices, region_pair_dataset, n_components=128, scale=True, device=None):
        super().__init__()
        self.n_components = n_components
        self.device = device
        self.region_pair_dataset = region_pair_dataset
        self.X = region_pair_dataset.X

        # Fit PCA on training data only
        X_train = self.X[train_indices]
        print(f"Fitting PCA on shape: {X_train.shape}")
        self.pca = PCA(n_components=n_components)
        self.pca.fit(X_train)

        # Store projection matrix and mean for forward pass
        self.mean = torch.FloatTensor(self.pca.mean_).to(device)
        self.proj_matrix = torch.FloatTensor(self.pca.components_.T).to(device)
        self.cached_projection = torch.matmul(torch.FloatTensor(self.X).to(device) - self.mean, self.proj_matrix)

    def forward(self, x, expanded_idx):
        idxs = expanded_idx.view(-1).tolist()
        region_pairs = np.array([self.region_pair_dataset.expanded_idx_to_valid_pair[idx] for idx in idxs])
        region_i = region_pairs[:, 0]
        region_j = region_pairs[:, 1]
        x_scores_i = self.cached_projection[region_i]
        x_scores_j = self.cached_projection[region_j]
        return x_scores_i, x_scores_j

class SharedSelfAttentionPCAModel(BaseTransformerModel):
    def __init__(self, input_dim, train_indices, test_indices, region_pair_dataset,
                 n_components=128, d_model=128, nhead=4, num_layers=4,
                 deep_hidden_dims=[256, 128], dropout_rate=0.1, aug_prob=0.0,
                 transformer_dropout=0.1, learning_rate=1e-3, weight_decay=0.0,
                 batch_size=256, epochs=100, num_workers=2, prefetch_factor=2):
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor, d_model, nhead, num_layers, deep_hidden_dims, aug_prob)
        
        self.aug_prob = aug_prob
        self.input_dim = input_dim
        self.n_components = n_components
        self.transformer_dropout = transformer_dropout
        self.dropout_rate = dropout_rate
        self.use_attention_pooling = True
        self.optimize_encoder = False

        self.PCAencoder = PCAEncoder(
            train_indices=train_indices,
            region_pair_dataset=region_pair_dataset,
            n_components=n_components,
            scale=True,
            device=self.device
        )
        
        self.encoder = FlashAttentionEncoder(
            token_encoder_dim=self.n_components,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=transformer_dropout,
            use_alibi=False,
            use_attention_pooling=True,
        )
        self.encoder = torch.compile(self.encoder)

        self.deep_layers, self.output_layer = self._set_mlp_head(
            prev_dim=d_model * 2,
            deep_hidden_dims=deep_hidden_dims,
            dropout_rate=dropout_rate
        )

        self._setup_optimizer_scheduler(learning_rate, weight_decay)

    def forward(self, x, idx):
        # PCA encode each half
        x_i, x_j = self.PCAencoder(x, idx)  # each: (B, n_components)

        if self.store_attn:
            x_i, attn_beta_i = self.encoder(x_i, return_attn=True)
            x_j, attn_beta_j = self.encoder(x_j, return_attn=True)
        else:
            x_i = self.encoder(x_i)
            x_j = self.encoder(x_j)

        x = torch.cat([x_i, x_j], dim=1)
        y_pred = self.output_layer(self.deep_layers(x))
        y_pred = torch.clamp(y_pred, min=-1.0, max=1.0)
        
        if self.store_attn:
            return {"output": y_pred.squeeze(), "attn_beta_i": attn_beta_i, "attn_beta_j": attn_beta_j}
        
        return y_pred.squeeze()

    def predict(self, loader, collect_attn=False, save_attn_path=None):
        self.eval()
        predictions = []
        targets = []
        
        all_attn = []
        total_batches = 0
        
        self.store_attn = collect_attn
        if collect_attn and not getattr(self, 'use_attention_pooling', False):
            collect_full_attention_heads(self.encoder.layers)
        
        with torch.no_grad():
            for batch_X, batch_y, _, batch_idx in loader:
                batch_X = batch_X.to(self.device)
                
                if collect_attn:
                    out = self(batch_X, batch_idx)
                    
                    if getattr(self, 'use_attention_pooling', False):
                        batch_preds = out["output"].cpu().numpy()
                        attns = (out["attn_beta_i"], out["attn_beta_j"])
                        all_attn.append(attns)
                    else:
                        batch_preds = out.cpu().numpy()
                        accumulate_attention_weights(self.encoder.layers, is_first_batch=(total_batches == 0))
                        total_batches += 1
                else:
                    batch_preds = self(batch_X, batch_idx).cpu().numpy()
                
                predictions.append(batch_preds)
                targets.append(batch_y.numpy())
        
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        self.store_attn = False  # Disable attention collection to prevent memory leaks
        if collect_attn:
            if getattr(self, 'use_attention_pooling', False):
                avg_attn_arr = collect_attention_pooling_weights(all_attn, save_attn_path)
            else:
                avg_attn = process_full_attention_heads(self.encoder.layers, total_batches, save_attn_path)
        
        return predictions, targets

class SharedSelfAttentionPLSModel(BaseTransformerModel):
    def __init__(self, input_dim, train_indices, test_indices, region_pair_dataset,
                 n_components=128, optimize_encoder=False, d_model=128, nhead=4, num_layers=4,
                 deep_hidden_dims=[256, 128], dropout_rate=0.1, aug_prob=0.0,
                 transformer_dropout=0.1, learning_rate=1e-3, weight_decay=0.0,
                 batch_size=256, epochs=100, num_workers=2, prefetch_factor=2):
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor, d_model, nhead, num_layers, deep_hidden_dims, aug_prob)
        
        self.n_components = n_components
        self.transformer_dropout = transformer_dropout
        self.dropout_rate = dropout_rate
        self.use_attention_pooling = True
        self.optimize_encoder = optimize_encoder

        self.PLSencoder = PLSEncoder(
            train_indices=train_indices,
            test_indices=test_indices,
            region_pair_dataset=region_pair_dataset,
            n_components=n_components,
            scale=True,
            optimize_encoder=optimize_encoder,
            device=self.device
        )
        
        self.encoder = FlashAttentionEncoder(
            token_encoder_dim=n_components,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=transformer_dropout,
            use_alibi=False,
            use_attention_pooling=True,
        )
        self.encoder = torch.compile(self.encoder)

        self.deep_layers, self.output_layer = self._set_mlp_head(
            prev_dim=d_model * 2,
            deep_hidden_dims=deep_hidden_dims,
            dropout_rate=dropout_rate
        )

        self._setup_optimizer_scheduler(learning_rate, weight_decay)

    def forward(self, x, idx):
        # PLS encode each half
        x_i, x_j = self.PLSencoder(x, idx)  # each: (B, n_components)

        if self.store_attn:
            x_i, attn_beta_i = self.encoder(x_i, return_attn=True)
            x_j, attn_beta_j = self.encoder(x_j, return_attn=True)
        else:
            x_i = self.encoder(x_i)
            x_j = self.encoder(x_j)

        x = torch.cat([x_i, x_j], dim=1)
        y_pred = self.output_layer(self.deep_layers(x))
        y_pred = torch.clamp(y_pred, min=-1.0, max=1.0)
        
        if self.store_attn:
            return {"output": y_pred.squeeze(), "attn_beta_i": attn_beta_i, "attn_beta_j": attn_beta_j}
        
        return y_pred.squeeze()

    def predict(self, loader, collect_attn=False, save_attn_path=None):
        self.eval()
        predictions = []
        targets = []
        
        all_attn = []
        total_batches = 0
        
        self.store_attn = collect_attn
        if collect_attn and not getattr(self, 'use_attention_pooling', False):
            collect_full_attention_heads(self.encoder.layers)
        
        with torch.no_grad():
            for batch_X, batch_y, _, batch_idx in loader:
                batch_X = batch_X.to(self.device)
                
                if collect_attn:
                    out = self(batch_X, batch_idx)
                    
                    if getattr(self, 'use_attention_pooling', False):
                        batch_preds = out["output"].cpu().numpy()
                        attns = (out["attn_beta_i"], out["attn_beta_j"])
                        all_attn.append(attns)
                    else:
                        batch_preds = out.cpu().numpy()
                        accumulate_attention_weights(self.encoder.layers, is_first_batch=(total_batches == 0))
                        total_batches += 1
                else:
                    batch_preds = self(batch_X, batch_idx).cpu().numpy()
                
                predictions.append(batch_preds)
                targets.append(batch_y.numpy())
        
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        self.store_attn = False  # Disable attention collection to prevent memory leaks
        if collect_attn:
            if getattr(self, 'use_attention_pooling', False):
                avg_attn_arr = collect_attention_pooling_weights(all_attn, save_attn_path)
            else:
                avg_attn = process_full_attention_heads(self.encoder.layers, total_batches, save_attn_path)
        
        return predictions, targets

class GeneConvEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, num_tokens=128,
                 kernel_size=32, out_channels=32, stride=16,
                 nhead=4, num_layers=4, dropout=0.1, use_alibi=False):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_tokens = num_tokens

        # First layer: learn local co-expression motifs (~32 genes each)
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2
        )
        self.relu = nn.ReLU()

        # Project motifs → latent embedding dimension
        self.proj = nn.Conv1d(in_channels=out_channels, out_channels=d_model, kernel_size=1)

        # Compress into exactly num_tokens tokens using adaptive pooling
        self.pool = nn.AdaptiveAvgPool1d(output_size=num_tokens)

        # Flash attention blocks
        self.attention_layers = nn.ModuleList([
            FlashAttentionBlock(d_model, nhead, dropout, use_alibi=use_alibi)
            for _ in range(num_layers)
        ])

        # Attention pooling
        self.attn_pool = AttentionPooling(d_model, hidden_dim=32)

    def forward(self, x, return_attn=False):
        # Input: (B, g) → (B, 1, g)
        x = x.unsqueeze(1)

        # Local convolution → motif embeddings
        x = self.relu(self.conv1(x))     # (B, 64, L)

        # Project to latent space
        x = self.proj(x)                 # (B, d_model, L)

        # Compress into num_tokens positionally relevant tokens
        x = self.pool(x)                 # (B, d_model, num_tokens)
        x = x.transpose(1, 2)            # (B, num_tokens, d_model)

        # Process through flash attention blocks
        for layer in self.attention_layers:
            x = layer(x)

        # Final attention pooling
        if return_attn:
            x, attn = self.attn_pool(x, return_attn=True)
            return x, attn
        return self.attn_pool(x)

class SharedSelfAttentionConvModel(BaseTransformerModel):
    def __init__(self, input_dim, d_model=128, num_tokens=128, nhead=4, num_layers=4,
                 deep_hidden_dims=[256, 128], kernel_size=32, out_channels=32, stride=16, 
                 use_alibi=False, transformer_dropout=0.1, aug_prob=0.0,
                 dropout_rate=0.1, learning_rate=1e-3, weight_decay=0.0,
                 batch_size=256, epochs=100, num_workers=2, prefetch_factor=2):
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor, d_model, nhead, num_layers, deep_hidden_dims, aug_prob)

        self.input_dim = input_dim // 2
        self.num_tokens = num_tokens
        
        # Convolutional encoder with embedded flash attention
        self.conv_encoder = GeneConvEncoder(
            input_dim=self.input_dim,
            d_model=d_model,
            num_tokens=num_tokens,
            kernel_size=kernel_size,
            out_channels=out_channels,
            stride=stride,
            nhead=nhead,
            num_layers=num_layers,
            dropout=transformer_dropout,
            use_alibi=use_alibi
        )
        self.conv_encoder = torch.compile(self.conv_encoder)

        # Fully connected regression head
        prev_dim = d_model * 2  # both region embeddings
        self.deep_layers, self.output_layer = self._set_mlp_head(
            prev_dim=prev_dim,
            deep_hidden_dims=deep_hidden_dims,
            dropout_rate=dropout_rate
        )

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in Conv+Attention model: {num_params}")

        self._setup_optimizer_scheduler(learning_rate, weight_decay)

    def forward(self, x):
        # Split pairwise info
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)

        # Process through conv encoder with embedded attention
        if self.store_attn:
            x_i, attn_i = self.conv_encoder(x_i, return_attn=True)
            x_j, attn_j = self.conv_encoder(x_j, return_attn=True)
        else:
            x_i = self.conv_encoder(x_i)
            x_j = self.conv_encoder(x_j)

        # Concatenate embeddings
        x = torch.cat([x_i, x_j], dim=1)

        # Deep regression head
        y_pred = self.output_layer(self.deep_layers(x))
        y_pred = torch.clamp(y_pred, min=-1.0, max=1.0).squeeze()

        if self.store_attn:
            return {"output": y_pred, "attn_i": attn_i, "attn_j": attn_j}
        return y_pred

# === AUTOENCODER CLASSES === #
# Implementation from https://github.com/jamesruffle/compressed-transcriptomics/
class MLP(nn.Module):
    """
    A n-layer multi-layer perceptron

    An implementation of a n-layer MLP with
    ELU activation and batch normalization
    """

    def __init__(self, layer_sizes):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.mlp = self._create_model(self.layer_sizes)

    def _create_model(self, layer_sizes):
        layer_list = []
        previous_size = layer_sizes[0]
        j = 0
        for i, size in enumerate(layer_sizes[1:-1]):
            layer_list.append((f"fc{i}", nn.Linear(previous_size, size)))
            layer_list.append((f"bn{i}", nn.BatchNorm1d(size)))
            layer_list.append((f"elu{i}", nn.ELU()))
            previous_size = size
            j = i
        layer_list.append((f"fc{j+1}", nn.Linear(previous_size, layer_sizes[-1])))
        layers = OrderedDict(layer_list)
        return nn.Sequential(layers)

    def forward(self, x):
        return self.mlp(x)

class AE(nn.Module):
    """Deep Autoencoder

    Arbitrary-depth autoencoder
    with batch normalization and ELU
    activation on the hidden layers
    """

    def __init__(self, layer_sizes, sigmoid_output=False):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.sigmoid_output = sigmoid_output
        self.encoder = MLP(self.layer_sizes)
        self.decoder = MLP(list(reversed(self.layer_sizes)))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return torch.sigmoid(self.decode(z)) if self.sigmoid_output else self.decode(z)

class AEEncoder(nn.Module):
    def __init__(self, input_dim, n_components=128, weights_path='./data/compressed_transcriptomics/', 
                 finetune_last_layer=False, device=None):
        super().__init__()
        self.n_components = n_components
        self.device = device
        self.finetune_last_layer = finetune_last_layer
        self.weights_path = weights_path

        # Define autoencoder layer sizes (matching notebook pattern)
        # Assuming input_dim is the gene expression dimension
        layer_sizes = [input_dim, 500, 250, 125, n_components]
        
        # Create autoencoder
        self.autoencoder = AE(layer_sizes, sigmoid_output=True).to(device)
        
        # Load pretrained weights
        weight_file = f'autoencode_abagen_{n_components}_components.pt'
        full_path = weights_path + weight_file
        try:
            self.autoencoder.load_state_dict(torch.load(full_path, map_location=device))
            print(f"Loaded pretrained autoencoder weights from {full_path}")
        except FileNotFoundError:
            print(f"Warning: Could not find pretrained weights at {full_path}")
            print("Autoencoder will be initialized with random weights")
        
        # Freeze all parameters except optionally the last layer
        for param in self.autoencoder.parameters():
            param.requires_grad = False
            
        if finetune_last_layer:
            # Enable gradients for the last layer of the encoder
            for param in list(self.autoencoder.encoder.mlp.children())[-1].parameters():
                param.requires_grad = True
            print(f"Enabled fine-tuning for last layer of autoencoder encoder")

    def forward(self, x):
        # Direct encoding of input data
        return self.autoencoder.encode(x)

class SharedSelfAttentionAEModel(BaseTransformerModel):
    def __init__(self, input_dim, n_components=128, finetune_last_layer=False, d_model=128, encoder_output_dim=10, nhead=4, num_layers=4, 
                 deep_hidden_dims=[256, 128], transformer_dropout=0.1, dropout_rate=0.1, 
                 learning_rate=0.001, weight_decay=0.0, batch_size=256, aug_prob=0.0, epochs=100, 
                 num_workers=2, prefetch_factor=2):
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor, d_model, nhead, num_layers, deep_hidden_dims, aug_prob)
        
        self.input_dim = input_dim // 2 # gene_list must be '1' in sim run to match autoencoder dim
        self.n_components = n_components
        self.encoder_output_dim = encoder_output_dim
        self.use_alibi = False
        self.use_attention_pooling = True
        self.finetune_last_layer = finetune_last_layer

        # Pretrained autoencoder encoder-only for each region
        self.ae_encoder = AEEncoder(
            input_dim=self.input_dim,
            n_components=n_components,
            weights_path='./data/compressed_transcriptomics/',
            finetune_last_layer=self.finetune_last_layer,
            device=self.device
        )
        
        # Transformer encoder with attention pooling (uses AE embeddings as input)
        self.encoder = FlashAttentionEncoder(
            token_encoder_dim=self.n_components, # Learn a projection matrix for full embedding to d_model space
            d_model=self.d_model,
            output_dim=self.encoder_output_dim,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=transformer_dropout,
            use_alibi=self.use_alibi,
            use_attention_pooling=self.use_attention_pooling
        )
        self.encoder = torch.compile(self.encoder)

        prev_dim = self.d_model * 2
        self.deep_layers, self.output_layer = self._set_mlp_head(
            prev_dim=prev_dim,
            deep_hidden_dims=deep_hidden_dims,
            dropout_rate=dropout_rate
        )

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in SMT AE model: {num_params}")

        self._setup_optimizer_scheduler(learning_rate, weight_decay)

    def forward(self, x):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        
        # Encode through pretrained autoencoder
        x_i = self.ae_encoder(x_i)
        x_j = self.ae_encoder(x_j)
        
        if self.store_attn:
            x_i, attn_beta_i = self.encoder(x_i, return_attn=True)
            x_j, attn_beta_j = self.encoder(x_j, return_attn=True)
        else:
            x_i = self.encoder(x_i)
            x_j = self.encoder(x_j)

        x = torch.cat([x_i, x_j], dim=1)
        y_pred = self.output_layer(self.deep_layers(x))
        y_pred = torch.clamp(y_pred, min=-1.0, max=1.0)
        
        if self.store_attn:
            return {"output": y_pred.squeeze(), "attn_beta_i": attn_beta_i, "attn_beta_j": attn_beta_j}
        
        return y_pred.squeeze()
        
# === CELLTYPE ENCODER CLASSES === #
class CelltypeEncoder(nn.Module):
    def __init__(self, aux_feature_dim, d_model, nhead=4, num_layers=4, dropout=0.1, use_alibi=False, projection_layers=2, use_attention_pooling=True):
        super().__init__()
        self.aux_feature_dim = aux_feature_dim
        self.d_model = d_model
        self.projection_layers = projection_layers
        self.use_attention_pooling = use_attention_pooling
        
        # Project from (expression + aux_features) to transformer embedding dimension
        # Input: gene expression (1) + auxiliary features (aux_feature_dim) = (aux_feature_dim + 1)
        if projection_layers == 1:
            self.feature_projection = nn.Linear(aux_feature_dim + 1, d_model)
        else: # 3 is MLP
            # More expressive MLP projection
            hidden_dim = max(d_model, (aux_feature_dim + 1) * 2)
            self.feature_projection = nn.Sequential(
                nn.Linear(aux_feature_dim + 1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.LayerNorm(d_model)
            )
        
        # Transformer layers for self-attention across genes
        self.transformer_layers = nn.ModuleList([
            FlashAttentionBlock(d_model, nhead, dropout, use_alibi=use_alibi)
            for _ in range(num_layers)
        ])
        
        # Choose pooling method
        if use_attention_pooling:
            self.pooling = AttentionPooling(d_model, hidden_dim=32)
        else:
            self.pooling = nn.Linear(d_model, 1)
        
    def forward(self, gene_expression, aux_features, return_attn=False):
        """
        Args:
            gene_expression: (B, num_genes) - gene expression values
            aux_features: (num_genes, aux_feature_dim) - fixed auxiliary features per gene
            return_attn: bool - whether to return attention weights
        
        Returns:
            pooled representation: (B, d_model) if use_attention_pooling else (B, num_genes)
            optionally attention weights
        """
        B, num_genes = gene_expression.shape
        
        # Expand aux_features to match batch size: (B, num_genes, aux_feature_dim)
        aux_features_expanded = aux_features.unsqueeze(0).expand(B, -1, -1)
        
        # Expand gene expression: (B, num_genes, 1)
        gene_expression_expanded = gene_expression.unsqueeze(-1)
        
        # Concatenate: (B, num_genes, aux_feature_dim + 1)
        combined_features = torch.cat([gene_expression_expanded, aux_features_expanded], dim=-1)
        
        # Project to transformer embedding dimension: (B, num_genes, d_model)
        x = self.feature_projection(combined_features)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Pool output based on method
        if self.use_attention_pooling:
            if return_attn:
                pooled, attn_weights = self.pooling(x, return_attn=True)
                return pooled, attn_weights
            else:
                pooled = self.pooling(x)
                return pooled
        else:
            # Project each token to scalar
            x = self.pooling(x).squeeze(-1)  # (B, num_genes)
            if return_attn:
                # Return dummy attention weights when not using attention pooling
                return x, None
            return x

class SharedSelfAttentionCelltypeModel(BaseTransformerModel):
    def __init__(self, input_dim, region_pair_dataset, aux_data_path_dfc='./data/gene_emb/LakeDFC_gene_signature.csv', aux_data_path_vis='./data/gene_emb/LakeVIS_gene_signature.csv', 
                 d_model=128, nhead=4, num_layers=4, deep_hidden_dims=[512, 256, 128],
                 projection_layers=1, use_alibi=False, transformer_dropout=0.1, dropout_rate=0.1, pooling=True,
                 learning_rate=0.001, weight_decay=0.0, batch_size=512, epochs=100, 
                 aug_prob=0.0, num_workers=2, prefetch_factor=2):
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor, d_model, nhead, num_layers, deep_hidden_dims, aug_prob)

        self.input_dim = input_dim // 2
        self.aug_prob = aug_prob
        self.pooling = pooling
        self.valid_genes = region_pair_dataset.valid_genes

        # Load and process auxiliary data
        self._load_auxiliary_data(aux_data_path_dfc, aux_data_path_vis)
        
        # Create celltype encoder
        self.celltype_encoder = CelltypeEncoder(
            aux_feature_dim=self.aux_feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=transformer_dropout,
            use_alibi=use_alibi,
            projection_layers=projection_layers,
            use_attention_pooling=self.pooling
        )
        self.celltype_encoder = torch.compile(self.celltype_encoder)
        
        # MLP head for final prediction
        if self.pooling:
            prev_dim = d_model * 2  # Concatenated embeddings from both regions
        else:
            prev_dim = len(self.shared_gene_indices)  * 2  # Concatenated token projections from both regions
        
        self.deep_layers, self.output_layer = self._set_mlp_head(
            prev_dim=prev_dim,
            deep_hidden_dims=deep_hidden_dims,
            dropout_rate=dropout_rate
        )
        
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in Celltype model: {num_params}")
        print(f"Number of genes after intersection: {len(self.shared_gene_indices)}")
        print(f"Auxiliary feature dimension: {self.aux_feature_dim}")
        
        self._setup_optimizer_scheduler(learning_rate, weight_decay, use_cosine=True)
    
    def _load_auxiliary_data(self, aux_data_path_dfc, aux_data_path_vis):
        """Load and process auxiliary cell type data"""
        # Load auxiliary data
        lake_dfc_df = pd.read_csv(aux_data_path_dfc, index_col=0)
        lake_vis_df = pd.read_csv(aux_data_path_vis, index_col=0)
        
        # Get gene lists
        lake_dfc_genes = lake_dfc_df.index.tolist()
        lake_vis_genes = lake_vis_df.index.tolist()
        
        # Find intersection between DFC, VIS, and valid genes
        lake_shared_genes = list(set(lake_dfc_genes).intersection(set(lake_vis_genes)))
        lake_shared_valid = list(set(lake_shared_genes).intersection(set(self.valid_genes)))
        
        # Get indices of shared valid genes in the original gene list
        self.shared_gene_indices = [self.valid_genes.index(gene) for gene in lake_shared_valid]
        
        # Subset auxiliary data to shared valid genes
        lake_dfc_subset = lake_dfc_df.loc[lake_shared_valid]
        lake_vis_subset = lake_vis_df.loc[lake_shared_valid]
        
        # Convert to numpy and apply log1p normalization
        lake_dfc_matrix = np.log1p(lake_dfc_subset.values)
        lake_vis_matrix = np.log1p(lake_vis_subset.values)
        
        # Average the DFC and VIS matrices (or use separately if needed, logical since they are highly correlated (0.94), so using both may be redundant)
        # For now, averaging them as a combined cell type signature
        combined_aux_matrix = (lake_dfc_matrix + lake_vis_matrix) / 2
        
        # Store as tensor
        self.aux_features = torch.FloatTensor(combined_aux_matrix).to(self.device)
        self.aux_feature_dim = combined_aux_matrix.shape[1]

        self.input_dim = len(self.shared_gene_indices) + self.aux_feature_dim
        
        print(f"Loaded auxiliary data: {combined_aux_matrix.shape[0]} genes x {combined_aux_matrix.shape[1]} cell types")
    
    def _subset_gene_expression(self, x):
        """Subset gene expression to shared valid genes"""
        # x shape: (B, input_dim) where input_dim = 2 * num_genes
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)  # Each: (B, num_genes)
        
        # Subset to shared genes
        x_i_subset = x_i[:, self.shared_gene_indices]  # (B, num_shared_genes)
        x_j_subset = x_j[:, self.shared_gene_indices]  # (B, num_shared_genes)
        
        return x_i_subset, x_j_subset
    
    def forward(self, x):
        # Subset gene expression to shared valid genes
        x_i, x_j = self._subset_gene_expression(x)
        
        # Encode both regions using celltype encoder
        if self.store_attn:
            x_i, attn_beta_i = self.celltype_encoder(x_i, self.aux_features, return_attn=True)
            x_j, attn_beta_j = self.celltype_encoder(x_j, self.aux_features, return_attn=True)
        else:
            x_i = self.celltype_encoder(x_i, self.aux_features)
            x_j = self.celltype_encoder(x_j, self.aux_features)
        
        # Concatenate region embeddings
        x = torch.cat([x_i, x_j], dim=1)
        
        # Final prediction
        y_pred = self.output_layer(self.deep_layers(x))
        
        if self.store_attn:
            return {"output": y_pred.squeeze(), "attn_beta_i": attn_beta_i, "attn_beta_j": attn_beta_j}
        
        return y_pred.squeeze()

    def fit(self, dataset, train_indices, test_indices, save_model=None, verbose=True):
        """Shared fit function for all transformer models"""
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                                num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True,
                               num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)
        
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, scheduler=None, train_scheduler=self.scheduler, save_model=save_model, verbose=verbose, dataset=dataset)

class GeneformerEncoder(nn.Module):
    def __init__(self, token_encoder_dim, d_model, nhead=4, num_layers=4, dropout=0.1, use_alibi=False, use_attention_pooling=False, use_mlp_downsampler=False):
        super().__init__()
        
        # Linear input projection from scalar token (token_encoder_dim=1) to d_model space
        self.input_projection = nn.Linear(token_encoder_dim, d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            FlashAttentionBlock(d_model, nhead, dropout, use_alibi=use_alibi)
            for _ in range(num_layers)
        ])
        
        self.use_attention_pooling = use_attention_pooling
        if self.use_attention_pooling:
            self.pooling = AttentionPooling(d_model, hidden_dim=32)
        else:
            # Downsampler: either linear or 2-layer MLP (mimicking CelltypeEncoder logic)
            if use_mlp_downsampler:
                # 2-layer MLP downsampler
                self.downsampler = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1)
                )
            else:
                # Simple linear downsampler (like CelltypeEncoder)
                self.downsampler = nn.Linear(d_model, 1)
        
    def forward(self, x, return_attn=False):
        B, T = x.shape
        # Reshape to (B, num_tokens, token_encoder_dim) where token_encoder_dim=1
        x = x.view(B, T, 1)  # Each token is a scalar
        
        # Linear projection to d_model space
        x = self.input_projection(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        if self.use_attention_pooling:
            return self.pooling(x, return_attn=return_attn)
        else:
            # Apply downsampler to each token: (B, num_tokens, d_model) -> (B, num_tokens, 1)
            x = self.downsampler(x).squeeze(-1)  # (B, num_tokens)
            if return_attn:
                return x, None
            return x

class SharedSelfAttentionGeneformerModel(BaseTransformerModel):
    def __init__(self, input_dim, token_encoder_dim=1, d_model=128, nhead=4, num_layers=4, deep_hidden_dims=[256, 128],
                 use_alibi=False, transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0,
                 batch_size=512, aug_prob=0.0, epochs=100, num_workers=2, prefetch_factor=2, 
                 use_attention_pooling=False, use_mlp_downsampler=False):
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor, d_model, nhead, num_layers, deep_hidden_dims, aug_prob)
        
        self.input_dim = input_dim // 2  # Split for two regions
        self.token_encoder_dim = token_encoder_dim  # Should be 1 for Geneformer embeddings
        self.use_alibi = use_alibi
        self.use_attention_pooling = use_attention_pooling
        self.use_mlp_downsampler = use_mlp_downsampler
        
        # Number of tokens (approximately 770)
        self.num_tokens = self.input_dim // self.token_encoder_dim
        
        # Geneformer encoder
        self.encoder = GeneformerEncoder(
            token_encoder_dim=self.token_encoder_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=transformer_dropout,
            use_alibi=self.use_alibi,
            use_attention_pooling=self.use_attention_pooling,
            use_mlp_downsampler=self.use_mlp_downsampler
        )
        self.encoder = torch.compile(self.encoder)
        
        # Calculate output dimension from encoder
        if self.use_attention_pooling:
            prev_dim = self.d_model * 2  # Pooled output from both regions
        else:
            # Downsampler outputs (B, num_tokens) per region, so total is num_tokens * 2
            prev_dim = self.num_tokens * 2  # Both regions
        
        # MLP head for final prediction
        self.deep_layers, self.output_layer = self._set_mlp_head(
            prev_dim=prev_dim,
            deep_hidden_dims=deep_hidden_dims,
            dropout_rate=dropout_rate
        )
        
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in SharedSelfAttentionGeneformer model: {num_params}")
        print(f"Number of tokens per region: {self.num_tokens}")
        print(f"Attention pooling: {self.use_attention_pooling}")
        print(f"MLP downsampler: {self.use_mlp_downsampler}")
        
        self._setup_optimizer_scheduler(learning_rate, weight_decay) #, use_cosine=True)

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
        
        if self.store_attn:
            return {"output": y_pred.squeeze(), "attn_beta_i": attn_beta_i, "attn_beta_j": attn_beta_j}
        
        return y_pred.squeeze()

    '''
    def fit(self, dataset, train_indices, test_indices, save_model=None, verbose=True):
        """Shared fit function for all transformer models"""
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                                num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True,
                               num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)
        
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, scheduler = None, train_scheduler=self.scheduler, save_model=save_model, verbose=verbose, dataset=dataset)
    '''