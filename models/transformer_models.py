from env.imports import *
from data.data_utils import create_data_loader
from models.train_val import train_model
from models.transformer_utils import *
from models.pls import PLSEncoder
import torch
import torch.nn.functional as F
from torch.nn import RMSNorm
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
from torch.cuda.amp import autocast
from collections import OrderedDict

# === CORE TRANSFORMER COMPONENTS === #
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
        
        # FiLM modulation parameters (for CLS token conditioning)
        self.gamma_proj = nn.Linear(input_dim, input_dim)
        self.beta_proj = nn.Linear(input_dim, input_dim)
        self.use_residual = use_residual

        # control how much to bias towards cls token
        # gate_bias=-1.0
        # self.cls_gate = nn.Linear(input_dim, input_dim)
        # self.cls_gate.bias.data.fill_(gate_bias) # more negative values favor genetic data, more positive values favor cls data

    def forward(self, x, return_attn=False, cls_token=False):
        B, L, D = x.shape
        
        if cls_token:
            # Split CLS from rest
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

            # Residual connection (optional)
            pooled = x_gene + modulated if self.use_residual else modulated
            
            # GATED CLS TOKEN SCALING
            # # Compute pooled output
            # pooled_rest = torch.bmm(attn_weights.transpose(1, 2), x_rest)  # (B, 1, D)

            # # Gate CLS token
            # cls_gate = torch.sigmoid(self.cls_gate(cls_token))      # (B, 1, D)
            # pooled = (pooled_rest * (1 - cls_gate) + cls_token * cls_gate).squeeze(1)
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

class FlashAttentionEncoder(nn.Module):
    def __init__(self, token_encoder_dim, d_model, output_dim=10, nhead=4, num_layers=4, dropout=0.1, use_alibi=False, use_attention_pooling=True):
        super().__init__()
        
        self.input_projection = nn.Linear(token_encoder_dim, d_model)
        
        self.transformer_layers = nn.ModuleList([
            FlashAttentionBlock(d_model, nhead, dropout, use_alibi=use_alibi)
            for _ in range(num_layers)
        ])
        
        self.use_attention_pooling = use_attention_pooling
        if self.use_attention_pooling:
            self.pooling = AttentionPooling(d_model, hidden_dim=32)
        else:
            self.output_projection = nn.Linear(d_model, output_dim)
        
    def forward(self, x, return_attn=False):
        B, T = x.shape
        x = x.view(B, -1, self.input_projection.in_features) # Key chunking step
        
        x = self.input_projection(x)
        for layer in self.transformer_layers:
            x = layer(x)
        if self.use_attention_pooling:
            return self.pooling(x, return_attn=return_attn)
        else:
            x = self.output_projection(x)
            x = x.flatten(start_dim=1)
            if return_attn:
                return x, None
            return x

# === BASE TRANSFORMER MODEL === #
class BaseTransformerModel(nn.Module):
    def __init__(self, input_dim, learning_rate=0.0001, weight_decay=0.0001, batch_size=512, epochs=100, num_workers=2, prefetch_factor=4):
        super().__init__()
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.store_attn = False
        
        # Set by subclasses
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.patience = 50
        self.scheduler = None
    
    def _setup_optimizer_scheduler(self, learning_rate, weight_decay, patience=30):
        """Setup optimizer and scheduler - called by subclasses"""
        self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.3, patience=patience,
            threshold=0.1, cooldown=1, min_lr=1e-6, verbose=True
        )
    
    def _set_mlp_head(self, prev_dim, deep_hidden_dims, dropout_rate):
        deep_layers = []
        for hidden_dim in deep_hidden_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim, dtype=torch.float32),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        return nn.Sequential(*deep_layers), nn.Linear(prev_dim, 1)

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
            for batch_idx, batch_data in enumerate(loader):
                batch_X, batch_y = batch_data[0], batch_data[1]
                batch_coords = batch_data[2] if len(batch_data) > 2 else None
                
                batch_X = batch_X.to(self.device)
                if batch_coords is not None:
                    batch_coords = batch_coords.to(self.device)
                
                if collect_attn:
                    out = self._forward_with_coords(batch_X, batch_coords) if batch_coords is not None else self(batch_X)
                    
                    if getattr(self, 'use_attention_pooling', False): # defaults to False
                        batch_preds = out["output"].cpu().numpy()
                        attns = (out["attn_beta_i"], out["attn_beta_j"])
                        all_attn.append(attns)
                    else:
                        batch_preds = out.cpu().numpy()
                        accumulate_attention_weights(self.encoder.layers, is_first_batch=(batch_idx == 0))
                        total_batches += 1
                else:
                    batch_preds = self._forward_with_coords(batch_X, batch_coords).cpu().numpy() if batch_coords is not None else self(batch_X).cpu().numpy()
                
                predictions.append(batch_preds)
                targets.append(batch_y.numpy())
        
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        self.store_attn = False  # Disable attention collection to prevent memory leaks
        if collect_attn:
            if getattr(self, 'use_attention_pooling', False):
                avg_attn_arr = collect_attention_pooling_weights(all_attn, save_attn_path)
                return predictions, targets, avg_attn_arr, all_attn
            else:
                avg_attn = process_full_attention_heads(self.encoder.layers, total_batches, save_attn_path)
                return predictions, targets, avg_attn
        
        return predictions, targets

    def _forward_with_coords(self, x, coords):
        """Default implementation for models without coords - override in subclasses"""
        return self(x)

    def fit(self, dataset, train_indices, test_indices, save_model=None, verbose=True):
        """Shared fit function for all transformer models"""
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                                num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True,
                               num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)
        
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, self.scheduler, save_model=save_model, verbose=verbose, dataset=dataset)

class SharedSelfAttentionModel(BaseTransformerModel):
    def __init__(self, input_dim, token_encoder_dim=20, d_model=128, encoder_output_dim=10, nhead=2, num_layers=2, deep_hidden_dims=[256, 128],
                 use_alibi=False, transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0,
                 batch_size=256, aug_prob=0.0, epochs=100, num_workers=2, prefetch_factor=2):
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor)
        
        self.input_dim = input_dim // 2
        self.token_encoder_dim = token_encoder_dim
        self.d_model = d_model
        self.encoder_output_dim = encoder_output_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.aug_prob = aug_prob
        self.use_alibi = use_alibi
        self.use_attention_pooling = False
        
        self.num_tokens = self.input_dim // self.token_encoder_dim
        
        self.encoder = FlashAttentionEncoder(
            token_encoder_dim=self.token_encoder_dim,
            d_model=self.d_model,
            output_dim=self.encoder_output_dim,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=transformer_dropout,
            use_alibi=self.use_alibi,
            use_attention_pooling=False,
        )
        self.encoder = torch.compile(self.encoder)
        
        prev_dim = (self.encoder_output_dim * self.num_tokens * 2)
        
        deep_layers = []
        for hidden_dim in deep_hidden_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim, dtype=torch.float32),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        self.deep_layers = nn.Sequential(*deep_layers)
        self.output_layer = nn.Linear(prev_dim, 1)
        
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in SMT model: {num_params}")
        
        self._setup_optimizer_scheduler(learning_rate, weight_decay)

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
        
        return y_pred.squeeze()

class SharedSelfAttentionPoolingModel(BaseTransformerModel):
    def __init__(self, input_dim, token_encoder_dim=20, d_model=128, encoder_output_dim=10, nhead=2, num_layers=2, deep_hidden_dims=[256, 128],
                 use_alibi=False, transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0,
                 batch_size=256, aug_prob=0.0, epochs=100, num_workers=2, prefetch_factor=2):
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor)
        
        self.input_dim = input_dim // 2
        self.token_encoder_dim = token_encoder_dim
        self.d_model = d_model
        self.encoder_output_dim = encoder_output_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.aug_prob = aug_prob
        self.use_alibi = use_alibi
        self.use_attention_pooling = True
        self.num_tokens = self.input_dim // self.token_encoder_dim
        
        self.encoder = FlashAttentionEncoder(
            token_encoder_dim=self.token_encoder_dim,
            d_model=self.d_model,
            output_dim=self.encoder_output_dim,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=transformer_dropout,
            use_alibi=self.use_alibi,
            use_attention_pooling=True
        )
        self.encoder = torch.compile(self.encoder)
        
        prev_dim = self.d_model * 2
        self.deep_layers, self.output_layer = self._set_mlp_head(
            prev_dim=prev_dim,
            deep_hidden_dims=deep_hidden_dims,
            dropout_rate=dropout_rate
        )
        
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in SMT model: {num_params}")
        
        self._setup_optimizer_scheduler(learning_rate, weight_decay)

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
        
        if self.store_attn:
            return {"output": y_pred.squeeze(), "attn_beta_i": attn_beta_i, "attn_beta_j": attn_beta_j}
        
        return y_pred.squeeze()

class SelfAttentionCLSEncoder(nn.Module):    
    def __init__(self, token_encoder_dim, d_model, output_dim, nhead=4, num_layers=4, dropout=0.1, cls_init='spatial_learned', use_alibi=False, num_tokens=None, use_attention_pooling=True, cls_in_seq=True, gate_bias=0.5, cls_dropout=0.5):
        super().__init__()
        self.token_encoder_dim = token_encoder_dim
        self.d_model = d_model
        self.cls_init = cls_init
        self.use_alibi = use_alibi
        self.num_tokens = num_tokens
        self.use_attention_pooling = use_attention_pooling
        self.cls_in_seq = cls_in_seq
        self.gate_bias = gate_bias
        self.input_projection = nn.Linear(token_encoder_dim, d_model)
        
        # === CLS token: (x, y, z, distance) → d_model ===
        cls_coord_dim = 4  # x, y, z, distance
        if self.cls_in_seq:
            self.coord_to_cls = nn.Linear(cls_coord_dim, d_model)
        else:
            self.coord_encoder = nn.Sequential(
                nn.Linear(cls_coord_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            )
        self.cls_dropout = nn.Dropout(cls_dropout)  # Adjust the rate as needed

        @staticmethod
        def build_coord_remap_table(coord_tensor, seed=42, scale=100.0):
            torch.manual_seed(seed)
            remap = (torch.rand_like(coord_tensor) * 2 - 1) * scale
            return remap
        
        if self.cls_init == 'random_learned':
            df = pd.read_csv('./data/UKBB/atlas-4S456Parcels_dseg_reformatted.csv')
            region_coords = torch.tensor(df.iloc[:, -3:].values, dtype=torch.float32)
            remap_tensor = self.build_coord_remap_table(region_coords)
            self.register_buffer("coord_ref_table", region_coords)
            self.register_buffer("coord_remap_table", remap_tensor)
        elif self.cls_in_seq and self.cls_init == 'spatial_fixed':
            self.coord_to_cls.weight.requires_grad = False
            self.coord_to_cls.bias.requires_grad = False

        self.transformer_layers = nn.ModuleList([
            FlashAttentionBlock(d_model, nhead, dropout, use_alibi=self.use_alibi)
            for _ in range(num_layers)
        ])

        if self.use_alibi:
            slopes = FlashAttentionBlock.build_alibi_slopes(nhead)
            self.register_buffer("alibi_slopes", slopes)
            for layer in self.transformer_layers:
                layer.alibi_slopes = self.alibi_slopes

        if self.use_attention_pooling:
            self.pooling = AttentionPooling(d_model, hidden_dim=32) #, gate_bias=self.gate_bias)
        else:
            self.output_projection = nn.Linear(d_model, output_dim)
    
    def replace_coords_with_remapped(self, coords):
        B, _ = coords.shape
        match = (coords[:, None, :] == self.coord_ref_table[None, :, :]).all(dim=-1)
        idx = match.float().argmax(dim=1)
        remapped = self.coord_remap_table[idx]
        return remapped
    
    def forward(self, gene_exp, coords, dist_to_target, return_attn=False):
        B, T = gene_exp.shape
        x = gene_exp.view(B, -1, self.token_encoder_dim)
        x = self.input_projection(x)

        coords_input = torch.cat([coords, dist_to_target], dim=-1)  # (B, 4)

        if self.cls_in_seq:
            if self.cls_init == 'random_learned':
                coords_input[:, :3] = self.replace_coords_with_remapped(coords_input[:, :3])
            cls_token = self.coord_to_cls(coords_input).unsqueeze(1)
            x = torch.cat([cls_token, x], dim=1)

        for layer in self.transformer_layers:
            x = layer(x)

        if not self.cls_in_seq:
            if self.cls_init == 'random_learned':
                coords_input[:, :3] = self.replace_coords_with_remapped(coords_input[:, :3])
        
            cls_token = self.coord_encoder(coords_input).unsqueeze(1)
            cls_token = self.cls_dropout(cls_token)
            x = torch.cat([cls_token, x], dim=1)
        
        if self.use_attention_pooling:
            return self.pooling(x, return_attn=return_attn, cls_token=True)
        else:
            x = self.output_projection(x)
            x = x.flatten(start_dim=1)
            return (x, None) if return_attn else x

class SharedSelfAttentionCLSModel(BaseTransformerModel):
    def __init__(self, input_dim, token_encoder_dim=20, d_model=128, encoder_output_dim=10, nhead=2, num_layers=2, deep_hidden_dims=[256, 128], 
                 cls_init='spatial_learned', cls_in_seq=True, use_alibi=False, transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0, 
                 batch_size=128, epochs=100, aug_prob=0.0, num_workers=2, prefetch_factor=2):
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor)
        
        self.include_coords = True
        self.cls_init = cls_init
        self.cls_in_seq = cls_in_seq
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
        self.use_attention_pooling = False
        
        self.num_tokens = self.input_dim // self.token_encoder_dim

        self.encoder = SelfAttentionCLSEncoder(
            token_encoder_dim=self.token_encoder_dim,
            d_model=self.d_model,
            output_dim=self.encoder_output_dim, 
            nhead=self.nhead, 
            num_layers=self.num_layers,
            dropout=self.transformer_dropout,
            cls_init=self.cls_init,
            use_alibi=self.use_alibi,
            use_attention_pooling=False,
            cls_in_seq=self.cls_in_seq
        )
        self.encoder = torch.compile(self.encoder)

        prev_dim = (self.encoder_output_dim * (self.num_tokens + 1) * 2)
        self.deep_layers, self.output_layer = self._set_mlp_head(
            prev_dim=prev_dim,
            deep_hidden_dims=deep_hidden_dims,
            dropout_rate=dropout_rate
        )
        
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in SMT w/ CLS model: {num_params}")

        self._setup_optimizer_scheduler(learning_rate, weight_decay)

    def forward(self, x, coords):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        coords_i, coords_j = torch.chunk(coords, chunks=2, dim=1)
        x_i = self.encoder(x_i, coords_i)
        x_j = self.encoder(x_j, coords_j)
        x = torch.cat([x_i, x_j], dim=1)
        y_pred = self.output_layer(self.deep_layers(x))
        y_pred = torch.clamp(y_pred, min=-0.8, max=1.0)
        return y_pred.squeeze()

    def _forward_with_coords(self, x, coords):
        return self.forward(x, coords)

class SharedSelfAttentionCLSPoolingModel(BaseTransformerModel):
    def __init__(self, input_dim, token_encoder_dim=20, d_model=128, encoder_output_dim=10, nhead=2, num_layers=2,
                 deep_hidden_dims=[256, 128], cls_init='spatial_learned', cls_in_seq=False, cls_dropout=0.5, use_alibi=False,
                 transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0, 
                 batch_size=128, epochs=100, aug_prob=0.0, num_workers=2, prefetch_factor=2, gate_bias=0.5):
        
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor)
        
        self.include_coords = True
        self.cls_init = cls_init
        self.cls_in_seq = cls_in_seq
        self.cls_dropout = cls_dropout
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
        self.use_attention_pooling = True
        self.gate_bias = gate_bias
        self.num_tokens = self.input_dim // self.token_encoder_dim

        self.encoder = SelfAttentionCLSEncoder(
            token_encoder_dim=self.token_encoder_dim,
            d_model=self.d_model,
            output_dim=self.encoder_output_dim, 
            nhead=self.nhead, 
            num_layers=self.num_layers,
            dropout=self.transformer_dropout,
            cls_init=self.cls_init,
            use_alibi=self.use_alibi,
            use_attention_pooling=True,
            cls_in_seq=self.cls_in_seq,
            gate_bias=self.gate_bias,
            cls_dropout=self.cls_dropout
        )
        self.encoder = torch.compile(self.encoder)

        prev_dim = self.d_model * 2
        self.deep_layers, self.output_layer = self._set_mlp_head(
            prev_dim=prev_dim,
            deep_hidden_dims=deep_hidden_dims,
            dropout_rate=dropout_rate
        )

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in SMT w/ CLS pooled model: {num_params}")

        self._setup_optimizer_scheduler(learning_rate, weight_decay)

    def forward(self, x, coords):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        coords_i, coords_j = torch.chunk(coords, chunks=2, dim=1)
        dists = torch.norm(coords_i - coords_j, dim=1, keepdim=True)  # (B, 1)

        if self.store_attn:
            x_i, attn_beta_i = self.encoder(x_i, coords_i, dist_to_target=dists, return_attn=True)
            x_j, attn_beta_j = self.encoder(x_j, coords_j, dist_to_target=dists, return_attn=True)
        else:
            x_i = self.encoder(x_i, coords_i, dist_to_target=dists)
            x_j = self.encoder(x_j, coords_j, dist_to_target=dists)

        x = torch.cat([x_i, x_j], dim=1)
        y_pred = self.output_layer(self.deep_layers(x))
        y_pred = torch.clamp(y_pred, min=-0.8, max=1.0)

        if self.store_attn:
            return {
                "output": y_pred.squeeze(),
                "attn_beta_i": attn_beta_i,
                "attn_beta_j": attn_beta_j
            }

        return y_pred.squeeze()

    def _forward_with_coords(self, x, coords):
        return self.forward(x, coords)


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
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor)

        self.input_dim = input_dim // 2
        self.d_model = d_model
        self.num_tokens = num_tokens
        self.aug_prob = aug_prob
        
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
        # Split into hemispheres
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
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor)
        
        self.aug_prob = aug_prob
        self.input_dim = input_dim
        self.n_components = n_components
        self.d_model = d_model
        self.transformer_dropout = transformer_dropout
        self.nhead = nhead
        self.num_layers = num_layers
        self.deep_hidden_dims = deep_hidden_dims
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
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor)
        
        self.aug_prob = aug_prob
        self.input_dim = input_dim
        self.n_components = n_components
        self.d_model = d_model
        self.transformer_dropout = transformer_dropout
        self.nhead = nhead
        self.num_layers = num_layers
        self.deep_hidden_dims = deep_hidden_dims
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
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor)
        
        self.input_dim = input_dim // 2 # gene_list must be '1' in sim run to match autoencoder dim
        self.n_components = n_components
        self.d_model = d_model
        self.encoder_output_dim = encoder_output_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.aug_prob = aug_prob
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
    def __init__(self, aux_feature_dim, d_model, nhead=4, num_layers=4, dropout=0.1, use_alibi=False, projection_layers=2):
        super().__init__()
        self.aux_feature_dim = aux_feature_dim
        self.d_model = d_model
        self.projection_layers = projection_layers
        
        # Project from (expression + aux_features) to transformer embedding dimension
        # Input: gene expression (1) + auxiliary features (aux_feature_dim) = (aux_feature_dim + 1)
        if projection_layers == 1:
            self.feature_projection = nn.Linear(aux_feature_dim + 1, d_model)
        else:
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
        
        # Attention pooling to get fixed-size representation
        self.pooling = AttentionPooling(d_model, hidden_dim=32)
        
    def forward(self, gene_expression, aux_features, return_attn=False):
        """
        Args:
            gene_expression: (B, num_genes) - gene expression values
            aux_features: (num_genes, aux_feature_dim) - fixed auxiliary features per gene
            return_attn: bool - whether to return attention weights
        
        Returns:
            pooled representation: (B, d_model)
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
        
        # Attention pooling to get fixed-size representation
        if return_attn:
            pooled, attn_weights = self.pooling(x, return_attn=True)
            return pooled, attn_weights
        else:
            pooled = self.pooling(x)
            return pooled


class SharedSelfAttentionCelltypeModel(BaseTransformerModel):
    def __init__(self, input_dim, region_pair_dataset, aux_data_path_dfc='./data/gene_emb/LakeDFC_gene_signature.csv', aux_data_path_vis='./data/gene_emb/LakeVIS_gene_signature.csv', 
                 d_model=128, nhead=4, num_layers=4, deep_hidden_dims=[512, 256, 128],
                 projection_layers=2, use_alibi=False, transformer_dropout=0.1, dropout_rate=0.1, 
                 learning_rate=0.001, weight_decay=0.0, batch_size=512, epochs=100, 
                 aug_prob=0.0, num_workers=2, prefetch_factor=2):
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor)

        self.input_dim = input_dim // 2
        self.d_model = d_model
        self.aug_prob = aug_prob
        self.use_attention_pooling = True
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
            projection_layers=projection_layers
        )
        self.celltype_encoder = torch.compile(self.celltype_encoder)
        
        # MLP head for final prediction
        prev_dim = d_model * 2  # Concatenated embeddings from both regions
        self.deep_layers, self.output_layer = self._set_mlp_head(
            prev_dim=prev_dim,
            deep_hidden_dims=deep_hidden_dims,
            dropout_rate=dropout_rate
        )
        
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in Celltype model: {num_params}")
        print(f"Number of genes after intersection: {len(self.shared_gene_indices)}")
        print(f"Auxiliary feature dimension: {self.aux_feature_dim}")
        
        self._setup_optimizer_scheduler(learning_rate, weight_decay)
    
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

            q = self.split_heads(q).transpose(1, 2)
            k = self.split_heads(k).transpose(1, 2)
            v = self.split_heads(v).transpose(1, 2)
            attn_output = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
            attn_output = attn_output.transpose(1, 2)        
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
        self.transformer_layers = nn.ModuleList([
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

        for layer in self.transformer_layers:
            q = layer(q, kv)

        out = self.output_projection(q)
        out = out.flatten(start_dim=1)
        return out

class CrossAttentionModel(BaseTransformerModel):
    def __init__(self, input_dim, token_encoder_dim=20, d_model=128, encoder_output_dim=10, nhead=2, num_layers=2, deep_hidden_dims=[256, 128],
                 transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0,
                 batch_size=128, epochs=100):
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs)

        self.input_dim = input_dim // 2
        self.token_encoder_dim = token_encoder_dim
        self.d_model = d_model
        self.encoder_output_dim = encoder_output_dim

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
        self.deep_layers, self.output_layer = self._set_mlp_head(
            prev_dim=prev_dim,
            deep_hidden_dims=deep_hidden_dims,
            dropout_rate=dropout_rate
        )

        self._setup_optimizer_scheduler(learning_rate, weight_decay)

    def forward(self, x):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        encoded = self.encoder(x_i, x_j)
        return self.deep_layers(encoded).squeeze()