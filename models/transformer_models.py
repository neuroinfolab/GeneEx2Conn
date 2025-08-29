from env.imports import *
from data.data_utils import create_data_loader
from models.train_val import train_model
from models.transformer_utils import *
import torch
import torch.nn.functional as F
from torch.nn import RMSNorm
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
from torch.cuda.amp import autocast

# === CORE TRANSFORMER COMPONENTS === #
class AttentionPooling(nn.Module):
    # CellSpliceNet implementation
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.theta1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.theta2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_attn=False):
        B, L, D = x.shape
        scores = self.theta2(F.gelu(self.bn(self.theta1(x).transpose(1, 2)).transpose(1, 2)))
        attn_weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(attn_weights.transpose(1, 2), x).squeeze(1)
        
        if return_attn:
            return pooled, attn_weights.squeeze(-1)
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
                    alibi_slopes=self.alibi_slopes if self.use_alibi else None
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
        x = x.view(B, -1, self.input_projection.in_features) # Key chunking step
        
        x = self.input_projection(x)
        for layer in self.layers:
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
        self.patience = 35
        self.scheduler = None

    def _setup_optimizer_scheduler(self, learning_rate, weight_decay, patience=25):
        """Setup optimizer and scheduler - called by subclasses"""
        self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.3, patience=patience,
            threshold=0.1, cooldown=1, min_lr=1e-6, verbose=True
        )

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
            else:
                avg_attn = process_full_attention_heads(self.encoder.layers, total_batches, save_attn_path)
        
        return predictions, targets

    def _forward_with_coords(self, x, coords):
        """Default implementation for models without coords - override in subclasses"""
        return self(x)

    def fit(self, dataset, train_indices, test_indices, verbose=True):
        """Shared fit function for all transformer models"""
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                                num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True,
                               num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)
        
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, self.scheduler, verbose=verbose, dataset=dataset)

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
            num_tokens=self.num_tokens,
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
            num_tokens=self.num_tokens,
            use_attention_pooling=True,
        )
        self.encoder = torch.compile(self.encoder)
        
        prev_dim = self.d_model * 2
        
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
        
        if self.store_attn:
            return {"output": y_pred.squeeze(), "attn_beta_i": attn_beta_i, "attn_beta_j": attn_beta_j}
        
        return y_pred.squeeze()

class SelfAttentionCLSEncoder(nn.Module):    
    def __init__(self, token_encoder_dim, d_model, output_dim, nhead=4, num_layers=4, dropout=0.1, cls_init='spatial_learned', use_alibi=False, num_tokens=None, use_attention_pooling=True):
        super().__init__()
        self.token_encoder_dim = token_encoder_dim
        self.d_model = d_model
        self.cls_init = cls_init
        self.use_alibi = use_alibi
        self.num_tokens = num_tokens
        self.use_attention_pooling = use_attention_pooling
        
        self.input_projection = nn.Linear(token_encoder_dim, d_model)
        self.coord_to_cls = nn.Linear(3, d_model)
        
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
        elif self.cls_init == 'spatial_fixed':
            self.coord_to_cls.weight.requires_grad = False
            self.coord_to_cls.bias.requires_grad = False

        self.layers = nn.ModuleList([
            FlashAttentionBlock(d_model, nhead, dropout, use_alibi=self.use_alibi)
            for _ in range(num_layers)
        ])

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
        return remapped
    
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
            x = self.output_projection(x)
            x = x.flatten(start_dim=1)
            if return_attn:
                return x, None
            return x

class SharedSelfAttentionCLSModel(BaseTransformerModel):
    def __init__(self, input_dim, token_encoder_dim=20, d_model=128, encoder_output_dim=10, nhead=2, num_layers=2, deep_hidden_dims=[256, 128], 
                 cls_init='spatial_learned', use_alibi=False, transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0, 
                 batch_size=128, epochs=100, aug_prob=0.0, num_workers=2, prefetch_factor=2):
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor)
        
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
            num_tokens=self.num_tokens,
            use_attention_pooling=False,
        )
        self.encoder = torch.compile(self.encoder)

        prev_dim = (self.encoder_output_dim * (self.num_tokens + 1) * 2)
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
    def __init__(self, input_dim, token_encoder_dim=20, d_model=128, encoder_output_dim=10, nhead=2, num_layers=2, deep_hidden_dims=[256, 128], 
                 cls_init='spatial_learned', use_alibi=False, transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0, 
                 batch_size=128, epochs=100, aug_prob=0.0, num_workers=2, prefetch_factor=2):
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor)
        
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
        self.use_attention_pooling = True
        
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
            num_tokens=self.num_tokens,
            use_attention_pooling=True
        )
        self.encoder = torch.compile(self.encoder)

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
        
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in SMT w/ CLS pooled model: {num_params}")

        self._setup_optimizer_scheduler(learning_rate, weight_decay)

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
        if self.store_attn:
            return {"output": y_pred.squeeze(), "attn_beta_i": attn_beta_i, "attn_beta_j": attn_beta_j}
        return y_pred.squeeze()

    def _forward_with_coords(self, x, coords):
        return self.forward(x, coords)

class GeneConvEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, num_tokens=128,
                 kernel_size=32, stride=16):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_tokens = num_tokens

        # First layer: learn local co-expression motifs (~32 genes each)
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2
        )
        self.relu = nn.ReLU()

        # Project motifs → latent embedding dimension
        self.proj = nn.Conv1d(in_channels=64, out_channels=d_model, kernel_size=1)

        # Compress into exactly num_tokens tokens using adaptive pooling
        self.pool = nn.AdaptiveAvgPool1d(output_size=num_tokens)

    def forward(self, x):
        # Input: (B, g) → (B, 1, g)
        x = x.unsqueeze(1)

        # Local convolution → motif embeddings
        x = self.relu(self.conv1(x))     # (B, 64, L)

        # Project to latent space
        x = self.proj(x)                 # (B, d_model, L)

        # Compress into num_tokens positionally relevant tokens
        x = self.pool(x)                 # (B, d_model, num_tokens)

        return x.transpose(1, 2)         # (B, num_tokens, d_model)

class SharedSelfAttentionConvModel(BaseTransformerModel):
    def __init__(self, input_dim, d_model=128, num_tokens=128, nhead=4, num_layers=4,
                 deep_hidden_dims=[256, 128], use_alibi=False, transformer_dropout=0.1,
                 dropout_rate=0.1, learning_rate=1e-3, weight_decay=0.0,
                 batch_size=256, epochs=100, num_workers=2, prefetch_factor=2):
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor)

        self.input_dim = input_dim // 2
        self.d_model = d_model
        self.num_tokens = num_tokens
        self.use_attention_pooling = True
        
        # Convolutional encoder → outputs (B, num_tokens, d_model)
        self.conv_encoder = GeneConvEncoder(
            input_dim=self.input_dim,
            d_model=d_model,
            num_tokens=num_tokens
        )

        # FlashAttention encoder over tokens created by convolution
        self.encoder = FlashAttentionEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=transformer_dropout,
            use_alibi=use_alibi, 
            use_attention_pooling=True # this will do the pooling automatically 
        )
        self.encoder = torch.compile(self.encoder)

        # Fully connected regression head
        prev_dim = d_model * 2  # both region embeddings
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
        print(f"Number of learnable parameters in Conv+Attention model: {num_params}")

        self._setup_optimizer_scheduler(learning_rate, weight_decay)

    def forward(self, x):
        # Split into hemispheres
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)

        # Convert into structured tokens
        x_i = self.conv_encoder(x_i)  # (B, num_tokens, d_model)
        x_j = self.conv_encoder(x_j)

        # Process via FlashAttention blocks
        x_i = self.encoder(x_i)
        x_j = self.encoder(x_j)

        # Attention pooling
        if self.store_attn:
            x_i, attn_i = self.attn_pool(x_i, return_attn=True)
            x_j, attn_j = self.attn_pool(x_j, return_attn=True)
        else:
            x_i = self.attn_pool(x_i)
            x_j = self.attn_pool(x_j)

        # Concatenate pooled embeddings
        x = torch.cat([x_i, x_j], dim=1)

        # Deep regression head
        y_pred = self.output_layer(self.deep_layers(x))
        y_pred = torch.clamp(y_pred, min=-1.0, max=1.0).squeeze()

        if self.store_attn:
            return {"output": y_pred, "attn_i": attn_i, "attn_j": attn_j}
        return y_pred


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
        self.deep_layers = nn.Sequential(
            nn.Linear(prev_dim, deep_hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(deep_hidden_dims[0]),
            nn.Dropout(dropout_rate),
            nn.Linear(deep_hidden_dims[0], 1)
        )

        self._setup_optimizer_scheduler(learning_rate, weight_decay)

    def forward(self, x):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        encoded = self.encoder(x_i, x_j)
        return self.deep_layers(encoded).squeeze()