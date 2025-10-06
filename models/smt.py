from env.imports import *
from models.train_val import train_model
from models.smt_utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
from transformers import get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast

# === BASE TRANSFORMER MODEL CLASS === #
class BaseTransformerModel(nn.Module):
    def __init__(self, input_dim, learning_rate=0.0001, weight_decay=0.0001, batch_size=512, epochs=100, num_workers=2, prefetch_factor=4, 
    d_model=128, nhead=4, num_layers=4, token_encoder_dim=60, deep_hidden_dims=[512, 256, 128], aug_prob=0.0):
        super().__init__()
        # training
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.aug_prob = aug_prob
        self.criterion = nn.MSELoss()
        self.patience = 50 # default is 50
        self.scheduler = None
        
        # architecture
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.deep_hidden_dims = deep_hidden_dims
        self.store_attn = False
        self.token_encoder_dim = token_encoder_dim
        
    def _setup_optimizer_scheduler(self, learning_rate, weight_decay, patience=30, use_cosine=False):
        """Setup optimizer and scheduler - called by subclasses"""
        self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if use_cosine:
            training_samples = 120000
            # Calculate total steps and warmup steps based on rough number of training samples
            total_steps = int(self.epochs * training_samples / self.batch_size)
            warmup_steps = int(0.1 * total_steps)  # 10% warmup
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps)
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.3, patience=patience,
                threshold=0.1, cooldown=1, min_lr=1e-6, verbose=True)
    
    def _set_mlp_head(self, prev_dim, deep_hidden_dims, dropout_rate):
        """Dynamic MLP prediction head setup"""
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
            # 1. Set forward pass to collect weights instead of FlashAttention
            collect_full_attention_heads(self.encoder.transformer_layers)
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(loader):
                batch_X, batch_y, batch_coords, batch_expanded_idx = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
                batch_X = batch_X.to(self.device)
                batch_coords = batch_coords.to(self.device)
                batch_expanded_idx = batch_expanded_idx.to(self.device)
                
                if collect_attn:
                    out = self(batch_X, batch_coords, batch_expanded_idx)
                    if getattr(self, 'use_attention_pooling', False): # defaults to False
                        batch_preds = out["output"].cpu().numpy()
                        attns = (out["attn_beta_i"], out["attn_beta_j"])
                        all_attn.append(attns)
                    else:
                        # 2. Compute batch-wise average attention weights from last layer of transformer
                        batch_preds = out.cpu().numpy()
                        accumulate_attention_weights(self.encoder.transformer_layers, is_first_batch=(batch_idx == 0))
                        total_batches += 1
                else:
                    batch_preds = self(batch_X, batch_coords, batch_expanded_idx).cpu().numpy()
                
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
                # 3. Compute overall average attention weights over all batches
                avg_attn = process_full_attention_heads(self.encoder.transformer_layers, total_batches, save_attn_path, self.token_encoder_dim)
                return predictions, targets, avg_attn
        
        return predictions, targets

    def fit(self, dataset, train_indices, test_indices, save_model=None, verbose=True):
        """Shared fit function for all transformer models"""
        self.train_indices = train_indices
        self.test_indices = test_indices
        train_dataset = Subset(dataset, self.train_indices)
        test_dataset = Subset(dataset, self.test_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True,
                                num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True,
                               num_workers=self.num_workers, prefetch_factor=self.prefetch_factor)
        
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, scheduler=self.scheduler, save_model=save_model, verbose=verbose, dataset=dataset)


class FlashAttentionEncoder(nn.Module):
    def __init__(self, token_encoder_dim, d_model, output_dim=10, nhead=4, num_layers=4, dropout=0.1, use_alibi=False, use_attention_pooling=True):
        super().__init__()
        
        self.input_projection = nn.Linear(token_encoder_dim, d_model) # Bring from token chunk space to d_model space
        
        self.transformer_layers = nn.ModuleList([
            FlashAttentionBlock(d_model, nhead, dropout, use_alibi=use_alibi)
            for _ in range(num_layers)
        ])
        
        self.use_attention_pooling = use_attention_pooling
        if self.use_attention_pooling:
            self.pooling = AttentionPooling(d_model, hidden_dim=32)
        else:
            self.output_projection = nn.Linear(d_model, output_dim) # Bring from d_model space to output_dim space
        
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
            x = x.flatten(start_dim=1) # Flatten so transformer embedding information is retained but downsampled
            if return_attn:
                return x, None
            return x

class SharedSelfAttentionModel(BaseTransformerModel):
    def __init__(self, input_dim, token_encoder_dim=20, d_model=128, encoder_output_dim=10, nhead=4, num_layers=4, deep_hidden_dims=[512, 256, 128],
                 use_alibi=True, transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0,
                 batch_size=512, aug_prob=0.0, aug_style='linear_decay', epochs=100, num_workers=2, prefetch_factor=2):
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor,
                         d_model, nhead, num_layers, deep_hidden_dims, aug_prob)
        
        self.input_dim = input_dim // 2
        self.token_encoder_dim = token_encoder_dim
        self.encoder_output_dim = encoder_output_dim
        self.use_alibi = use_alibi
        self.use_attention_pooling = False
        self.num_tokens = self.input_dim // self.token_encoder_dim
        self.aug_style = aug_style
        
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

    def forward(self, x, coords, idx):
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

class SelfAttentionCLSEncoder(nn.Module):
    def __init__(self, token_encoder_dim, d_model, output_dim, nhead=4, num_layers=4, dropout=0.1, cls_init='spatial_learned', use_alibi=False, num_tokens=None, use_attention_pooling=True, cls_in_seq=True, cls_dropout=0.5):
        super().__init__()
        self.token_encoder_dim = token_encoder_dim
        self.d_model = d_model
        self.cls_init = cls_init
        self.cls_in_seq = cls_in_seq
        self.cls_dropout = cls_dropout
        self.use_alibi = use_alibi
        self.num_tokens = num_tokens
        self.use_attention_pooling = use_attention_pooling
        self.input_projection = nn.Linear(token_encoder_dim, d_model)
        
        if self.cls_in_seq:
            cls_coord_dim = 3  # x, y, z
            self.coord_to_cls = nn.Linear(cls_coord_dim, d_model)
        else:
            cls_coord_dim = 4  # x, y, z, distance
            self.coord_encoder = nn.Sequential(
                nn.Linear(cls_coord_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model))
            self.cls_dropout = nn.Dropout(cls_dropout)

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
            self.pooling = AttentionPooling(d_model, hidden_dim=32)
        else:
            self.output_projection = nn.Linear(d_model, output_dim)
    
    def replace_coords_with_remapped(self, coords):
        """Replace coordinates with randomized coordinates for ablation experiments"""
        B, _ = coords.shape
        match = (coords[:, None, :] == self.coord_ref_table[None, :, :]).all(dim=-1)
        idx = match.float().argmax(dim=1)
        remapped = self.coord_remap_table[idx]
        return remapped
    
    def forward(self, gene_exp, coords, dist_to_target=None, return_attn=False):
        B, T = gene_exp.shape
        x = gene_exp.view(B, -1, self.token_encoder_dim)
        x = self.input_projection(x)

        # (B, 4) or (B, 3)
        coords_input = torch.cat([coords, dist_to_target], dim=-1) if dist_to_target is not None else coords

        if self.cls_init == 'random_learned': # initialize CLS token as random learned aggregation token
            coords_input[:, :3] = self.replace_coords_with_remapped(coords_input[:, :3])
        
        if self.cls_in_seq:
            cls_token = self.coord_to_cls(coords_input).unsqueeze(1) # linear projection to d_model space
            x = torch.cat([cls_token, x], dim=1)

        for layer in self.transformer_layers:
            x = layer(x)

        if not self.cls_in_seq:
            cls_token = self.coord_encoder(coords_input).unsqueeze(1) # 2-layer MLP to d_model space
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
                 batch_size=128, epochs=100, aug_prob=0.0, aug_style='linear_decay', num_workers=2, prefetch_factor=2):
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor,
                         d_model, nhead, num_layers, deep_hidden_dims, aug_prob)
        
        self.input_dim = input_dim // 2
        self.token_encoder_dim = token_encoder_dim 
        self.encoder_output_dim = encoder_output_dim
        self.include_coords = True
        self.cls_init = cls_init
        self.cls_in_seq = cls_in_seq
        self.use_alibi = use_alibi
        self.aug_style = aug_style
        self.transformer_dropout = transformer_dropout
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
            cls_in_seq=self.cls_in_seq,
            use_alibi=self.use_alibi,
            use_attention_pooling=False
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

    def forward(self, x, coords, idx):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        coords_i, coords_j = torch.chunk(coords, chunks=2, dim=1)
        x_i = self.encoder(x_i, coords_i)
        x_j = self.encoder(x_j, coords_j)
        x = torch.cat([x_i, x_j], dim=1)
        y_pred = self.output_layer(self.deep_layers(x))
        return y_pred.squeeze()

class SharedSelfAttentionPoolingModel(BaseTransformerModel):
    def __init__(self, input_dim, token_encoder_dim=20, d_model=128, encoder_output_dim=10, nhead=2, num_layers=2, deep_hidden_dims=[256, 128],
                 use_alibi=False, transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0,
                 batch_size=256, aug_prob=0.0, epochs=100, num_workers=2, prefetch_factor=2):
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor,
                         d_model, nhead, num_layers, deep_hidden_dims, aug_prob)
        
        self.input_dim = input_dim // 2
        self.token_encoder_dim = token_encoder_dim
        self.encoder_output_dim = encoder_output_dim
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
            dropout_rate=dropout_rate)
        
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in SMT model: {num_params}")
        
        self._setup_optimizer_scheduler(learning_rate, weight_decay)

    def forward(self, x, coords, idx):
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

class SharedSelfAttentionCLSPoolingModel(BaseTransformerModel):
    def __init__(self, input_dim, token_encoder_dim=20, d_model=128, encoder_output_dim=10, nhead=2, num_layers=2,
                 deep_hidden_dims=[512, 256, 128], cls_init='spatial_learned', cls_in_seq=False, cls_dropout=0.3, use_alibi=False,
                 transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0, 
                 batch_size=128, epochs=100, aug_prob=0.0, num_workers=2, prefetch_factor=2):
        
        super().__init__(input_dim, learning_rate, weight_decay, batch_size, epochs, num_workers, prefetch_factor,
                         d_model, nhead, num_layers, deep_hidden_dims, aug_prob)
        
        self.input_dim = input_dim // 2
        self.token_encoder_dim = token_encoder_dim 
        self.encoder_output_dim = encoder_output_dim
        self.transformer_dropout = transformer_dropout
        self.dropout_rate = dropout_rate
        self.include_coords = True
        self.cls_init = cls_init
        self.cls_in_seq = cls_in_seq
        self.cls_dropout = cls_dropout
        self.use_alibi = use_alibi
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
            use_attention_pooling=self.use_attention_pooling,
            cls_in_seq=self.cls_in_seq,
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

    def forward(self, x, coords, idx):
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

        if self.store_attn:
            return {
                "output": y_pred.squeeze(),
                "attn_beta_i": attn_beta_i,
                "attn_beta_j": attn_beta_j
            }

        return y_pred.squeeze()