from env.imports import *
from data.data_utils import create_data_loader
from models.train_val import train_model
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
from torch.cuda.amp import autocast


def scaled_dot_product_attention_with_weights(query, key, value, dropout_p=0.0, is_causal=False, scale=None):
    '''
    Helper function to compute attention output and weights at inference
    '''
    # similar to https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention for inference
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    weights = torch.softmax(scores, dim=-1)
    weights = F.dropout(weights, p=dropout_p, training=True)
    output = torch.matmul(weights, value)
    return output, weights

def plot_avg_attention(avg_attn):
    '''
    Helper function to plot average attention weights
    '''
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

# === FAST TRANSFORMER IMPLEMENTATION === #
class FastSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, use_alibi=False):
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

        # Store attention weights
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

            # Project QKV jointly
            qkv = self.qkv_proj(x)  # (B, L, 3 * d_model)
            if self.store_attn:
                q, k, v = qkv.chunk(3, dim=-1)
                q = self.split_heads(q)  # (B, nhead, L, head_dim)
                k = self.split_heads(k)
                v = self.split_heads(v)
                # Manual scaled dot-product attention
                attn_output, attn_weights = scaled_dot_product_attention_with_weights(q, k, v, dropout_p=0.0, is_causal=False)
                self.last_attn_weights = attn_weights.detach().cpu()
                attn_output = self.merge_heads(attn_output)
            else:
                qkv = qkv.view(x.size(0), x.size(1), 3, self.nhead, self.head_dim)  # (B, L, 3, nhead, head_dim)
                attn_output = flash_attn_qkvpacked_func(
                    qkv,
                    dropout_p=0.0,
                    causal=False,
                    alibi_slopes=self.alibi_slopes if self.use_alibi else None
                )
                # ALiBi indicates positionality in the transformer by simply modifying the attention mechanism,
                # biasing the attention mechanism to have words that are away from each other interact less than words that are nearby.
                attn_output = attn_output.transpose(1, 2)  # (B, nhead, L, head_dim)
                attn_output = self.merge_heads(attn_output)
            attn_output = self.attn_dropout(attn_output)

            x = self.attn_norm(residual + attn_output)
            residual = x

            x = self.ffn(x)
            x = self.ffn_norm(residual + x)

            return x

class FastSelfAttentionEncoder(nn.Module):
    def __init__(self, token_encoder_dim, d_model, output_dim, nhead=4, num_layers=4, dropout=0.1, use_alibi=False):
        super().__init__()
        self.input_projection = nn.Linear(token_encoder_dim, d_model)
        self.layers = nn.ModuleList([
            FastSelfAttentionBlock(d_model, nhead, dropout, use_alibi=use_alibi)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(self, x):
        batch_size, total_features = x.shape
        L = total_features // self.input_projection.in_features
        x = x.view(batch_size, L, -1)
        x = self.input_projection(x)

        for layer in self.layers:
            x = layer(x)

        x = self.output_projection(x)
        x = x.flatten(start_dim=1)
        return x

class SharedSelfAttentionModel(nn.Module): # true name FastSharedSelfAttentionModel
    def __init__(self, input_dim, binarize=False, token_encoder_dim=20, d_model=128, encoder_output_dim=10, nhead=2, num_layers=2, deep_hidden_dims=[256, 128],
                 use_alibi=False, transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0,
                 batch_size=256, aug_prob=0.0, epochs=100):
        super().__init__()
        
        self.binarize = binarize
        self.input_dim = input_dim // 2
        self.token_encoder_dim = token_encoder_dim
        self.d_model = d_model
        self.encoder_output_dim = encoder_output_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.aug_prob = aug_prob
        self.use_alibi = use_alibi
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = FastSelfAttentionEncoder(
            token_encoder_dim=self.token_encoder_dim,
            d_model=self.d_model,
            output_dim=self.encoder_output_dim,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=transformer_dropout,
            use_alibi=self.use_alibi
        )
        self.encoder = torch.compile(self.encoder)

        prev_dim = (self.input_dim // self.token_encoder_dim * self.encoder_output_dim) * 2
        deep_layers = []
        for hidden_dim in deep_hidden_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim, dtype=torch.float32),
                nn.Dropout(dropout_rate)])
            prev_dim = hidden_dim
        self.deep_layers = nn.Sequential(*deep_layers)
        self.output_layer = nn.Linear(prev_dim, 1)

        self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.patience = 25
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.3,
            patience=20,
            threshold=0.1,
            cooldown=1,
            min_lr=1e-6,
            verbose=True
        )

    def forward(self, x):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        encoded_i = self.encoder(x_i)
        encoded_j = self.encoder(x_j)
        concatenated_embedding = torch.cat((encoded_i, encoded_j), dim=1)
        deep_output = self.deep_layers(concatenated_embedding)
        output = self.output_layer(deep_output)
        return output.squeeze()

    def predict(self, loader, collect_attn=False, save_attn_path=None):
        self.eval()
        predictions = []
        targets = []

        if collect_attn:
            for layer in self.encoder.layers:
                layer.store_attn = True
            avg_attn = None
            total_batches = 0

        with torch.no_grad():
            for batch_X, batch_y, batch_coords, _ in loader:
                batch_X = batch_X.to(self.device)
                batch_preds = self(batch_X).cpu().numpy()
                predictions.append(batch_preds)
                targets.append(batch_y.numpy())

                if collect_attn:
                    attn_weights = self.encoder.layers[-1].last_attn_weights
                    if attn_weights is not None:
                        # Mean across batch
                        batch_avg = attn_weights.mean(dim=0)  # (nhead, L, L)
                        if avg_attn is None:
                            avg_attn = batch_avg
                        else:
                            avg_attn += batch_avg
                        total_batches += 1

        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)

        if collect_attn and total_batches > 0:
            avg_attn /= total_batches  # Final average (nhead, L, L)
            plot_avg_attention(avg_attn.cpu())
            if save_attn_path is not None:
                np.save(save_attn_path, avg_attn.cpu().numpy())

        return ((predictions > 0.5).astype(int) if self.binarize else predictions), targets

    def fit(self, dataset, train_indices, test_indices, verbose=True):
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, self.scheduler, verbose=verbose, dataset=dataset)


# === SelfAttentionCLSEncoder using FastSelfAttentionBlock, FlashAttention, and CLS token === # 
class SelfAttentionCLSEncoder(nn.Module):    
    def __init__(self, token_encoder_dim, d_model, output_dim, nhead=4, num_layers=4, dropout=0.1, use_positional_encoding=False, cls_init='spatial_learned', use_alibi=False):
        super().__init__()
        self.token_encoder_dim = token_encoder_dim
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding
        self.cls_init = cls_init
        self.use_alibi = use_alibi
        
        # Token projection
        self.input_projection = nn.Linear(token_encoder_dim, d_model)
        self.coord_to_cls = nn.Linear(3, d_model)
        
        @staticmethod
        def build_coord_remap_table(coord_tensor, seed=42, scale=100.0):
            '''
            Helper function for hash table to initialize coordinate vector randomly in deteterministic way
            '''
            torch.manual_seed(seed)
            remap = (torch.rand_like(coord_tensor) * 2 - 1) * scale  # [-100, 100]
            return remap
        
        # CLS token randomization/projection freeze
        if self.cls_init == 'random_learned':
            df = pd.read_csv('./data/UKBB/atlas-4S456Parcels_dseg_reformatted.csv')
            region_coords = torch.tensor(df.iloc[:, -3:].values, dtype=torch.float32)
            remap_tensor = self.build_coord_remap_table(region_coords)
            self.register_buffer("coord_ref_table", region_coords)
            self.register_buffer("coord_remap_table", remap_tensor)
        elif self.cls_init == 'spatial_fixed':
            self.coord_to_cls.weight.requires_grad = False
            self.coord_to_cls.bias.requires_grad = False            

        # Transformer layers
        self.layers = nn.ModuleList([
            FastSelfAttentionBlock(d_model, nhead, dropout, use_alibi=self.use_alibi)
            for _ in range(num_layers)
        ])

        # Register per-head ALiBi slopes once if use_alibi is True
        if self.use_alibi:
            slopes = FastSelfAttentionBlock.build_alibi_slopes(nhead)
            self.register_buffer("alibi_slopes", slopes)
            for layer in self.layers:
                layer.alibi_slopes = self.alibi_slopes

        self.output_projection = nn.Linear(d_model, output_dim)
    
    def replace_coords_with_remapped(self, coords):
        '''
        Helper function to replace true coordinates with randomized remapped coordinates
        '''
        B, _ = coords.shape
        match = (coords[:, None, :] == self.coord_ref_table[None, :, :]).all(dim=-1)
        idx = match.float().argmax(dim=1)
        remapped = self.coord_remap_table[idx]
        return remapped  # shape (B, 3)
    
    def forward(self, gene_exp, coords):
        batch_size, total_features = gene_exp.shape
        L = total_features // self.token_encoder_dim

        gene_exp = gene_exp.view(batch_size, L, self.token_encoder_dim)
        x_proj = self.input_projection(gene_exp)  # (B, L, d_model)

        if self.cls_init == 'random_learned': # Optionally replace true coordinates with randomized - like a summary token
            coords = self.replace_coords_with_remapped(coords)  # (B, 3)

        cls_token = self.coord_to_cls(coords).unsqueeze(1)  # (B, 1, d_model)
        x = torch.cat([cls_token, x_proj], dim=1)  # (B, L+1, d_model)

        for layer in self.layers:
            x = layer(x)

        x = self.output_projection(x)
        x = x.reshape(batch_size, -1)  # flatten all tokens including CLS
        # x = x[:, 0]  # return just the CLS token
        return x


class SharedSelfAttentionCLSModel(nn.Module):
    def __init__(self, input_dim, binarize=False, token_encoder_dim=20, d_model=128, encoder_output_dim=10, nhead=2, num_layers=2, deep_hidden_dims=[256, 128], 
                 use_positional_encoding=False, cls_init='spatial_learned', use_alibi=False, transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0, 
                 batch_size=128, epochs=100, aug_prob=0.0):
        super().__init__()
        
        self.binarize = binarize 
        self.include_coords = True
        self.cls_init = cls_init # 'random_learned' 'spatial_fixed' 'spatial_learned' 
        self.aug_prob = aug_prob
        self.use_alibi = use_alibi
        # Transformer parameters
        self.input_dim = input_dim // 2
        self.token_encoder_dim = token_encoder_dim 
        self.d_model = d_model
        self.encoder_output_dim = encoder_output_dim
        self.transformer_dropout = transformer_dropout
        self.use_positional_encoding = use_positional_encoding
        self.nhead = nhead
        self.num_layers = num_layers

        # Deep layers parameters
        self.deep_hidden_dims = deep_hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create self-attention encoder
        self.encoder = SelfAttentionCLSEncoder(
                                            token_encoder_dim=self.token_encoder_dim,
                                            d_model=self.d_model,
                                            output_dim=self.encoder_output_dim, 
                                            nhead=self.nhead, 
                                            num_layers=self.num_layers,
                                            dropout=self.transformer_dropout,
                                            use_positional_encoding=self.use_positional_encoding, 
                                            cls_init=self.cls_init,
                                            use_alibi=self.use_alibi
                                            )
        self.encoder = torch.compile(self.encoder)
        
        # Use full sequence
        prev_dim = (self.input_dim // self.token_encoder_dim * self.encoder_output_dim) * 2 + 2 * self.encoder_output_dim # Concatenated outputs of encoder
        # Use CLS token only 
        # prev_dim = self.encoder_output_dim * 2 # Concatenated outputs of encoder CLS token only or mean pooled

        deep_layers = [] # Deep layers for concatenated outputs
        for hidden_dim in self.deep_hidden_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim, dtype=torch.float32), 
                nn.Dropout(dropout_rate)])
            prev_dim = hidden_dim
        self.deep_layers = nn.Sequential(*deep_layers)
        self.output_layer = nn.Linear(prev_dim, 1)
        
        self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.patience = 25
        self.scheduler = ReduceLROnPlateau( 
            self.optimizer, 
            mode='min', 
            factor=0.3,  # Reduce LR by 70%
            patience=20,  # Reduce LR after patientce epochs of no improvement
            threshold=0.1,  # Threshold to detect stagnation
            cooldown=1,  # Reduce cooldown period
            min_lr=1e-6,  # Prevent LR from going too low
            verbose=True
        )

    def forward(self, x, coords):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        coords_i, coords_j = torch.chunk(coords, chunks=2, dim=1)

        encoded_i = self.encoder(x_i, coords_i)
        encoded_j = self.encoder(x_j, coords_j)
        
        pairwise_embedding = torch.cat((encoded_i, encoded_j), dim=1)

        deep_output = self.deep_layers(pairwise_embedding)
        output = self.output_layer(deep_output)
        
        return output.squeeze()
        
    def predict(self, loader, collect_attn=False, save_attn_path=None):
        self.eval()
        predictions = []
        targets = []

        if collect_attn:
            for layer in self.encoder.layers:
                layer.store_attn = True
            avg_attn = None
            total_batches = 0

        with torch.no_grad():
            for batch_X, batch_y, batch_coords, _ in loader:
                batch_X = batch_X.to(self.device)
                batch_coords = batch_coords.to(self.device)
                batch_preds = self(batch_X, batch_coords).cpu().numpy()
                predictions.append(batch_preds)
                targets.append(batch_y.numpy())

                if collect_attn:
                    attn_weights = self.encoder.layers[-1].last_attn_weights
                    if attn_weights is not None:
                        # Mean across batch
                        batch_avg = attn_weights.mean(dim=0)  # (nhead, L, L)
                        if avg_attn is None:
                            avg_attn = batch_avg
                        else:
                            avg_attn += batch_avg
                        total_batches += 1

        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)

        if collect_attn and total_batches > 0:
            avg_attn /= total_batches  # Final average (nhead, L, L)
            plot_avg_attention(avg_attn.cpu())
            if save_attn_path is not None:
                np.save(save_attn_path, avg_attn.cpu().numpy())

        return ((predictions > 0.5).astype(int) if self.binarize else predictions), targets
    
    def fit(self, dataset, train_indices, test_indices, verbose=True):
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, self.scheduler, verbose=verbose, dataset=dataset)


# === CrossAttentionEncoder ===
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

            # Pytorch SDPA implementation - will use flash attention backend if available
            '''
            q = self.split_heads(q)
            k = self.split_heads(k)
            v = self.split_heads(v)
            attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            attn_output = self.merge_heads(attn_output)
            attn_output = self.attn_dropout(attn_output)
            '''

            # FlashAttention implementation - expects (B, L, nhead, head_dim)
            q = self.split_heads(q).transpose(1, 2)  # (B, L, nhead, head_dim)
            k = self.split_heads(k).transpose(1, 2)
            v = self.split_heads(v).transpose(1, 2)
            attn_output = flash_attn_func(q, k, v, dropout_p=0.0, causal=False) # , window_size=(128, 128)) # experiment with window size
            attn_output = attn_output.transpose(1, 2)  # (B, nhead, L, head_dim)        
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

# Update CrossAttentionModel to pass num_layers to encoder
class CrossAttentionModel(nn.Module):
    def __init__(self, input_dim, binarize=False, token_encoder_dim=20, d_model=128, encoder_output_dim=10, nhead=2, num_layers=2, deep_hidden_dims=[256, 128],
                 use_positional_encoding=False, transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0,
                 batch_size=128, epochs=100):

        super().__init__()

        self.binarize = binarize
        self.input_dim = input_dim // 2
        self.token_encoder_dim = token_encoder_dim
        self.d_model = d_model
        self.encoder_output_dim = encoder_output_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.patience = 20
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.3,
            patience=20,
            threshold=0.1,
            cooldown=1,
            min_lr=1e-6,
            verbose=True
        )

    def forward(self, x):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        encoded = self.encoder(x_i, x_j)
        return self.deep_layers(encoded).squeeze()

    def predict(self, loader):
        self.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for batch_X, batch_y, _, _ in loader:
                batch_X = batch_X.to(self.device)
                batch_preds = self(batch_X).cpu().numpy()
                predictions.append(batch_preds)
                targets.append(batch_y.numpy())
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        return ((predictions > 0.5).astype(int) if self.binarize else predictions), targets

    def fit(self, dataset, train_indices, test_indices, verbose=True):
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, self.scheduler, verbose=verbose, dataset=dataset)