from env.imports import *
from data.data_utils import create_data_loader
from models.train_val import train_model


class SelfAttentionEncoder(nn.Module):
    def __init__(self, token_encoder_dim, d_model, output_dim, nhead=4, num_layers=4, dropout=0.1):
        super(SelfAttentionEncoder, self).__init__()
        self.token_encoder_dim = token_encoder_dim
        self.d_model = d_model

        self.input_projection = nn.Linear(token_encoder_dim, d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            batch_first=True, 
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward= 4 * d_model,
            dropout=dropout
        )

        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim) # Linear output layer (after transformer)

    def forward(self, x): 
        batch_size, total_features = x.shape
        gene_exp = x
        
        L = gene_exp.shape[1] // self.token_encoder_dim # number of tokens
        gene_exp = gene_exp.view(batch_size, L, self.token_encoder_dim) # (batch_size, seq len, token_encoder_dim)
        x_proj = self.input_projection(gene_exp)  # (batch_size, L, d_model)

        x_enc = self.transformer(x_proj)
        
        x_enc = self.fc(x_enc)
        x_enc = x_enc.reshape(batch_size, -1) # flatten for downstream tasks

        return x_enc


class SharedSelfAttentionModel(nn.Module):
    def __init__(self, input_dim, binarize=False, token_encoder_dim=20, d_model=128, encoder_output_dim=10, nhead=2, num_layers=2, deep_hidden_dims=[256, 128], 
                 use_positional_encoding=False, transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0, lambda_reg=0.0, 
                 batch_size=128, epochs=100):
        super().__init__()

        self.binarize=binarize 
        
        # Transformer parameters
        self.input_dim = input_dim // 2
        self.token_encoder_dim = token_encoder_dim 
        self.d_model = d_model # increase for more complex embeddings
        self.encoder_output_dim = encoder_output_dim
        self.transformer_dropout = transformer_dropout
        self.nhead = nhead
        self.num_layers = num_layers # increase for more complex interactions

        # Deep layers parameters
        self.deep_hidden_dims = deep_hidden_dims
        self.dropout_rate = dropout_rate
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create self-attention encoder
        self.encoder = SelfAttentionEncoder(token_encoder_dim=self.token_encoder_dim,
                                            d_model=self.d_model,
                                            output_dim=self.encoder_output_dim, 
                                            nhead=self.nhead, 
                                            num_layers=self.num_layers,
                                            dropout=self.transformer_dropout)

        prev_dim = (self.input_dim // self.token_encoder_dim * self.encoder_output_dim) * 2 # Concatenated outputs of encoder
    
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
        
        if torch.cuda.device_count() > 1: # Wrap the model with DataParallel if multiple GPUs are available
            self.encoder = nn.DataParallel(self.encoder)
            self.deep_layers = nn.DataParallel(self.deep_layers)
            self.output_layer = nn.DataParallel(self.output_layer)
                
        self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)

        encoded_i = self.encoder(x_i)
        encoded_j = self.encoder(x_j)
        
        concatenated_embedding = torch.cat((encoded_i, encoded_j), dim=1)

        deep_output = self.deep_layers(concatenated_embedding)
        output = self.output_layer(deep_output)

        return output.squeeze()
    
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
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, self.scheduler, verbose=verbose)




class SelfAttentionCLSEncoder(nn.Module):
    def __init__(self, token_encoder_dim, d_model, output_dim, nhead=4, num_layers=4, dropout=0.1, use_positional_encoding=False):
        super(SelfAttentionCLSEncoder, self).__init__()
        
        self.token_encoder_dim = token_encoder_dim
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding

        self.input_projection = nn.Linear(token_encoder_dim, d_model)
        self.coord_to_cls = nn.Linear(3, d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            batch_first=True, 
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward= 4 * d_model,
            dropout=dropout
        )

        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, gene_exp, coords): 
        batch_size, total_features = gene_exp.shape

        L = gene_exp.shape[1] // self.token_encoder_dim # number of tokens
        gene_exp = gene_exp.view(batch_size, L, self.token_encoder_dim) # (batch_size, seq len, token_encoder_dim)
        x_proj = self.input_projection(gene_exp)  # (batch_size, L, d_model)

        cls_vec = self.coord_to_cls(coords).unsqueeze(1)  # (batch_size, 1, d_model)
        x_proj = torch.cat([cls_vec, x_proj], dim=1)      # (batch_size, L+1, d_model)

        if self.use_positional_encoding:
            x_proj = self.add_positional_encoding(x_proj)

        x_enc = self.transformer(x_proj)

        # x_enc = x_enc[:, 0, :] # grab the context vector
        # x_enc = x_enc.mean(dim=1)  # mean pool across sequence length
        
        x_enc = self.fc(x_enc)
        x_enc = x_enc.reshape(batch_size, -1) # flatten for downstream tasks

        return x_enc

    def add_positional_encoding(self, x): # REVISE
        seq_length = x.size(1)
        position = torch.arange(seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_length, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(x.device)  # Add batch dimension and move to device
        return x + pe


class SharedSelfAttentionCLSModel(nn.Module):
    def __init__(self, input_dim, binarize=False, token_encoder_dim=20, d_model=128, encoder_output_dim=10, nhead=2, num_layers=2, deep_hidden_dims=[256, 128], 
                 use_positional_encoding=False, transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0, lambda_reg=0.0, 
                 batch_size=128, epochs=100):
        super().__init__()
        
        self.binarize=binarize 
        self.include_coords = True

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
        self.lambda_reg = lambda_reg
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
                                            use_positional_encoding=self.use_positional_encoding)

        
        #prev_dim = self.d_model # if adding CLS tokens
        # prev_dim = self.d_model * 2 # Concatenated outputs of encoder CLS token only or mean pooled
        # print(f"input_dim: {self.input_dim}")
        # print(f"token_encoder_dim: {self.token_encoder_dim}")
        # print(f"encoder_output_dim: {self.encoder_output_dim}")
        prev_dim = (self.input_dim // self.token_encoder_dim * self.encoder_output_dim) * 2 + 2 * self.encoder_output_dim # Concatenated outputs of encoder
        #print(f"prev_dim: {prev_dim}")

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
        
        if torch.cuda.device_count() > 1: # Wrap the model with DataParallel if multiple GPUs are available
            self.encoder = nn.DataParallel(self.encoder)
            self.deep_layers = nn.DataParallel(self.deep_layers)
            self.output_layer = nn.DataParallel(self.output_layer)
        
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
        # pairwise_embedding = encoded_i + encoded_j

        deep_output = self.deep_layers(pairwise_embedding)
        output = self.output_layer(deep_output)
        
        return output.squeeze()
        
    def predict(self, loader):
        self.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for batch_X, batch_y, batch_coords, batch_idx in loader:
                batch_X = batch_X.to(self.device)
                batch_coords = batch_coords.to(self.device)
                batch_preds = self(batch_X, batch_coords).cpu().numpy()
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
        
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, self.scheduler, verbose=verbose)
    
    '''
    def predict(self, X):
        """Make predictions in memory-efficient batches."""
        self.eval()
        X = torch.as_tensor(X, dtype=torch.float32)
        predictions = []
        
        # Create dataloader for batched prediction
        predict_loader = DataLoader(
            TensorDataset(X),
            batch_size=self.batch_size // 2, # reduce batch size by double to avoid memory issues
            shuffle=False,
            pin_memory=True  # More efficient GPU memory transfer
        )
        with torch.no_grad():
            for batch in predict_loader:
                # Move batch to device and make prediction
                batch = batch[0].to(self.device, non_blocking=True)
                batch_preds = self(batch)
                # Move predictions back to CPU and store
                predictions.append(batch_preds.cpu().numpy())
            torch.cuda.empty_cache()  # Clear unused memory on the GPU
    
        return np.concatenate(predictions, axis=0)

    def fit(self, X_train, y_train, X_test=None, y_test=None, verbose=True):
        train_loader = create_data_loader(X_train, y_train, self.batch_size, self.device)
        val_loader = None
        if X_test is not None and y_test is not None:
            val_loader = create_data_loader(X_test, y_test, self.batch_size, self.device)
        return train_model(self, train_loader, val_loader, self.epochs, self.criterion, self.optimizer, scheduler=None, verbose=verbose)
        '''


class CrossAttentionEncoder(nn.Module):
    # NEED TO REWORK THIS PROPERLY 
    def __init__(self, token_encoder_dim, d_model, output_dim, nhead=4, num_layers=4, dropout=0.1):
        super(CrossAttentionEncoder, self).__init__()
        self.token_encoder_dim = token_encoder_dim
        self.d_model = d_model

        self.input_projection = nn.Linear(token_encoder_dim, d_model)

        # self.encoder_layer = nn.TransformerEncoderLayer(
        #     batch_first=True, 
        #     d_model=d_model, 
        #     nhead=nhead,
        #     dim_feedforward=4 * d_model,
        #     dropout=dropout
        # )

        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

        self.fc = nn.Linear(d_model, output_dim)  # Linear output layer (after attention)

    def forward(self, x_i, x_j): 
        batch_size, total_features = x_i.shape
        
        L = total_features // self.token_encoder_dim
        x_i = x_i.view(batch_size, L, self.token_encoder_dim)  # (batch_size, seq_len, token_encoder_dim)
        x_j = x_j.view(batch_size, L, self.token_encoder_dim)  # (batch_size, seq_len, token_encoder_dim)

        x_i = self.input_projection(x_i)  # (batch_size, L, d_model)
        x_j = self.input_projection(x_j)  # (batch_size, L, d_model)

        # Perform cross-attention (x_i attends to x_j and vice versa)
        attn_output_i, _ = self.cross_attention(query=x_i, key=x_j, value=x_j)
        attn_output_j, _ = self.cross_attention(query=x_j, key=x_i, value=x_i)

        # Pass through linear layer without mean pooling
        x_enc_i = self.fc(attn_output_i)  # (batch_size, L, output_dim)
        x_enc_j = self.fc(attn_output_j)  # (batch_size, L, output_dim)
        # Flatten for downstream tasks
        x_enc_i = x_enc_i.reshape(batch_size, -1)  # (batch_size, L * output_dim)
        x_enc_j = x_enc_j.reshape(batch_size, -1)  # (batch_size, L * output_dim)

        return x_enc_i, x_enc_j

class CrossAttentionModel(nn.Module):
    def __init__(self, input_dim, binarize=False, token_encoder_dim=20, d_model=128, encoder_output_dim=10, nhead=2, num_layers=2, deep_hidden_dims=[256, 128], 
                  use_positional_encoding=False, transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0, lambda_reg=0.0, 
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
            dropout=dropout_rate
        )

        # prev_dim = self.encoder_output_dim  # Concatenated outputs of encoder
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
            factor=0.3,  # Reduce LR by 70%
            patience=20,  # Reduce LR after patientce epochs of no improvement
            threshold=0.1,  # Threshold to detect stagnation
            cooldown=1,  # Reduce cooldown period
            min_lr=1e-6,  # Prevent LR from going too low
            verbose=True
        )

    def forward(self, x):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        encoded_i, encoded_j = self.encoder(x_i, x_j)
        concatenated_embedding = encoded_i + encoded_j
        return self.deep_layers(concatenated_embedding).squeeze()
    
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
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, self.scheduler, verbose=verbose)
