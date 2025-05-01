from env.imports import *
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist


def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, patience=100, scheduler=None, verbose=True):
    train_history = {"train_loss": [], "val_loss": []}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} - Memory Allocated: {torch.cuda.memory_allocated(i)/1024**3:.2f} GB")
    
    cudnn.benchmark = True  # Auto-tune GPU kernels
    scaler = GradScaler()  # Enable FP16 training
    #scaler = None
    
    best_val_loss = float("inf")  # Track the best validation loss
    best_model_state = None  # Store the best model state
    patience_counter = 0  # Counts epochs without improvement

    for epoch in range(epochs):
        start_time = time.time() if (epoch + 1) % 5 == 0 else None

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        train_history["train_loss"].append(train_loss)

        if val_loader:
            val_loss = evaluate(model, val_loader, criterion, device, scheduler)
            train_history["val_loss"].append(val_loss)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Best val loss so far at epoch {epoch+1}: {best_val_loss:.4f}")
                best_model_state = model.state_dict()  # Save best model
                patience_counter = 0  # Reset counter if improvement
            else:
                patience_counter += 1  # Increment counter if no improvement

            if patience_counter >= patience or epoch == epochs - 1:
                model.load_state_dict(best_model_state)  # Rewind to best model
                try:
                    predictions, targets = model.predict(val_loader, collect_attn=False)
                except:
                    predictions, targets = model.predict(val_loader)
                pearson_corr = pearsonr(predictions, targets)[0]
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}. Restoring best model with Val Loss: {best_val_loss:.4f}, Pearson Correlation: {pearson_corr:.4f}")
                else:
                    print(f"\nReached final epoch {epoch+1}. Restoring best model with Val Loss: {best_val_loss:.4f}, Pearson Correlation: {pearson_corr:.4f}")
                break            
        
        if verbose and (epoch + 1) % 5 == 0:
            epoch_time = time.time() - start_time
            try: 
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")
            except: 
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Time: {epoch_time:.2f}s")

    return train_history

def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Combined training function for regular and mixed precision training"""
    model.train()
    total_train_loss = 0

    for batch_X, batch_y, batch_coords, batch_idx in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        '''
        # roll a dice at this point and replace batch_y with values from underlying population
        if np.random.rand() < 1/6:
            for expanded_idx in batch_idx:
                true_idx = model.region_pair_dataset.expanded_idx_to_true_pair[expanded_idx]
                # Reorder true idx pairs so smaller value always comes first
                true_idx = tuple(sorted(true_idx))
                print(true_idx)
        '''


            batch_y = batch_y.to(device)
        batch_coords = batch_coords.to(device)
        
        optimizer.zero_grad()

        if scaler is not None: # Mixed precision training path            
            with autocast(dtype=torch.bfloat16):
                if hasattr(model, 'include_coords'):
                    predictions = model(batch_X, batch_coords).squeeze()
                elif hasattr(model, 'optimize_encoder'):
                    predictions = model(batch_X, batch_idx).squeeze()
                else:
                    predictions = model(batch_X).squeeze()
                loss = criterion(predictions, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else: # Regular training path
            if hasattr(model, 'include_coords'):
                predictions = model(batch_X, batch_coords).squeeze()
            elif hasattr(model, 'optimize_encoder'):
                    predictions = model(batch_X, batch_idx).squeeze()
            else:
                predictions = model(batch_X).squeeze()
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
        
        total_train_loss += loss.item()
    
    return total_train_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device, scheduler=None):
    model.eval()
    total_val_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_X, batch_y, batch_coords, batch_idx in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_coords = batch_coords.to(device)

            if hasattr(model, 'include_coords'):
                predictions = model(batch_X, batch_coords).squeeze()
            elif hasattr(model, 'optimize_encoder'):
                predictions = model(batch_X, batch_idx).squeeze()
            else:
                predictions = model(batch_X).squeeze()
                
            val_loss = criterion(predictions, batch_y)            
            total_val_loss += val_loss.item()
            
            # integrate these eventually 
            if hasattr(model, 'binarize') and model.binarize:
                pred_labels = (torch.sigmoid(predictions) > 0.5).float()
            else:
                pred_labels = predictions.round()
            
            total_correct += (pred_labels == batch_y).sum().item()
            total_samples += batch_y.size(0)
    
    mean_val_loss = total_val_loss / len(val_loader)
    accuracy = total_correct / total_samples

    if scheduler is not None:
        prev_lr = scheduler.optimizer.param_groups[0]['lr']
        scheduler.step(mean_val_loss)
        new_lr = scheduler.optimizer.param_groups[0]['lr']
        if new_lr < prev_lr:
            print(f"\nLR REDUCED: {prev_lr:.6f} → {new_lr:.6f} at Val Loss: {mean_val_loss:.6f}")

    return mean_val_loss


'''
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
                 use_positional_encoding=False, transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0, 
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
                 use_positional_encoding=False, transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0, 
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