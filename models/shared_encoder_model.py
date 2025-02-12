# GeneEx2Conn/models/shared_encoder_model.py

from env.imports import *
from data.data_utils import create_data_loader
from models.train_val import train_model

class SelfAttentionEncoder(nn.Module):
    def __init__(self, input_dim, token_encoder_dim, d_model, output_dim, nhead=4, num_layers=4, dropout=0.1, use_positional_encoding=False, include_coords=False):
        super(SelfAttentionEncoder, self).__init__()
        
        self.token_encoder_dim = token_encoder_dim
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding
        self.include_coords = include_coords

        # Linear projection to d_model
        self.input_projection = nn.Linear(token_encoder_dim, d_model)
        self.coord_to_cls = nn.Linear(3, d_model) if self.include_coords else None

        self.encoder_layer = nn.TransformerEncoderLayer(
            batch_first=True, 
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout
        )

        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim) # Linear output layer (after transformer)

    def forward(self, x): 
        batch_size, total_features = x.shape
        
        if self.include_coords:
            coords = x[:, -3:]                     # (batch_size, 3)
            gene_exp = x[:, :-3]                   # (batch_size, gene_dim)
        else:
            gene_exp = x
        
        L = gene_exp.shape[1] // self.token_encoder_dim # number of tokens
        # print('L', L)
        # print('gene_exp shape', gene_exp.shape)
        gene_exp = gene_exp.view(batch_size, L, self.token_encoder_dim) # (batch_size, seq len, token_encoder_dim)
        # print('gene_exp shape', gene_exp.shape)
        x_proj = self.input_projection(gene_exp)  # (batch_size, L, d_model)
        # print('x_proj shape', x_proj.shape)

        if self.include_coords:
            cls_vec = self.coord_to_cls(coords).unsqueeze(1)  # (batch_size, 1, d_model)
            x_proj = torch.cat([cls_vec, x_proj], dim=1)      # (batch_size, L+1, d_model)

        if self.use_positional_encoding:
            x_proj = self.add_positional_encoding(x_proj)

        x_enc = self.transformer(x_proj)
        # print('x_enc shape post transformer', x_enc.shape)

        if self.include_coords:
            x_enc = x_enc[:, 0, :] # take the [CLS] token
        else:
            x_enc = self.fc(x_enc)
            x_enc = x_enc.reshape(batch_size, -1) # flatten for downstream tasks
        # print('x_enc shape post linear layer', x_enc.shape)

        return x_enc

    def add_positional_encoding(self, x):
        """
        Add positional encoding to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, d_model].
        Returns:
            torch.Tensor: Input tensor with positional encoding added.
        """
        seq_length = x.size(1)
        position = torch.arange(seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_length, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(x.device)  # Add batch dimension and move to device
        return x + pe

class SharedSelfAttentionModel(nn.Module):
    def __init__(self, input_dim, include_coords, token_encoder_dim=10, d_model=128, encoder_output_dim=1, nhead=2, num_layers=2, deep_hidden_dims=[256, 128], 
                 use_positional_encoding=False, transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0, lambda_reg=0.0, 
                 batch_size=128, epochs=100):
        super().__init__()

        # Transformer parameters
        self.input_dim = input_dim // 2
        self.token_encoder_dim = token_encoder_dim 
        self.d_model = d_model
        self.encoder_output_dim = encoder_output_dim
        self.transformer_dropout = transformer_dropout
        self.use_positional_encoding = use_positional_encoding
        self.nhead = nhead
        self.num_layers = num_layers
        self.include_coords = include_coords

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
        self.encoder = SelfAttentionEncoder(input_dim=self.input_dim, # change this
                                            token_encoder_dim=self.token_encoder_dim,
                                            d_model=self.d_model,
                                            output_dim=self.encoder_output_dim, 
                                            nhead=self.nhead, 
                                            num_layers=self.num_layers,
                                            include_coords=self.include_coords,
                                            dropout=self.transformer_dropout,
                                            use_positional_encoding=self.use_positional_encoding)

        if self.include_coords: # If coords => the encoder returns shape (batch_size, d_model) per region
            prev_dim = self.d_model * 2 # Concatenated outputs of encoder CLS token only
        else:
            # Dynamically set first layer size based on transformer operation
            prev_dim = (self.input_dim // self.token_encoder_dim * self.encoder_output_dim) * 2 # Concatenated outputs of encoder

        # Deep layers for concatenated outputs
        deep_layers = []
        for hidden_dim in self.deep_hidden_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)])
            prev_dim = hidden_dim
        self.deep_layers = nn.Sequential(*deep_layers)
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # Wrap the model with DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            self.encoder = nn.DataParallel(self.encoder)
            self.deep_layers = nn.DataParallel(self.deep_layers)
            self.output_layer = nn.DataParallel(self.output_layer)
        
        # Optimizer and loss function setup
        self.optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        """
        Forward pass for the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 2 * input_dim].
            
        Returns:
            torch.Tensor: Output predictions of shape [batch_size, 1].
        """
        # Split input into region i and region j
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        
        # Encode both regions using the self-attention encoder
        encoded_i = self.encoder(x_i)
        encoded_j = self.encoder(x_j)
        # print('encoded_i shape', encoded_i.shape)
        # print('encoded_j shape', encoded_j.shape)
        
        # Concatenate encoded outputs and pass through deep layers
        concatenated_embedding = torch.cat((encoded_i, encoded_j), dim=1)
        # print('concatenated_embedding shape', concatenated_embedding.shape)

        deep_output = self.deep_layers(concatenated_embedding)
        # print('deep_output shape', deep_output.shape)
        
        output = self.output_layer(deep_output)
        # print('output shape', output.shape)
        return output.squeeze()

    def get_params(self):
        """Get parameters for saving and model tuning."""
        params = {
            'input_dim': self.input_dim,
            'token_encoder_dim': self.token_encoder_dim,
            'd_model': self.d_model,
            'encoder_output_dim': self.encoder_output_dim,
            'use_positional_encoding': self.use_positional_encoding,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'deep_hidden_dims': self.deep_hidden_dims,
            'transformer_dropout': self.transformer_dropout,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'lambda_reg': self.lambda_reg,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }
        return params

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
        """Train the model."""
        train_loader = create_data_loader(X_train, y_train, self.batch_size, self.device)
        val_loader = None
        if X_test is not None and y_test is not None:
            val_loader = create_data_loader(X_test, y_test, self.batch_size, self.device)
        return train_model(self, train_loader, val_loader, self.epochs, self.criterion, self.optimizer, scheduler=None, verbose=verbose)




class SparseEncoderLoss(nn.Module):
    def __init__(self, base_loss=nn.MSELoss(), lambda_reg=0.1):
        """
        Custom loss function that combines a base loss with L1 regularization
        on the MLPEncoder parameters to encourage sparsity.

        Args:
            base_loss (nn.Module): The base loss function (e.g., MSELoss).
            lambda_reg (float): Regularization strength for L1 penalty.
        """
        super().__init__()
        self.base_loss = base_loss
        self.lambda_reg = lambda_reg

    def forward(self, predictions, targets, model):
        # Calculate the base loss
        loss = self.base_loss(predictions, targets)
        
        encoder = model.encoder
        # Apply L1 regularization only to the encoder's parameters
        l1_reg = 0
        for param in encoder.parameters():
            if param.requires_grad:
                l1_reg += torch.sum(torch.abs(param))
        
        # Combine the base loss with the L1 regularization term
        total_loss = loss + self.lambda_reg * l1_reg
        return total_loss


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        A simple MLP-based encoder for gene expression data.

        Args:
            input_dim (int): Number of input features (genes).
            hidden_dim (int): Number of hidden units in the first layer.
            output_dim (int): Number of output features from the encoder.
            dropout_rate (float): Dropout probability.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class SharedMLPEncoderModel(nn.Module):
    def __init__(self, input_dim, encoder_hidden_dim=64, encoder_output_dim=32, use_bilinear=True, deep_hidden_dims=[64], dropout_rate=0.0, learning_rate=0.01, weight_decay=0.0, lambda_reg=1.0, batch_size=256, epochs=100):
        """
        A model that uses a shared encoder and a bilinear layer for interactions.
        Args:
            input_dim (int): Number of input features (genes).
            encoder_hidden_dim (int): Number of hidden units in the first layer of the encoder.
            encoder_output_dim (int): Number of output features from the encoder.
            use_bilinear (bool): Whether to use a bilinear layer for interactions.
            deep_hidden_dims (list): List of hidden layer sizes for additional processing.
            dropout_rate (float): Dropout probability. 
            lambda_reg (float): Regularization strength for L1 penalty.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            batch_size (int): Batch size for training.
            epochs (int): Number of epochs to train for.
        """
        super().__init__()
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder_hidden_dim=encoder_hidden_dim
        self.encoder_output_dim=encoder_output_dim
        self.use_bilinear=use_bilinear
        self.deep_hidden_dims=deep_hidden_dims
        self.dropout_rate=dropout_rate
        self.lambda_reg=lambda_reg

        self.encoder = MLPEncoder(input_dim=input_dim//2, hidden_dim=encoder_hidden_dim, output_dim=encoder_output_dim)
        self.use_bilinear = use_bilinear
        
        if self.use_bilinear:
            self.bilinear = nn.Bilinear(encoder_output_dim, encoder_output_dim, 1, bias=True)
        else:
            # Deep layers for concatenated outputs
            deep_layers = []
            prev_dim = encoder_output_dim * 2  # Concatenated outputs of encoder
            for hidden_dim in deep_hidden_dims:
                deep_layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout_rate)
                ])
                prev_dim = hidden_dim
            self.deep_layers = nn.Sequential(*deep_layers)
            self.output_layer = nn.Linear(prev_dim, 1)  # Final output layer
        
        # Wrap the model with DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            self.encoder = nn.DataParallel(self.encoder)
            if not self.use_bilinear:
                self.deep_layers = nn.DataParallel(self.deep_layers)
                self.output_layer = nn.DataParallel(self.output_layer)
        
        self.criterion = SparseEncoderLoss(base_loss=nn.HuberLoss(delta=0.1), lambda_reg=lambda_reg)
        self.optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 2 * input_dim].
        Returns:
            torch.Tensor: Output predictions of shape [batch_size, 1].
        """
        # Split input into region i and region j
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        
        # Encode both regions using the shared encoder
        encoded_i = self.encoder(x_i)
        encoded_j = self.encoder(x_j)
        
        if self.use_bilinear:
            # Use bilinear layer for interactions
            output = self.bilinear(encoded_i, encoded_j)
        else:
            # Concatenate encoded outputs and pass through deep layers
            concatenated_embedding = torch.cat((encoded_i, encoded_j), dim=1)
            deep_output = self.deep_layers(concatenated_embedding)
            output = self.output_layer(deep_output)
        
        return output.squeeze()

    def get_params(self): # for local model saving
        params = {
            'input_dim': self.input_dim,  # multiply by 2 since input is split
            'encoder_hidden_dim': self.encoder_hidden_dim,
            'encoder_output_dim': self.encoder_output_dim,  # Use last layer for output dim
            'use_bilinear': self.use_bilinear,
            'deep_hidden_dims': self.deep_hidden_dims if not self.use_bilinear else None,
            'dropout_rate': self.dropout_rate if not self.use_bilinear else 0.0,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'lambda_reg': self.lambda_reg,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': str(self.device)  # Convert device to string for serialization
        }
        return params

    def predict(self, X):
        self.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self(X).cpu().numpy()
        return predictions

    def fit(self, X_train, y_train, X_test=None, y_test=None, verbose=True):
        train_loader = create_data_loader(X_train, y_train, self.batch_size, self.device, shuffle=True) # for bilinear models keeping unshuffled leads to more symmetric guess
        val_loader = None
        if X_test is not None and y_test is not None:
            val_loader = create_data_loader(X_test, y_test, self.batch_size, self.device, shuffle=True)
        return train_model(self, train_loader, val_loader, self.epochs, self.criterion, self.optimizer, verbose=verbose)


from models.bilinear import BilinearLoss

class SharedLinearEncoderModel(nn.Module):
    def __init__(self, input_dim, encoder_output_dim=32, deep_hidden_dims=[64], dropout_rate=0.0, learning_rate=0.01, weight_decay=0.0, regularization='l2', lambda_reg=0.1, batch_size=256, epochs=100):
        """
        A model that uses a shared encoder and a bilinear layer for interactions.
        Args:
            input_dim (int): Number of input features (genes).
            encoder_output_dim (int): Number of output features from the encoder.
            use_bilinear (bool): Whether to use a bilinear layer for interactions.
            deep_hidden_dims (list): List of hidden layer sizes for additional processing.
            dropout_rate (float): Dropout probability. 
            lambda_reg (float): Regularization strength for L1 penalty.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            batch_size (int): Batch size for training.
            epochs (int): Number of epochs to train for.
        """
        super().__init__()
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.regularization = regularization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder_output_dim=encoder_output_dim
        self.deep_hidden_dims=deep_hidden_dims
        self.dropout_rate=dropout_rate
        self.lambda_reg=lambda_reg

        self.encoder = nn.Linear(input_dim//2, encoder_output_dim)

        # Deep layers for concatenated outputs
        deep_layers = []
        prev_dim = encoder_output_dim * 2  # Concatenated outputs of encoder
        for hidden_dim in deep_hidden_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        self.deep_layers = nn.Sequential(*deep_layers)
        self.output_layer = nn.Linear(prev_dim, 1)  # Final output layer
        
        # Wrap the model with DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            self.encoder = nn.DataParallel(self.encoder)
            self.deep_layers = nn.DataParallel(self.deep_layers)
            self.output_layer = nn.DataParallel(self.output_layer)
        
        self.criterion = BilinearLoss(regularization='l2', lambda_reg=lambda_reg) # SparseEncoderLoss(base_loss=nn.HuberLoss(delta=0.1), lambda_reg=lambda_reg)
        self.optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 2 * input_dim].
        Returns:
            torch.Tensor: Output predictions of shape [batch_size, 1].
        """
        # Split input into region i and region j
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        
        # Encode both regions using the shared encoder
        encoded_i = self.encoder(x_i)
        encoded_j = self.encoder(x_j)
        
        # Concatenate encoded outputs and pass through deep layers
        concatenated_embedding = torch.cat((encoded_i, encoded_j), dim=1)
        deep_output = self.deep_layers(concatenated_embedding)
        output = self.output_layer(deep_output)
        
        return output.squeeze()

    def get_params(self): # for local model saving
        params = {
            'input_dim': self.input_dim,  # multiply by 2 since input is split
            'encoder_output_dim': self.encoder_output_dim,  # Use last layer for output dim
            'deep_hidden_dims': self.deep_hidden_dims,
            'dropout_rate': self.dropout_rate, 
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'regularization': self.regularization,
            'lambda_reg': self.lambda_reg,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': str(self.device)  # Convert device to string for serialization
        }
        return params

    def predict(self, X):
        self.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self(X).cpu().numpy()
        return predictions

    def fit(self, X_train, y_train, X_test=None, y_test=None, verbose=True):
        train_loader = create_data_loader(X_train, y_train, self.batch_size, self.device, shuffle=True) # for bilinear models keeping unshuffled leads to more symmetric guess
        val_loader = None
        if X_test is not None and y_test is not None:
            val_loader = create_data_loader(X_test, y_test, self.batch_size, self.device, shuffle=True)
        return train_model(self, train_loader, val_loader, self.epochs, self.criterion, self.optimizer, verbose=verbose)