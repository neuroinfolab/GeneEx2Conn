# GeneEx2Conn/models/shared_encoder_model.py

from imports import *
from data.data_utils import create_data_loader
from models.train_val import train_model

class SelfAttentionEncoder(nn.Module):
    def __init__(self, input_dim, token_encoder_dim, d_model, output_dim, nhead=4, num_layers=4, dropout=0.1, use_positional_encoding=False):
        """
        A self-attention encoder
        """
        super(SelfAttentionEncoder, self).__init__()
        
        self.d_model = d_model
        self.token_encoder_dim = token_encoder_dim
        self.use_positional_encoding = use_positional_encoding

        self.input_projection = nn.Linear(token_encoder_dim, d_model) # up project to d_model

        self.encoder_layer = nn.TransformerEncoderLayer(
            batch_first=True, 
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, output_dim) # Linear output layer (outside of transformer)

    def forward(self, x):
        """
        Forward pass for the self-attention encoder.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, input_dim].
        Returns:
            torch.Tensor: Output after applying attention.
        """
        batch_size, seq_length = x.size()
        x = x.view(batch_size, seq_length // self.token_encoder_dim, self.token_encoder_dim)

        x = self.input_projection(x)
        
        if self.use_positional_encoding: # need to update this with input projection 
            x = self.add_positional_encoding(x)
        
        x = self.transformer(x)
        
        x = self.fc(x).squeeze()
        #print('x shape after transformer and linear layer', x.shape)
        x = x.view(x.size(0), -1) # Flatten to [batch_size, seq_length * encoder_output_dim]
        #print('x shape flattened', x.shape)

        return x

    def add_positional_encoding(self, x):
        """
        Add positional encoding to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, token_encoder_dim].
            
        Returns:
            torch.Tensor: Input tensor with positional encoding added.
        """
        seq_length = x.size(1)
        position = torch.arange(seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.token_encoder_dim, 2).float() * (-math.log(10000.0) / self.token_encoder_dim))
        pe = torch.zeros(seq_length, self.token_encoder_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(x.device)  # Add batch dimension and move to device

        return x + pe

class SharedSelfAttentionModel(nn.Module):
    def __init__(self, input_dim, token_encoder_dim=10, d_model=128, encoder_output_dim=1, nhead=2, num_layers=2, deep_hidden_dims=[256, 128], 
                 use_positional_encoding=False, transformer_dropout=0.1, dropout_rate=0.1, learning_rate=0.001, weight_decay=0.0, lambda_reg=0.0, 
                 batch_size=128, epochs=100):
        """
        A model using a shared self-attention encoder with positional encoding.
        
        Args:
            input_dim (int): Number of input features (genes).
            encoder_hidden_dim (int): Number of hidden units in the self-attention encoder.
            encoder_output_dim (int): Number of output features from the encoder.
            deep_hidden_dims (list): List of hidden layer sizes for additional processing.
            dropout_rate (float): Dropout rate for the encoder and deep layers.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            lambda_reg (float): Regularization strength for L1 penalty.
            batch_size (int): Batch size for training.
            epochs (int): Number of epochs for training.
        """
        super().__init__()

        # Store parameters
        self.input_dim = input_dim // 2
        self.token_encoder_dim = token_encoder_dim # # chunk size for self-attention  OR vector length to project each token to if encoder used
        self.encoder_output_dim = encoder_output_dim # vector length of processed token
        self.transformer_dropout = transformer_dropout
        self.use_positional_encoding = use_positional_encoding
        self.nhead = nhead
        self.num_layers = num_layers
        self.d_model = d_model
        
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
                                            dropout=self.transformer_dropout,
                                            use_positional_encoding=self.use_positional_encoding)

        # Deep layers for concatenated outputs
        prev_dim = self.input_dim * 2 # // self.token_encoder_dim * self.encoder_output_dim # Concatenated outputs of encoder
        # concatenated encoder output

        deep_layers = []
        for hidden_dim in deep_hidden_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
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
        
        # Concatenate encoded outputs and pass through deep layers
        concatenated_embedding = torch.cat((encoded_i, encoded_j), dim=1)
        deep_output = self.deep_layers(concatenated_embedding)
        output = self.output_layer(deep_output)
        
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