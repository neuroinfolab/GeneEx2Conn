from env.imports import *
from data.data_utils import create_data_loader
from models.train_val import train_model
from models.bilinear import BilinearLoss


class SharedLinearEncoderModel(nn.Module):
    def __init__(self, input_dim, encoder_output_dim=32, deep_hidden_dims=[64], dropout_rate=0.0, learning_rate=0.01, weight_decay=0.0, regularization='l2', lambda_reg=0.1, batch_size=256, epochs=100):
        ''' Model that uses a shared linear encoder and deep MLP decoder'''
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
        
        self.criterion = BilinearLoss(self.parameters(), regularization='l2', lambda_reg=lambda_reg) # SparseEncoderLoss(base_loss=nn.HuberLoss(delta=0.1), lambda_reg=lambda_reg)
        self.optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def forward(self, x):
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

    def predict(self, X):
        self.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self(X).cpu().numpy()
        return predictions

    def fit(self, X_train, y_train, X_test=None, y_test=None, verbose=True):
        train_loader = create_data_loader(X_train, y_train, self.batch_size, self.device) # for bilinear models keeping unshuffled leads to more symmetric guess
        val_loader = None
        if X_test is not None and y_test is not None:
            val_loader = create_data_loader(X_test, y_test, self.batch_size, self.device)
        return train_model(self, train_loader, val_loader, self.epochs, self.criterion, self.optimizer, verbose=verbose)


class SparseEncoderLoss(nn.Module):
    def __init__(self, encoder, base_loss=nn.MSELoss(), lambda_reg=0.1):
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
        self.encoder = encoder

    def forward(self, predictions, targets):
        # Calculate the base loss
        loss = self.base_loss(predictions, targets)
        
        # Apply L1 regularization only to the encoder's parameters
        l1_reg = 0
        for param in self.encoder.parameters():
            if param.requires_grad:
                l1_reg += torch.sum(torch.abs(param))
        
        # Combine the base loss with the L1 regularization term
        total_loss = loss + self.lambda_reg * l1_reg
        return total_loss


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        ''' One hidden layer MLP encoder'''
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
        '''Model with shared MLP encoder and bilinear layer decoder'''
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
        
        if torch.cuda.device_count() > 1:
            self.encoder = nn.DataParallel(self.encoder)
            if not self.use_bilinear:
                self.deep_layers = nn.DataParallel(self.deep_layers)
                self.output_layer = nn.DataParallel(self.output_layer)
        
        self.criterion = SparseEncoderLoss(self.encoder, base_loss=nn.HuberLoss(delta=0.1), lambda_reg=lambda_reg)
        self.optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def forward(self, x):
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

    def predict(self, X):
        self.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self(X).cpu().numpy()
        return predictions

    def fit(self, X_train, y_train, X_test=None, y_test=None, verbose=True):
        train_loader = create_data_loader(X_train, y_train, self.batch_size, self.device) # for bilinear models keeping unshuffled leads to more symmetric guess
        val_loader = None
        if X_test is not None and y_test is not None:
            val_loader = create_data_loader(X_test, y_test, self.batch_size, self.device)
        return train_model(self, train_loader, val_loader, self.epochs, self.criterion, self.optimizer, verbose=verbose)