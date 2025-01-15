# GeneEx2Conn/models/shared_encoder_model.py

from imports import *
from data.data_utils import create_data_loader
from models.train_val import train_model


class SparseMLPEncoderLoss(nn.Module):
    def __init__(self, base_loss=nn.MSELoss(), lambda_reg=1.0):
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
    def __init__(self, input_dim, encoder_hidden_dim=64, encoder_output_dim=32, use_bilinear=False, deep_hidden_dims=[64], dropout_rate=0.0, learning_rate=0.01, weight_decay=0.0, lambda_reg=1.0, batch_size=256, epochs=100):
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
            self.bilinear = nn.Bilinear(encoder_output_dim, encoder_output_dim, 1)
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

        self.criterion = SparseMLPEncoderLoss(base_loss=nn.MSELoss(), lambda_reg=lambda_reg)
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