import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedEncoderModel(nn.Module):
    def __init__(self, input_dim, encoder_hidden_dim=64, encoder_output_dim=32, 
                 deep_hidden_dims=[128, 64], dropout_rate=0.0, sparsity_weight=1e-5):
        super(SharedEncoderModel, self).__init__()
        
        self.sparsity_weight = sparsity_weight  # For L1 regularization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Shared encoder for gene-level features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(encoder_hidden_dim),
            nn.Linear(encoder_hidden_dim, encoder_output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(encoder_output_dim),
            nn.Dropout(dropout_rate)
        )
        
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
        
        # Final output layer
        self.output_layer = nn.Linear(prev_dim, 1)

    def encode(self, x):
        """
        Pass input through the shared encoder.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
        Returns:
            torch.Tensor: Encoded tensor of shape [batch_size, encoder_output_dim].
        """
        return self.encoder(x)

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
        encoded_i = self.encode(x_i)
        encoded_j = self.encode(x_j)
        
        # Concatenate encoded outputs
        concatenated = torch.cat((encoded_i, encoded_j), dim=1)
        
        # Pass through deep layers and output layer
        deep_output = self.deep_layers(concatenated)
        output = self.output_layer(deep_output)
        return output.squeeze()

    def compute_sparsity_loss(self):
        """
        Compute L1 regularization loss for the encoder weights.
        Returns:
            torch.Tensor: Sparsity loss value.
        """
        sparsity_loss = 0.0
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                sparsity_loss += torch.sum(torch.abs(layer.weight))
        return self.sparsity_weight * sparsity_loss