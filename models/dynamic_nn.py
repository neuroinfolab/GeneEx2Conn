# Gene2Conn/models/dynamic_nn.py

from imports import *

from models.base_models import BaseModel
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

class DynamicNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.0, learning_rate=1e-3, weight_decay=0, batch_size=64, epochs=100):
        super(DynamicNN, self).__init__()
        
        # Model hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build network layers dynamically
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self.to(self.device)  # Move model to the appropriate device
        
        # Optimizer and loss function
        self.optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.network(x).squeeze()

    def _create_data_loader(self, X, y, shuffle=False):
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def symmetry_loss(self, predictions):
        """
        Compute symmetry loss for consecutive pairs in the batch.
        
        Args:
            predictions (tensor): Predicted values for the batch.
        
        Returns:
            symmetry_loss (tensor): Calculated symmetry loss.
        """
        # Split predictions into consecutive pairs (i, j) and (j, i)
        predictions_i_j = predictions[::2]
        predictions_j_i = predictions[1::2]
        
        # Compute symmetry loss as the mean absolute difference between consecutive pairs
        return torch.mean(torch.abs(predictions_i_j - predictions_j_i))

    
    def train_model(self, train_loader, val_loader=None, verbose=True):
        train_history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(self.epochs):
            # Training phase
            self.train()
            total_train_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                # Forward pass
                predictions = self(batch_X)
                # Compute MSE loss
                mse_loss = self.criterion(predictions, batch_y)
                # Compute symmetry loss
                sym_loss = self.symmetry_loss(predictions)
                # Combine losses
                total_loss = mse_loss + self.symmetry_weight * sym_loss # symmetry weight likely needs to be tuned here to determine importance
                # Backward pass and optimization
                total_loss.backward()
                self.optimizer.step()
                total_train_loss += total_loss.item()
            average_train_loss = total_train_loss / len(train_loader)
            train_history["train_loss"].append(average_train_loss)
            
            # Validation phase
            if val_loader:
                self.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        predictions = self(batch_X)
                        val_loss = self.loss_fn(predictions, batch_y)
                        total_val_loss += val_loss.item()
                average_val_loss = total_val_loss / len(val_loader)
                train_history["val_loss"].append(average_val_loss)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}")
            elif verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {average_train_loss:.4f}")

        return train_history

    def fit(self, X, y, val_data=None, verbose=True):
        train_loader = self._create_data_loader(X, y, shuffle=True)
        val_loader = self._create_data_loader(*val_data, shuffle=False) if val_data else None
        return self.train_model(train_loader, val_loader, verbose=verbose)

    def predict(self, X):
        self.eval()
        X = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            predictions = self(X).cpu().numpy()
        return predictions

    def get_params(self):
        return {
            'input_dim': self.network[0].in_features,
            'hidden_dims': [layer.out_features for layer in self.network if isinstance(layer, nn.Linear)][:-1],
            'dropout_rate': next((layer.p for layer in self.network if isinstance(layer, nn.Dropout)), 0.0),
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': str(self.device)
        }

