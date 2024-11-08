# Gene2Conn/models/dynamic_nn.py

from imports import *

from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from models.base_models import BaseModel
import wandb
import torch
import torch.nn as nn

class DynamicNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.0, learning_rate=1e-3, weight_decay=0, batch_size=64, epochs=100):
        super().__init__()

        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build network layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.MSELoss()
    
    def get_params(self):
        """
        Get the parameters of the current model configuration.
        
        Returns:
            Dictionary containing model parameters.
        """
        return {
            'input_dim': self.input_dim,
            'hidden_dims': [layer.out_features for layer in self.network if isinstance(layer, nn.Linear)][:-1],
            'dropout_rate': next((layer.p for layer in self.network if isinstance(layer, nn.Dropout)), 0.0),
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': str(self.device)
        }
    
    def forward(self, x):
        return self.network(x).squeeze()

    def create_data_loader(self, X, y, batch_size, shuffle=False):
        """
        Create a DataLoader from input features and targets
        
        Args:
            X: Input features (numpy array or tensor)
            y: Target values (numpy array or tensor)
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the data
        
        Returns:
            DataLoader object with data moved to correct device
        """
        # Convert to tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.FloatTensor(y)
        
        # Move to device
        X = X.to(self.device)
        y = y.to(self.device)
        # Print shapes of input tensors
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        # Create dataset and loader
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train_fold(self, train_loader, val_loader, epochs=100, mode='learn', verbose=True):
        """
        Train the model with validation using DataLoaders
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
            mode: 'learn' for training with validation, 'retrain' for full dataset training
            verbose: Whether to print training progress
        
        Returns:
            Dictionary containing training history
        """
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            epoch_train_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                outputs = self(batch_X)
                train_loss = self.criterion(outputs, batch_y)
                
                train_loss.backward()
                self.optimizer.step()
                
                epoch_train_loss += train_loss.item()           
            
            # Average training loss for the epoch
            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)
            
            # Validation phase
            if mode == 'learn':
                self.eval()
                epoch_val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        val_outputs = self(batch_X)
                        val_loss = self.criterion(val_outputs, batch_y)
                        epoch_val_loss += val_loss.item()
                    
                    epoch_val_loss /= len(val_loader)
                    val_losses.append(epoch_val_loss)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs} - '
                          f'Train Loss: {epoch_train_loss:.6f} - '
                          f'Val Loss: {epoch_val_loss:.6f}')
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs} - '
                          f'Train Loss: {epoch_train_loss:.6f}')
        
        return {
            'mode': mode,
            'train_losses': train_losses,
            'val_losses': val_losses if mode == 'learn' else None,
            'epochs_trained': epochs,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1] if val_losses else None
        }

    def fit(self, X, y, epochs=None, verbose=True):
        epochs = self.epochs # Use class epochs if not specified
        print('NUM EPOCHS', epochs)
        
        train_loader = self.create_data_loader(X, y, batch_size=self.batch_size, shuffle=False)
        train_losses = []

        for epoch in range(epochs):
            self.train()
            epoch_train_loss = 0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self(batch_X)
                train_loss = self.criterion(outputs, batch_y)
                train_loss.backward()
                self.optimizer.step()
                
                epoch_train_loss += train_loss.item()
        
            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)
        
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {epoch_train_loss:.6f}')

        return {
            'train_losses': train_losses,
            'epochs_trained': epochs,
            'final_train_loss': train_losses[-1]
        }
        
    def predict(self, X):
        """
        Make predictions for input tensor X
        
        Args:
            X: Input tensor of shape (n_samples, input_dim) or (input_dim,)
        
        Returns:
            Predictions as numpy array
        """
        # Ensure model is in evaluation mode
        self.eval()
        
        # Get the device the model is on
        device = next(self.parameters()).device
        
        # Handle single sample vs batch
        if X.ndim == 1:
            X = X.unsqueeze(0)
        
        # Convert to tensor if not already
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        
        # Move to correct device
        X = X.to(device)
        
        # Make predictions
        with torch.no_grad():
            predictions = self(X)
        
        # Convert to numpy and handle single sample vs batch
        predictions = predictions.cpu().numpy()
        if predictions.shape[0] == 1:
            predictions = predictions[0]
            
        return predictions

