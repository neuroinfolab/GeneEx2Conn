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
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.0, learning_rate=1e-3, weight_decay=0):
        super().__init__()

        '''
        # Define sweep configuration
        self.sweep_config = {
            'method': 'random',
            'metric': {
                'name': 'val_loss',
                'goal': 'minimize'
            },
            'parameters_dict': {
                'hidden_dims': {
                    'values': [
                        [64, 32],
                        [128, 64],
                        [256, 128, 64]
                    ]
                },
                'learning_rate': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-4,
                    'max': 1e-2
                },
                'batch_size': {
                    'distribution': 'q_log_uniform_values',
                    'q': 8,
                    'min': 32,
                    'max': 256
                },
                'dropout_rate': {
                    'distribution': 'uniform',
                    'min': 0.1,
                    'max': 0.5
                },
                'weight_decay': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-5,
                    'max': 1e-3
                },
                'epochs': {'value': 100}
            }
        }
        '''

        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
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

    def forward(self, x):
        return self.network(x).squeeze()

    def fit(self, train_loader, val_loader=None, epochs=100, mode='learn', verbose=True):
        """
        Train the model in either learning or retraining mode
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data (used only in 'learn' mode)
            epochs: Number of epochs to train
            mode: 'learn' for training with validation, 'retrain' for full dataset training
            verbose: Whether to print training progress
        
        Returns:
            Dictionary containing training history
        """
        device = next(self.parameters()).device
        train_losses = []
        val_losses = []
        
        # Determine if we're in learning or retraining mode
        is_learning = mode == 'learn'
        if not is_learning:
            val_loader = None  # Ignore validation in retrain mode
            self.train()  # Ensure all layers are in training mode
        
        # Convert data to DataLoader if not already
        if not isinstance(train_loader, DataLoader):
            train_dataset = TensorDataset(train_loader[0], train_loader[1])
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
        if val_loader is not None and not isinstance(val_loader, DataLoader):
            val_dataset = TensorDataset(val_loader[0], val_loader[1])
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
        for epoch in range(epochs):
            # Training phase
            if is_learning:
                self.train()
            epoch_loss = 0
            for batch in train_loader:
                x, y = [b.to(device) for b in batch]
                self.optimizer.zero_grad()
                outputs = self(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase (only in learning mode)
            if is_learning and val_loader is not None:
                self.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        x, y = [b.to(device) for b in batch]
                        outputs = self(x)
                        loss = self.criterion(outputs, y)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs} - '
                          f'Train Loss: {avg_train_loss:.6f} - '
                          f'Val Loss: {avg_val_loss:.6f}')
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs} - '
                          f'Train Loss: {avg_train_loss:.6f}')
        
        return {
            'mode': mode,
            'train_losses': train_losses,
            'val_losses': val_losses if is_learning else None,
            'epochs_trained': epochs,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1] if val_losses else None
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



class NeuralNetworkModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define sweep configuration
        self.sweep_config = {
            'method': 'random',
            'metric': {
                'name': 'val_loss',
                'goal': 'minimize'
            },
            'parameters': {
                'hidden_dims': {
                    'values': [
                        [64, 32],
                        [128, 64],
                        [256, 128, 64]
                    ]
                },
                'learning_rate': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-4,
                    'max': 1e-2
                },
                'batch_size': {
                    'distribution': 'q_log_uniform_values',
                    'q': 8,
                    'min': 32,
                    'max': 256
                },
                'dropout_rate': {
                    'distribution': 'uniform',
                    'min': 0.1,
                    'max': 0.5
                },
                'weight_decay': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-5,
                    'max': 1e-3
                },
                'epochs': {'value': 100}
            }
        }
    
    def init_model(self, config):
        """Initialize model with given config"""
        model = CustomNN(
            input_dim=self.input_dim,
            hidden_dims=config.hidden_dims,
            dropout_rate=config.dropout_rate
        ).to(self.device)
        return model

    def train_fold(self, model, X_train, y_train, X_val, y_val, config, fold_idx):
        """Train model on a single fold"""
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=config.batch_size,
            shuffle=True
        )
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(config.epochs):
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
            
            # Log metrics for this fold
            wandb.log({
                f"train_loss_fold_{fold_idx}": train_loss / len(train_loader),
                f"val_loss_fold_{fold_idx}": val_loss,
                "epoch": epoch
            })
        
        # Return best model for this fold
        model.load_state_dict(best_model_state)
        return model, best_val_loss
    
    def fit(self, X, Y, config):
        """Train model on input data"""
        model = self.init_model(config)
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X).to(self.device),
            torch.FloatTensor(Y).to(self.device)
        )
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=config.batch_size,
            shuffle=True
        )
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        criterion = nn.MSELoss()
        
        for epoch in range(config.epochs):
            model.train()
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        return model