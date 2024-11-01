# Gene2Conn/models/test_nn.py

from imports import *

from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from models.base_models import BaseModel
import wandb

class CustomNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.2):
        super().__init__()
        self.input_dim = input_dim
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
            
        # Final layer for regression
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()

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