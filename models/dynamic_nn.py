# Gene2Conn/models/dynamic_nn.py

from imports import *

from models.base_models import BaseModel
from models.metrics.eval import pearson_numpy, pearson_cupy
from torchmetrics import PearsonCorrCoef
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class DynamicNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.0, learning_rate=1e-3, weight_decay=0, batch_size=64, symmetry_weight=0.1, epochs=100):
        super(DynamicNN, self).__init__()
        
        # Model hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.symmetry_weight = symmetry_weight
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

        # Initialize distributed training
        if torch.cuda.device_count() > 1:
            dist.init_process_group(backend='nccl')
            self.network = DDP(self.network)
    
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
        """
        # Split predictions into consecutive pairs (i, j) and (j, i)
        predictions_i_j = predictions[::2]
        predictions_j_i = predictions[1::2]
        
        # Compute symmetry loss as the mean absolute difference between consecutive pairs
        return torch.mean(torch.abs(predictions_i_j - predictions_j_i))

    def train_model(self, train_loader, val_loader=None, verbose=True, patience=100, min_delta=0.00, max_grad_norm=1.0):
        train_history = {"train_loss": [], "val_loss": [], "train_pearson": [], "val_pearson": []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training phase
            self.train()
            total_train_loss = 0
            train_pearson_values = []
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                predictions = self(batch_X)
                mse_loss = self.criterion(predictions, batch_y)
                sym_loss = self.symmetry_loss(predictions)
                total_loss = mse_loss + self.symmetry_weight * sym_loss
                total_loss.backward()
                
                # Apply gradient clipping, may need to tune this parameter
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

                self.optimizer.step()
                total_train_loss += total_loss.item()
                pearson = PearsonCorrCoef().to(self.device)
                train_pearson_values.append(pearson(predictions, batch_y).item())
           
            train_history["train_loss"].append(total_train_loss / len(train_loader))
            train_history["train_pearson"].append(np.mean(train_pearson_values))
            
            # Validation phase
            if val_loader:
                self.eval()
                total_val_loss = 0
                val_pearson_values = []
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        predictions = self(batch_X)
                        val_loss = self.criterion(predictions, batch_y)
                        total_val_loss += val_loss.item()
                        pearson = PearsonCorrCoef().to(self.device)
                        val_pearson_values.append(pearson(predictions, batch_y).item())
                
                avg_val_loss = total_val_loss / len(val_loader)
                train_history["val_loss"].append(avg_val_loss)
                train_history["val_pearson"].append(np.mean(val_pearson_values))
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_history['train_loss'][-1]:.4f}, Val Loss: {train_history['val_loss'][-1]:.4f}")

                # Early stopping check
                if avg_val_loss < best_val_loss - min_delta:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1               
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
            elif verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_history['train_loss'][-1]:.4f}")

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
            'symmetry_weight': self.symmetry_weight,
            'epochs': self.epochs,
            'device': str(self.device)
        }