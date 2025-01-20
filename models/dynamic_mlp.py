# GeneEx2Conn/models/dynamic_mlp.py

from imports import *
from data.data_utils import create_data_loader
from models.train_val import train_model


class DynamicMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout_rate=0.0, learning_rate=1e-3, weight_decay=0, batch_size=64, epochs=100):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        layers = []
        prev_dim = input_dim # adjust first layer based on input size
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in MLP: {num_params}")

        if torch.cuda.device_count() > 1: # distributed training
            dist.init_process_group(backend='nccl')
            self.model = DDP(self.model)
        self.to(self.device)  # Move model to the appropriate device
        
        self.criterion = nn.HuberLoss(delta=0.1) # this can be tuned to nn.MSELoss() or other
        self.optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=20, verbose=True)
    
    def get_params(self): # for local model saving
        return {
            'input_dim': self.model[0].in_features,
            'hidden_dims': [layer.out_features for layer in self.model if isinstance(layer, nn.Linear)][:-1],
            'dropout_rate': next((layer.p for layer in self.model if isinstance(layer, nn.Dropout)), 0.0),
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'device': str(self.device)
        }

    def forward(self, x):
        return self.model(x).squeeze()

    def predict(self, X):
        self.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self(X).cpu().numpy()
        return predictions

    def fit(self, X_train, y_train, X_test, y_test, verbose=True):
        train_loader = create_data_loader(X_train, y_train, self.batch_size, self.device, shuffle=True)
        val_loader = create_data_loader(X_test, y_test, self.batch_size, self.device, shuffle=True)
        return train_model(self, train_loader, val_loader, self.epochs, self.criterion, self.optimizer, self.scheduler, verbose=verbose)