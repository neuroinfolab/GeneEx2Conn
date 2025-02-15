from env.imports import *
from data.data_utils import create_data_loader
from models.train_val import train_model


class DynamicMLP(nn.Module):
    def __init__(self, input_dim, binarize, hidden_dims=[256, 128], dropout_rate=0.0, learning_rate=1e-3, weight_decay=0, batch_size=64, epochs=100):
        super().__init__()
        self.input_dim = input_dim
        self.binarize = binarize
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout_rate = dropout_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        if self.binarize:
            layers.append(nn.Sigmoid())
            self.criterion = nn.BCELoss()
        else: 
            self.criterion = nn.HuberLoss(delta=0.1) # this can be tuned to nn.MSELoss() or other

        self.model = nn.Sequential(*layers)
        
        self.optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.scheduler = ReduceLROnPlateau( 
            self.optimizer, 
            mode='min', 
            factor=0.1,  # Reduce LR by 70%
            patience=30,  # Reduce LR after 30 epochs of no improvement
            threshold=0.005,  # Smaller threshold to detect stagnation
            cooldown=1,  # Reduce cooldown period
            min_lr=1e-6,  # Prevent LR from going too low
            verbose=True
        )

        # HELPER FUNC THIS AT SOME POINT
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in MLP: {num_params}")

        if torch.cuda.device_count() > 1:
            dist.init_process_group(backend='nccl')
            self.model = DDP(self.model)
        self.to(self.device)
    
    def forward(self, x):
        return self.model(x).squeeze()

    def predict(self, X):
        self.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self(X).cpu().numpy()
        return (predictions > 0.5).astype(int) if self.binarize else predictions

    def fit(self, X_train, y_train, X_test, y_test, verbose=True):
        # loaders will automatically detect if working in binary target setting
        train_loader = create_data_loader(X_train, y_train, self.batch_size, self.device)
        val_loader = create_data_loader(X_test, y_test, self.batch_size, self.device, validation=True)
        return train_model(self, train_loader, val_loader, self.epochs, self.criterion, self.optimizer, self.scheduler, verbose=verbose)