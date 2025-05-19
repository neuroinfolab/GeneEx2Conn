from env.imports import *
from data.data_utils import create_data_loader
from models.train_val import train_model

class TweedieLoss(nn.Module):
    def __init__(self, p: float = 1.5, reduction: str = "mean", eps: float = 1e-8):
        """
        Tweedie Loss for log-link regression, adapted for min-max scaled count data.

        Args:
            p (float, optional): Tweedie variance power (1 ≤ p < 2).
                - `p ≈ 1`: Approaches Poisson distribution.
                - `p ≈ 2`: Approaches Gamma distribution.
                - Default `p=1.5` models compound Poisson-Gamma.
            reduction (str, optional): Reduction method, either "mean", "sum", or "none".
            eps (float, optional): Small constant for numerical stability (default: 1e-8).
        """
        super().__init__()
        assert 1.0 <= p < 2.0, "p must be in the range [1, 2)"
        self.p = p
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        Compute Tweedie loss.

        Args:
            y_pred (torch.Tensor): Model output (log scale, needs exponentiation).
            y_true (torch.Tensor): Target values (assumed min-max scaled and log-transformed).

        Returns:
            torch.Tensor: Tweedie loss.
        """
        # Ensure predictions are positive by exponentiating outputs
        y_pred = torch.exp(y_pred)  # This ensures positivity

        # Compute Tweedie loss
        loss = - (y_true * (y_pred ** (1 - self.p))) / (1 - self.p) + \
                 (y_pred ** (2 - self.p)) / (2 - self.p)

        # Apply reduction
        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss  # Return per-sample loss

class DynamicMLP(nn.Module):
    def __init__(self, input_dim, binarize, hidden_dims=[256, 128], dropout_rate=0.0, learning_rate=1e-3, weight_decay=0.0, batch_size=64, epochs=100):
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
            self.criterion = nn.BCEWithLogitsLoss()
        else: 
            self.criterion = nn.MSELoss()
            # consider HuberLoss for less sensitivity to outliers; QuantileLoss to prioritize outliers
            # self.criterion = nn.PoissonNLLLoss(log_input=True)
            # self.criterion = TweedieLoss(p=1.5)

        self.model = nn.Sequential(*layers)
        self.optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        self.patience = 100
        self.scheduler = ReduceLROnPlateau( 
            self.optimizer, 
            mode='min', 
            factor=0.3,  # Reduce LR by 70%
            patience=20,  # Reduce LR after patience epochs of no improvement
            threshold=0.05,  # Threshold to detect stagnation
            cooldown=1,  # Reduce cooldown period
            min_lr=1e-6,  # Prevent LR from going too low
            verbose=True
        )
        
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in MLP: {num_params}")
        
        self.to(self.device)
    
    def forward(self, x):
        return self.model(x).squeeze()

    def predict(self, loader):
        self.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for batch_X, batch_y, _, _ in loader:
                batch_X = batch_X.to(self.device)
                batch_preds = self(batch_X).cpu().numpy()
                predictions.append(batch_preds)
                targets.append(batch_y.numpy())
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        return ((predictions > 0.5).astype(int) if self.binarize else predictions), targets
    
    def fit(self, dataset, train_indices, test_indices, verbose=True):
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, self.scheduler, verbose=verbose)