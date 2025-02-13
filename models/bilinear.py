# GeneEx2Conn/models/bilinear.py

from env.imports import *
from data.data_utils import create_data_loader
from models.train_val import train_model

class BilinearLoss(nn.Module):
    """MSE loss with optional L1/L2 regularization for bilinear models."""
    def __init__(self, regularization='l1', lambda_reg=1.0):
        super().__init__()
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets, model):
        mse_loss = self.mse(predictions, targets)
        if self.lambda_reg > 0:
            # Concatenate all parameters into a single tensor
            params = torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad])
            if self.regularization == 'l1':
                reg_loss = torch.linalg.norm(params, ord=1)  # L1 norm
            elif self.regularization == 'l2':
                reg_loss = torch.linalg.norm(params, ord=2) # L2 norm
            return mse_loss + self.lambda_reg * reg_loss

        return mse_loss


class BilinearLowRank(nn.Module):
    def __init__(self, input_dim, reduced_dim, activation='none', learning_rate=0.01, epochs=100, 
                 batch_size=128, regularization='l1', lambda_reg=1.0, shared_weights=True):
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.shared_weights = shared_weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.linear = nn.Linear(input_dim//2, reduced_dim, bias=False)
        if not shared_weights:
            self.linear2 = nn.Linear(input_dim//2, reduced_dim, bias=False)
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in bilinear low rank model: {num_params}")
        
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        else:  # 'none'
            self.activation = nn.Identity()

        self.criterion = BilinearLoss(regularization=regularization, lambda_reg=lambda_reg)
        self.optimizer = Adam(self.parameters(), lr=learning_rate)

    def forward(self, x): 
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        out1 = self.activation(self.linear(x_i))
        out2 = self.activation(self.linear(x_j) if self.shared_weights else self.linear2(x_j))
        return torch.sum(out1 * out2, dim=1) # dot product for paired samples

    def predict(self, X):
        self.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self(X).cpu().numpy()
        return predictions

    def fit(self, X_train, y_train, X_test, y_test, verbose=True):
        train_loader = create_data_loader(X_train, y_train, self.batch_size, self.device, shuffle=True)
        val_loader = create_data_loader(X_test, y_test, self.batch_size, self.device, shuffle=True)
        return train_model(self, train_loader, val_loader, self.epochs, self.criterion, self.optimizer, verbose=verbose)


class BilinearSCM(nn.Module):
    def __init__(self, input_dim, learning_rate=0.01, epochs=100, 
                 batch_size=128, regularization='l2', lambda_reg=1.0, bias=True):
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bilinear = nn.Bilinear(input_dim//2, input_dim//2, 1, bias=bias)
        num_params = sum(p.numel() for p in self.bilinear.parameters() if p.requires_grad)
        print(f"Number of learnable parameters in bilinear SCM layer: {num_params}")
        
        self.criterion = BilinearLoss(regularization=regularization, lambda_reg=lambda_reg)
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=20, verbose=True)
    
    def forward(self, x):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        return self.bilinear(x_i, x_j).squeeze()

    def predict(self, X):
        self.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self(X).cpu().numpy()
        return predictions

    def fit(self, X_train, y_train, X_test=None, y_test=None, verbose=True):
        train_loader = create_data_loader(X_train, y_train, self.batch_size, self.device, shuffle=True)
        if X_test is not None and y_test is not None:
            val_loader = create_data_loader(X_test, y_test, self.batch_size, self.device, shuffle=True)
        else:
            val_loader = None
        return train_model(self, train_loader, val_loader, self.epochs, self.criterion, self.optimizer, self.scheduler, verbose=verbose)