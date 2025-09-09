from env.imports import *
from data.data_utils import create_data_loader
from scipy.optimize import curve_fit
from torch.utils.data import DataLoader, Subset

def compute_loss_for_loader(model, loader):
    """Helper function to compute loss for a data loader"""
    total_loss = 0
    n_samples = 0
    for batch_X, batch_y, _, _ in loader:
        batch_X = batch_X.to(model.device)
        batch_y = batch_y.to(model.device)
        pred = model(batch_X)
        total_loss += model.criterion(pred, batch_y).item() * len(batch_y)
        n_samples += len(batch_y)
    return total_loss / n_samples

def predict_from_loader(model, loader):
    """Helper function to get predictions from a loader"""
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch_X, batch_y, _, _ in loader:
            batch_X = batch_X.to(model.device)
            batch_preds = model(batch_X).cpu().numpy()
            predictions.append(batch_preds)
            targets.append(batch_y.numpy())
    return np.concatenate(predictions), np.concatenate(targets)

class CGEModel(nn.Module):
    """Simple correlation between gene expression vectors."""
    def __init__(self, input_dim, binarize=None, scale_range=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.scale_range = scale_range
    
    def forward(self, x):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        # Compute correlation between the two chunks
        x_i_centered = x_i - x_i.mean(dim=1, keepdim=True)
        x_j_centered = x_j - x_j.mean(dim=1, keepdim=True)
        corr = torch.sum(x_i_centered * x_j_centered, dim=1) / (
            torch.norm(x_i_centered, dim=1) * torch.norm(x_j_centered, dim=1)
        )
        
        if self.scale_range and hasattr(self, 'train_min'):
            # Scale predictions to range of training data
            corr = ((corr - self.pred_min) / (self.pred_max - self.pred_min)) * (self.train_max - self.train_min) + self.train_min
            
        return corr

    def fit(self, dataset, train_indices, test_indices, save_model=None):
        train_X = dataset.X_expanded[train_indices].to(self.device)
        train_y = dataset.Y_expanded[train_indices].to(self.device)
        test_X = dataset.X_expanded[test_indices].to(self.device)
        test_y = dataset.Y_expanded[test_indices].to(self.device)
        
        train_pred = self(train_X)
        
        if self.scale_range:
            # Store the ranges for scaling
            self.train_min = train_y.min()
            self.train_max = train_y.max()
            self.pred_min = train_pred.min()
            self.pred_max = train_pred.max()
            # Scale predictions
            train_pred = ((train_pred - self.pred_min) / (self.pred_max - self.pred_min)) * (self.train_max - self.train_min) + self.train_min
        
        test_pred = self(test_X)
        train_loss = self.criterion(train_pred, train_y).item()
        val_loss = self.criterion(test_pred, test_y).item()
        return {'train_loss': [train_loss], 'val_loss': [val_loss]}

    def predict(self, loader):
        return predict_from_loader(self, loader)

class GaussianKernelModel(nn.Module):
    """Gaussian kernel based on euclidean distance."""
    def __init__(self, input_dim, binarize=None, init_sigma=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        # Initialize sigma as a learnable parameter, value will be set in fit()
        self.sigma = nn.Parameter(torch.tensor(1.0))
        self.init_sigma = init_sigma
        
    def forward(self, x):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        x_i[:, 0] = -torch.abs(x_i[:, 0])  # Make x coordinate negative
        x_j[:, 0] = -torch.abs(x_j[:, 0])  # Make x coordinate negative
        dist = torch.sum((x_i - x_j)**2, dim=1) # Compute euclidean distance
        return torch.exp(-dist / (2 * self.sigma**2)) # Apply gaussian kernel
        
    def fit(self, dataset, train_indices, test_indices, save_model=None):
        train_X = dataset.X_expanded[train_indices].to(self.device)
        train_y = dataset.Y_expanded[train_indices].to(self.device)
        test_X = dataset.X_expanded[test_indices].to(self.device)
        test_y = dataset.Y_expanded[test_indices].to(self.device)
        
        # Calculate distances and initialize sigma based on std dev if not provided
        x_i, x_j = torch.chunk(train_X, chunks=2, dim=1)
        x_i[:, 0] = -torch.abs(x_i[:, 0])  # Make x coordinate negative
        x_j[:, 0] = -torch.abs(x_j[:, 0])  # Make x coordinate negative
        dist = torch.sum((x_i - x_j)**2, dim=1).sqrt().cpu().numpy()
        if self.init_sigma is None:
            init_sigma = np.std(dist)
        else:
            init_sigma = self.init_sigma
        
        # Optimize sigma using curve_fit with initialization
        def gaussian(x, sigma):
            return np.exp(-x**2 / (2 * sigma**2))
        
        popt, _ = curve_fit(gaussian, dist, train_y.cpu().numpy(),
                           p0=[init_sigma], bounds=(10.0, 40.0))
        print('Optimized sigma:', f'{popt[0]:.3f}')
        self.sigma.data = torch.tensor(popt[0])
        
        train_pred = self(train_X)
        test_pred = self(test_X)
        train_loss = self.criterion(train_pred, train_y).item()
        val_loss = self.criterion(test_pred, test_y).item()
        
        return {'train_loss': [train_loss], 'val_loss': [val_loss]}
        
    def predict(self, loader):
        return predict_from_loader(self, loader)
        
        # x_i = -torch.abs(x_i[:, 0:1])  # Take negative of absolute value of first coordinate
        # x_j = -torch.abs(x_j[:, 0:1])  # Take negative of absolute value of first coordinate

class ExponentialDecayModel(nn.Module):
    """Exponential decay based on euclidean distance."""
    def __init__(self, input_dim, binarize=None, SA_inf=-0.2, SA_lambda=15.0):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        # Initialize learnable parameters - in theory these can be CV-able
        self.SA_inf = nn.Parameter(torch.tensor(float(SA_inf)))
        self.SA_lambda = nn.Parameter(torch.tensor(float(SA_lambda)))
        
    def forward(self, x):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1) 
        x_i[:, 0] = -torch.abs(x_i[:, 0])  # Make x coordinate negative
        x_j[:, 0] = -torch.abs(x_j[:, 0])  # Make x coordinate negative
        dist = torch.sum((x_i - x_j)**2, dim=1).sqrt() # Compute euclidean distance
        return self.SA_inf + (1 - self.SA_inf) * torch.exp(-dist / self.SA_lambda) # Apply exponential decay
        
    def fit(self, dataset, train_indices, test_indices, save_model=None):
        train_X = dataset.X_expanded[train_indices].to(self.device) # indices are expanded
        train_y = dataset.Y_expanded[train_indices].to(self.device)
        test_X = dataset.X_expanded[test_indices].to(self.device)
        test_y = dataset.Y_expanded[test_indices].to(self.device)
        
        # Optimize parameters using curve_fit
        x_i, x_j = torch.chunk(train_X, chunks=2, dim=1)
        x_i[:, 0] = -torch.abs(x_i[:, 0])  # Make x coordinate negative
        x_j[:, 0] = -torch.abs(x_j[:, 0])  # Make x coordinate negative
        dist = torch.sum((x_i - x_j)**2, dim=1).sqrt().cpu().numpy() # if slow can replace this with distances_expanded
        
        def exp_decay(x, SA_inf, SA_lambda):
            return SA_inf + (1 - SA_inf) * np.exp(-x / SA_lambda)
            
        popt, _ = curve_fit(exp_decay, dist, train_y.cpu().numpy(),
                           p0=[0, 10], bounds=([-1, 0], [1, 100]))
        print('Optimized parameters:', f'SA_inf={popt[0]:.3f}', f'SA_lambda={popt[1]:.3f}')
        self.SA_inf.data = torch.tensor(float(popt[0]))
        self.SA_lambda.data = torch.tensor(float(popt[1]))
        
        train_pred = self(train_X)
        test_pred = self(test_X)
        train_loss = self.criterion(train_pred, train_y).item()
        val_loss = self.criterion(test_pred, test_y).item()
        
        return {'train_loss': [train_loss], 'val_loss': [val_loss]}
        
    def predict(self, loader):
        return predict_from_loader(self, loader)