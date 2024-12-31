# Gene2Conn/models/base_models.py

from imports import *
from skopt.space import Real, Categorical, Integer
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from models.metrics.eval import mse_cupy

class BaseModel:
    """Base class for all models."""
    def __init__(self):
        self.model = None
        self.param_grid = {}

    def get_model(self):
        """Return the model instance."""
        return self.model

    def get_param_grid(self):
        """Return the parameter grid for GridSearchCV."""
        return self.param_grid
    
    def get_param_dist(self):
        """Return the parameter grid for GridSearchCV."""
        return self.param_dist


class RidgeModel(BaseModel):
    """Ridge Regression model with parameter grid."""

    def __init__(self):
        super().__init__()
        self.model = Ridge()
        self.param_grid = {
            'alpha': [0, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
            # 'alpha': np.logspace(-6, 2, 9)  # Explore 10^-6 to 10^2
            'solver': ['auto'] #, 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }
        self.param_dist = {
            'alpha': Real(1e-6, 1e2, prior='log-uniform')  # Log-uniform to explore a wide range of alphas
            #'solver': Categorical(['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])  # Different solvers to try
        }

class RidgeModelTorch(BaseEstimator, RegressorMixin):
    """PyTorch-based Ridge Regression model with L2 regularization (Ridge penalty)."""

    def __init__(self, input_dim, output_dim=1, alpha=1.0, lr=0.01, epochs=100, batch_size=32):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha  # Equivalent to L2 regularization
        self.lr = lr  # Learning rate
        self.epochs = epochs
        self.batch_size = batch_size

        # Define the model: a single layer linear model (Ridge regression is just linear regression with L2 regularization)
        self.model = nn.Linear(input_dim, output_dim)

        # Loss function: MSE with L2 regularization (weight_decay equivalent to alpha)
        self.criterion = nn.MSELoss()

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

    def fit(self, X, y):
        """Fit the Ridge regression model using PyTorch."""
        # Convert X and y to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Ensure y is a column vector

        # Move data to GPU if available
        if torch.cuda.is_available():
            X_tensor = X_tensor.to('cuda')
            y_tensor = y_tensor.to('cuda')

        # Define the optimizer with weight decay to apply L2 regularization (alpha)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.alpha)

        # Training loop
        self.model.train()  # Set model to training mode
        for epoch in range(self.epochs):
            optimizer.zero_grad()  # Reset gradients
            outputs = self.model(X_tensor)  # Forward pass
            loss = self.criterion(outputs, y_tensor)  # Calculate the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

    def score(self, X, y):
        """Score the model using a custom scorer."""
        y_pred = self.predict(X)
        return mse_cupy(y, y_pred)  # or another custom metric
    
    def predict(self, X):
        """Make predictions using the trained Ridge regression model."""
        self.model.eval()  # Set model to evaluation mode
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Move data to GPU if available
        if torch.cuda.is_available():
            X_tensor = X_tensor.to('cuda')

        with torch.no_grad():  # Disable gradient calculation for inference
            predictions = self.model(X_tensor).cpu().numpy()  # Move predictions back to CPU
        return predictions

    def get_param_grid(self):
        """Return a parameter grid for hyperparameter tuning."""
        return {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100],
            'lr': [0.001, 0.01],
            'epochs': [100, 200, 300],
            'batch_size': [32, 64]
        }
    
    def get_param_dist(self):
        """Return the parameter distribution for Bayesian optimization."""
        return {
            'alpha': Real(1e-6, 1e2, prior='log-uniform'),
            #'lr': Real(1e-4, 1e-2, prior='log-uniform'),
            #'epochs': Integer(50, 500)
        }

    def get_model(self):
        """Return the PyTorch model instance."""
        return self


class PLSModel(BaseModel):
    """Partial Least Squares Regression model with parameter grid."""

    def __init__(self):
        super().__init__()
        self.model = PLSRegression()
        self.param_grid = {
            'n_components': [1, 2, 3, 5], # 7, 9, 15],
            'max_iter': [1000],
            'tol': [1e-07],
            'scale': [True, False] #, False]
        }
        self.param_dist = {
            'n_components': Integer(1, 5),  # Integer range for number of components
            #'max_iter': Integer(500, 2000),  # Range for max iterations
            #'tol': Real(1e-7, 1e-4, prior='log-uniform'),  # Log-uniform for tolerance
            'scale': Categorical([True, False])  # Whether to scale the data
        }


class XGBModel(BaseModel):
    """XGBoost model with parameter grid."""

    def __init__(self):
        super().__init__()
        self.model = XGBRegressor()
        
        self.param_grid = {
            'n_estimators': [50, 150, 250, 250],  # Num trees
            'max_depth': [2, 3, 5, 7],                 # Maximum depth of each tree
            'learning_rate': [0.01, 0.1, 0.3],         # Learning rate (shrinkage)
            'subsample': [0.6, 0.8, 1],                # Subsample ratio of the training data
            'colsample_bytree': [0.5, 0.8, 1],         # Subsample ratio of columns when constructing each tree
            'gamma': [0, 0.1],                         # Minimum loss reduction required to make a split
            'reg_lambda': [0.01, 0.1, 1],              # L2 regularization term (Ridge penalty)
            'reg_alpha': [0.01, 0.1, 1],               # L1 regularization term (Lasso penalty)
            'random_state': [42],                      # Seed for reproducibility
            'min_child_weight': [1, 3, 5],             # Child weight for pruning
            'tree_method':['gpu_hist'],                    # Use the GPU
            'device':['cuda'],                         # Use GPU predictor
            'n_gpus':[-1],
            'verbosity': [0]
        }
        
        self.param_dist = {
            'learning_rate': Categorical([1e-3, 1e-2, 1e-1, 0.3]), #1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]), #Real(1e-6, 1e-1, prior='log-uniform'),
            'n_estimators': Categorical([50, 150, 250, 350]), # Categorical([10, 100, 300]), #Categorical([50, 100, 150, 200, 250, 300, 350, 400]), # Integer(50, 400)
            'max_depth': Categorical([2, 3, 4, 5, 6, 7]), # Categorical([1, 2, 6]), # Integer(2, 6),  #, 10),
            'subsample': Categorical([0.6, 0.8, 1]), # Categorical([0.6, 0.6]), #, 0.7, 0.8, 0.9, 1.0]),
            'colsample_bytree': Categorical([0.6, 0.8, 1]), # Categorical([0.6, 0.6]), # 0.7, 0.8, 0.9, 1.0]),
            'reg_lambda': Categorical([0, 1e-4, 1e-2, 1e-1, 1]),  # L2 regularization term (Ridge penalty)
            'reg_alpha': Categorical([0, 1e-4, 1e-2, 1e-1, 1]),             # L1 regularization term (Lasso penalty)
            'tree_method': Categorical(['gpu_hist']),
            'device':['cuda'],
            'n_gpus':[-1],
            'random_state': [42],
            'verbosity': [0]
        }
        # consider adding this hyperparam
        # Sampling method. Used only by the GPU version of hist tree method. uniform: select random training instances uniformly. gradient_based select random training instances with higher probability when the gradient and hessian are larger. (cf. CatBoost)
        #


class RandomForestModel(BaseModel):
    """Random Forest model with parameter grid."""

    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor()
        self.param_grid = {
            'n_estimators': [50, 100, 150, 200, 250, 300],  # Number of trees in the forest
            'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum depth of the tree
            'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
            'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
            'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider when looking for the best split
            'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
        }
        self.param_dist = {
            'n_estimators': [50, 100, 150, 200, 250, 300],  # Number of trees in the forest
            'max_depth': randint(10, 50),  # Maximum depth of the tree
            'min_samples_split': randint(2, 11),  # Minimum number of samples required to split an internal node
            'min_samples_leaf': randint(1, 5),  # Minimum number of samples required to be at a leaf node
            'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider when looking for the best split
            'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
        }
        


class MLPModel(BaseEstimator, RegressorMixin):
    """Basic MLP model using PyTorch with support for bayesian hyperparameter tuning."""
    
    def __init__(self, input_dim, output_dim=1, hidden_dims=None, dropout=0.05, l2_reg=1e-4, lr=0.001, epochs=100, batch_size=32,  max_grad_norm=1.0):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else self._default_hidden_dims(input_dim)
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.criterion = nn.MSELoss()

        # Define the architecture dynamically based on input size
        layers = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        

    def _default_hidden_dims(self, input_dim):
        """Private method to define default hidden dimensions based on input size."""
        if input_dim <= 100:
            return [128, 64]
        elif 100 < input_dim <= 300:
            return [256, 128] # [512,256, 128]
        else:
            return [512, 256, 128] # [1024, 512, 256, 128]
    
    def forward(self, x):
        return self.model(x)

    def fit(self, X, y):
        """Train the model with PyTorch."""
        self.model.train()
        #print('model', self.model)
        #print('self params',self.dropout, self.l2_reg, self.lr, self.epochs, self.batch_size, self.criterion)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_reg)
        
        X_tensor = torch.FloatTensor(X) # requires_grad_(True) - can consider using this for gradient tracking
        y_tensor = torch.FloatTensor(y).unsqueeze(1)

        # Move to device
        X_tensor = X_tensor.to(self.device)
        y_tensor = y_tensor.to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False) # keep unshuffled so bidirectional pairs are passed in simultaneously during training
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_epoch_loss:.4f}')

    def predict(self, X):
        """Make predictions with the trained model."""
        self.model.eval()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        if torch.cuda.is_available():
            X_tensor = X_tensor.to('cuda')

        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
        
    def get_param_grid(self):
        """Return a parameter grid for hyperparameter tuning."""
        return {
            #'hidden_dims': [[64, 64], [128, 64], [128, 128, 64]],
            #'dropout': [0.1, 0.3],
            # 'l2_reg': [1e-4, 1e-2, 0],
            # 'lr': [0.001, 0.01, 0.03],
            # 'epochs': [300, 500, 1000], #[50, 100, 300],
            # 'batch_size': [8, 64]

            # single run params for debugging
            'l2_reg': [1e-3], # [1e-3, 0]
            'lr': [1e-3], #, 0.01, 0.03],
            'epochs': [100], #[50, 100, 300],
            'batch_size': [64] # [32, 64]. has to be an even number, 32 works well with 1e_3 learning rate, no reg, dropout 0.2
        }
    
    def get_param_dist(self):
        """Return a parameter distribution for random search or Bayesian optimization."""
        return {
            #'hidden_dims': Categorical([(64, 64)]), # , (128, 64), (128, 128, 64)]),  # Use tuples instead of lists 
            #'dropout': Real(0.0, 0.5),
            # 'l2_reg': Real(0, 1e-0), #, prior='log-uniform'),
            # 'lr': Real(1e-3, 1e-1), # , prior='log-uniform'),
            # 'epochs': Integer(100, 300),
            # 'batch_size': Integer(8, 96)
            
            'dropout': Categorical([0.0, 0.2, 0.3, 0.4, 0.5]),
            'l2_reg': Categorical([0.0, 1e-4, 1e-3, 1e-2]),
            'lr': Categorical([1e-4, 1e-3, 1e-2, 3e-2]),
            'epochs': Categorical([100, 300]),
            'batch_size': Categorical([16, 32, 64])
        }

    def get_model(self):
        """Return the PyTorch model instance."""
        return self
    
    # def score(self, X, y):
    #     """Score the model using a custom scorer."""
    #     y_pred = self.predict(X)
    #     return mse_cupy(y, y_pred)  # or another custom metric


# SUBMODELS
'''
class BilinearSigmoidRegressionModel(nn.Module):
    """Bilinear model with sigmoid activation for bounded output."""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.linear2 = nn.Linear(input_size, output_size, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # Project and apply sigmoid to constrain outputs between 0 and 1
        out1 = self.sigmoid(self.linear(x1))
        out2 = self.sigmoid(self.linear2(x2))
        return torch.matmul(out1, out2.T)

class BilinearReLURegressionModel(nn.Module):
    """Bilinear model with ReLU activation for non-negative outputs."""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.linear2 = nn.Linear(input_size, output_size, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        # Project and apply ReLU to ensure non-negative outputs
        out1 = self.relu(self.linear(x1))
        out2 = self.relu(self.linear2(x2))
        return torch.matmul(out1, out2.T)

class BilinearSoftplusModel(nn.Module):
    """Bilinear model with Softplus activation for smooth non-negative outputs."""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.linear2 = nn.Linear(input_size, output_size, bias=False)
        self.softplus = nn.Softplus()

    def forward(self, x1, x2):
        # Project and apply Softplus for smooth, non-negative outputs
        out1 = self.softplus(self.linear(x1))
        out2 = self.softplus(self.linear2(x2))
        return torch.matmul(out1, out2.T)
'''

class BilinearRegressionModel(nn.Module):
    """Basic bilinear model without activation function."""
    def __init__(self, input_size, output_size):
        super(BilinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.linear2 = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x1, x2):
        # Project both inputs to lower dimensional space
        out1 = self.linear(x1)   # Shape: [batch_size, output_size]
        out2 = self.linear2(x2)  # Shape: [batch_size, output_size]
        # Element-wise multiply and sum across feature dimension
        return torch.sum(out1 * out2, dim=1)  # Shape: [batch_size]

# BASE MODEL
class BilinearModel(BaseEstimator, RegressorMixin):
    """PyTorch-based Bilinear Regression model for predicting connectivity from gene expression."""

    def __init__(self, input_dim, reduced_dim=10, activation='none', lr=0.01, 
                 epochs=100, batch_size=32, lambda_reg=1.0):
        super().__init__()
        self.input_dim = input_dim  # Gene expression dimension
        self.reduced_dim = reduced_dim  # Dimension to project to (k in the paper)
        self.activation = activation  # 'none', 'relu', 'sigmoid', or 'softplus'
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg  # L1 regularization weight

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BilinearRegressionModel(self.input_dim, self.reduced_dim).to(self.device)
        # self._create_model().to(self.device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        print(f"Model architecture:\n{self.model}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Hyperparameters:")
        print(f"  Input dimension: {self.input_dim}")
        print(f"  Reduced dimension: {self.reduced_dim}")
        print(f"  Activation: {self.activation}")
        print(f"  Learning rate: {self.lr}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  L1 regularization weight: {self.lambda_reg}")
        print(f"  Device: {self.device}")

        self.criterion = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()

    def _create_model(self):
        """Create the appropriate bilinear model based on activation type."""
        if self.activation == 'sigmoid':
            return BilinearSigmoidRegressionModel(self.input_dim, self.reduced_dim)
        elif self.activation == 'relu':
            return BilinearReLURegressionModel(self.input_dim, self.reduced_dim)
        elif self.activation == 'softplus':
            return Softplus_Model(self.input_dim, self.reduced_dim)
        else:
            return BilinearRegressionModel(self.input_dim, self.reduced_dim).to(self.device)
    
    def fit(self, X, y):
        """Fit the bilinear model using PyTorch."""
        # Convert inputs to tensors and move to device
        X_tensor = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.as_tensor(y, dtype=torch.float32).to(self.device)
        print(f"X tensor shape: {X_tensor.shape}")
        print(f"y tensor shape: {y_tensor.shape}")
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Define optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Training loop
        self.model.train()
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                # Split features for each edge
                mid = batch_X.size(1) // 2
                optimizer.zero_grad()
                
                # Forward pass
                pred = self.model(batch_X[:, :mid], batch_X[:, mid:])

                # Compute loss with L1 regularization
                mse_loss = self.criterion(pred, batch_y)
                l1_loss = (self.criterion_l1(self.model.linear.weight, torch.zeros_like(self.model.linear.weight)) +
                          self.criterion_l1(self.model.linear2.weight, torch.zeros_like(self.model.linear2.weight)))
                
                loss = mse_loss + self.lambda_reg * l1_loss

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(train_loader):.4f}')


    def predict(self, X):
        # used for trained model
        self.model.eval()
        X_tensor = torch.as_tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            mid = X_tensor.size(1) // 2
            predictions = self.model(X_tensor[:, :mid], X_tensor[:, mid:])
            return predictions.cpu().numpy()

    def get_param_grid(self):
        """Return a parameter grid for hyperparameter tuning."""
        return {
            'reduced_dim': [5, 10], # [5, 10, 20],
            'activation': ['none'], # ['none', 'relu', 'sigmoid', 'softplus'],
            'lr': [0.0001], #[0.001, 0.01],
            'lambda_reg': [0, 0.1], #[0.1, 1.0, 10.0],
            'batch_size': [16] #[32, 64]
        }
    
    def get_param_dist(self):
        """Return the parameter distribution for Bayesian optimization."""
        return {
            'reduced_dim': Integer(5, 50),
            'activation': Categorical(['none', 'relu', 'sigmoid', 'softplus']),
            'lr': Real(1e-4, 1e-2, prior='log-uniform'),
            'lambda_reg': Real(1e-1, 1e2, prior='log-uniform')
        }

    def get_model(self):
        """Return the PyTorch model instance."""
        return self



class ModelBuild:
    """Factory class to create models based on the given model type."""
        
    @staticmethod
    def init_model(model_type, input_size):
        model_mapping = {
            'xgboost': XGBModel,
            'random_forest': RandomForestModel,
            'ridge_torch': RidgeModelTorch,
            'bilinear_baseline': BilinearModel,
            'ridge': RidgeModel,
            'pls': PLSModel, 
            'mlp': MLPModel
        }
    
        if model_type in model_mapping:
            if model_type in ['mlp', 'ridge_torch']:
                print('GPU model input size', input_size)
                return model_mapping[model_type](input_dim=input_size)
            elif model_type == 'bilinear_baseline':
                return model_mapping[model_type](input_dim=int(input_size/2))
            else:
                return model_mapping[model_type]()
        else:
            raise ValueError(f"Model type '{model_type}' is not recognized.")

