# Gene2Conn/models/prebuilt_models.py

from imports import *
from skopt.space import Real, Categorical, Integer
input_dim = None

from metrics.eval import mse_cupy

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
            'alpha': [0, 0.001, 0.01, 0.1, 1.0, 10, 100],
            # 'alpha': np.logspace(-6, 2, 9)  # Explore 10^-6 to 10^2
            'solver': ['auto'] #, 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }
        self.param_dist = {
            'alpha': Real(1e-6, 1e2, prior='log-uniform')  # Log-uniform to explore a wide range of alphas
            #'solver': Categorical(['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])  # Different solvers to try
        }
        

class PLSModel(BaseModel):
    """Partial Least Squares Regression model with parameter grid."""

    def __init__(self):
        super().__init__()
        self.model = PLSRegression()
        self.param_grid = {
            'n_components': [1, 3, 5, 7, 9], #, 15],
            'max_iter': [1000],
            'tol': [1e-07], 
            'scale': [True, False] #, False]
        }
        self.param_dist = {
            'n_components': Integer(1, 8),  # Integer range for number of components
            #'max_iter': Integer(500, 2000),  # Range for max iterations
            'tol': Real(1e-7, 1e-4, prior='log-uniform'),  # Log-uniform for tolerance
            'scale': Categorical([True, False])  # Whether to scale the data
        }


class XGBModel(BaseModel):
    """XGBoost model with parameter grid."""

    def __init__(self):
        super().__init__()
        self.model = XGBRegressor()
        '''
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
        '''
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
        
        # syntax to specify params for a fine tuned run
        best_params = {
            'n_estimators': [200],
            'max_depth': [5],           # Maximum depth of each tree - makes a big diff
            'learning_rate': [0.01],     # Learning rate (shrinkage)
            'subsample': [1],              # Subsample ratio of the training data
            'colsample_bytree': [1],  # Subsample ratio of columns when constructing each tree
            'gamma': [0],             # Minimum loss reduction required to make a split
            'reg_lambda': [0],              # L2 regularization term (Ridge penalty)
            'reg_alpha': [0],             # L1 regularization term (Lasso penalty)
            'random_state': [42],        # Seed for reproducibility
            'min_child_weight': [1], 
            'tree_method':['hist'],  # Use the GPU
            'device':['cuda'],  # Use GPU predictor
            'verbosity': [2]
        }
        self.param_grid = best_params
        
        '''
        self.param_dist = {
            'n_estimators': [50, 100, 150, 200, 250, 300],  # Number of trees in the forest
            'max_depth': randint(3, 10),  # Maximum depth of each tree
            'learning_rate': uniform(0.01, 0.3),  # Learning rate (shrinkage)
            'subsample': uniform(0.6, 0.4),  # Subsample ratio of the training data
            'colsample_bytree': uniform(0.5, 0.5),  # Subsample ratio of columns when constructing each tree
            'gamma': uniform(0, 0.3),  # Minimum loss reduction required to make a split
            'reg_lambda': uniform(0.01, 1),  # L2 regularization term (Ridge penalty)
            'reg_alpha': uniform(0.01, 1),  # L1 regularization term (Lasso penalty)
            'random_state': [42],  # Seed for reproducibility
            'min_child_weight': randint(1, 6),  # Minimum sum of instance weight needed in a child
            'tree_method': ['gpu_hist'],  # Use the GPU
            'device': ['cuda'],  # Use GPU predictor
            'n_gpus':[-1],
            'verbosity': [2]  # Verbosity level
        }
        
        # consider adding this hyperparam
        # Sampling method. Used only by the GPU version of hist tree method. uniform: select random training instances uniformly. gradient_based select random training instances with higher probability when the gradient and hessian are larger. (cf. CatBoost)
        #
        '''


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


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin

class MLPModel(BaseEstimator, RegressorMixin):
    """Basic MLP model using PyTorch with support for L2 regularization and dropout."""
    
    def __init__(self, input_dim, output_dim=1, hidden_dims=[64, 64], dropout=0.5, l2_reg=1e-4, lr=0.001, epochs=100, batch_size=32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        # Define the architecture
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

        # Loss function
        self.criterion = nn.MSELoss()
        '''
        self.param_dist = {
        #'hidden_dims': Categorical([(64, 64)]), # , (128, 64), (128, 128, 64)]),  # Use tuples instead of lists
        'dropout': Real(0.0, 0.2, 0.5),
        'l2_reg': Real(1e-4, 1e-2),
        'lr': Real(1e-4, 1e-2),
        'epochs': Integer(50, 1000),
        'batch_size': Integer(50, 1000)
        }
        '''

    def fit(self, X, y):
        """Train the model with PyTorch."""
        self.model.train()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            X_tensor = X_tensor.to('cuda')
            y_tensor = y_tensor.to('cuda')

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_reg)

        # Training loop
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

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
            'hidden_dims': [[64, 64], [128, 64], [128, 128, 64]],
            'dropout': [0.2, 0.5],
            'l2_reg': [1e-4, 1e-3, 1e-2],
            'lr': [0.001, 0.01],
            'epochs': [50, 100],
            'batch_size': [32, 64]
        }
    
    def get_param_dist(self):
        """Return a parameter distribution for random search or Bayesian optimization."""
        return {
            #'hidden_dims': Categorical([(64, 64)]), # , (128, 64), (128, 128, 64)]),  # Use tuples instead of lists
            'dropout': Real(0.2, 0.5), 
            'l2_reg': Real(1e-5, 1e-3, prior='log-uniform'),  
            'lr': Real(1e-4, 1e-2, prior='log-uniform'), 
            'epochs': Integer(100, 500),  
            'batch_size': Integer(16, 128)  
        }

    def get_model(self):
        """Return the PyTorch model instance."""
        return self

    def score(self, X, y):
        return mse_cupy(X, y)


class ModelBuild:
    """Factory class to create models based on the given model type."""
        
    @staticmethod
    def init_model(model_type, input_size):
        model_mapping = {
            'xgboost': XGBModel,
            'random_forest': RandomForestModel,
            'ridge': RidgeModel,
            'pls': PLSModel, 
            'mlp': MLPModel  # Add the MLP model here
        }
    
        if model_type in model_mapping:
            if model_type == 'mlp':
                print('MLP input size', input_size)
                return model_mapping[model_type](input_dim=input_size)
            else:
                return model_mapping[model_type]()
        else:
            raise ValueError(f"Model type '{model_type}' is not recognized.")

