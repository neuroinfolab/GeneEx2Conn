# Gene2Conn/models/prebuilt_models.py

from imports import *
from skopt.space import Real, Categorical, Integer

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
            'alpha': [0, 0.01, 0.1, 0.5, 1.0, 10, 100],
            'solver': ['auto'] #, 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }


class PLSModel(BaseModel):
    """Partial Least Squares Regression model with parameter grid."""

    def __init__(self):
        super().__init__()
        self.model = PLSRegression()
        self.param_grid = {
            'n_components': [1, 3, 5, 7, 9, 15],
            'max_iter': [1000],
            'tol': [1e-07], 
            'scale': [True, False] #, False]
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
            'max_depth': Categorical([2, 4, 5, 7]), # Categorical([1, 2, 6]), # Integer(2, 6),  #, 10),
            'subsample': Categorical([0.8, 1]), # Categorical([0.6, 0.6]), #, 0.7, 0.8, 0.9, 1.0]),
            'colsample_bytree': Categorical([0.8, 1]), # Categorical([0.6, 0.6]), # 0.7, 0.8, 0.9, 1.0]),
            'reg_lambda': Categorical([0, 1e-2, 1e-1, 1]),  # L2 regularization term (Ridge penalty)
            'reg_alpha': Categorical([0, 1e-2, 1e-1, 1]),             # L1 regularization term (Lasso penalty)
            'tree_method': Categorical(['gpu_hist']),
            'device':['cuda'],
            'n_gpus':[-1],
            'random_state': [42],
            'verbosity': [0]
        }
        
        # syntax to specify params for a fine tuned run
        best_params = {
            'n_estimators': [250],
            'max_depth': [3],           # Maximum depth of each tree - makes a big diff
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


class ModelBuild:
    """Factory class to create models based on the given model type."""

    @staticmethod
    def init_model(model_type):
        model_mapping = {
            'xgboost': XGBModel,
            'random_forest': RandomForestModel,
            'ridge': RidgeModel,
            'pls': PLSModel
        }
        
        if model_type in model_mapping:
            return model_mapping[model_type]()
        else:
            raise ValueError(f"Model type '{model_type}' is not recognized.")

