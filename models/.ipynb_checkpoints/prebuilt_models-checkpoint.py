# Gene2Conn/models/prebuilt_models.py

from imports import *

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
            'alpha': [0.01, 0.1, 0.5, 1.0, 10, 100],
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
            
        self.param_grid = {
            'n_estimators': [50, 100, 150, 200, 250],
            'max_depth': [2, 3, 5, 7],           # Maximum depth of each tree - makes a big diff
            'learning_rate': [0.01, 0.1, 0.3],     # Learning rate (shrinkage)
            'subsample': [0.6, 0.8, 1],              # Subsample ratio of the training data
            'colsample_bytree': [0.5, 0.8, 1],  # Subsample ratio of columns when constructing each tree
            'gamma': [0, 0.1],             # Minimum loss reduction required to make a split
            'reg_lambda': [0.01, 0.1, 1],              # L2 regularization term (Ridge penalty)
            'reg_alpha': [0.01, 0.1, 1],             # L1 regularization term (Lasso penalty)
            'random_state': [42],        # Seed for reproducibility
            'min_child_weight': [1, 3, 5], 
            'tree_method':['hist'],  # Use the GPU
            'device':['cuda'],  # Use GPU predictor
            'verbosity': [2]
        }

        # syntax to specify params for a fine tuned run
        '''
        best_params = {
            'n_estimators': [250],
            'max_depth': [3],           # Maximum depth of each tree - makes a big diff
            'learning_rate': [0.01],     # Learning rate (shrinkage)
            'subsample': [0.6],              # Subsample ratio of the training data
            'colsample_bytree': [0.8],  # Subsample ratio of columns when constructing each tree
            'gamma': [0],             # Minimum loss reduction required to make a split
            'reg_lambda': [.01],              # L2 regularization term (Ridge penalty)
            'reg_alpha': [0.01],             # L1 regularization term (Lasso penalty)
            'random_state': [42],        # Seed for reproducibility
            'min_child_weight': [1], 
            'tree_method':['hist'],  # Use the GPU
            'device':['cuda'],  # Use GPU predictor
            'verbosity': [2]
        }
        self.param_grid = best_params
        '''
        '''
        consider adding this hyperparam
        Sampling method. Used only by the GPU version of hist tree method. uniform: select random training instances uniformly. gradient_based select random training instances with higher probability when the gradient and hessian are larger. (cf. CatBoost)
        '''
        
        # can also specify distributions
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
            'tree_method': ['hist'],  # Use the GPU
            'device': ['cuda'],  # Use GPU predictor
            'verbosity': [2]  # Verbosity level
        }'''


class LGBMModel(BaseModel):
    """LightGBM model with parameter grid and distribution."""

    def __init__(self):
        super().__init__()
        self.model = LGBMRegressor(device='gpu')  # Use GPU for training
        self.param_grid = {
            'num_leaves': [31, 50, 70],  # number of leaves in one tree
            'learning_rate': [0.01, 0.1, 0.2],  # shrinkage rate
            'n_estimators': [100, 200, 300],  # number of boosting rounds
            'boosting_type': ['gbdt', 'dart'],  # type of boosting algorithm
            'subsample': [0.8, 0.9, 1.0],  # fraction of data used to train each base learner
            'colsample_bytree': [0.8, 0.9, 1.0],  # fraction of features used for each base learner
            'reg_alpha': [0, 0.1, 0.5],  # L1 regularization term
            'reg_lambda': [0, 0.1, 0.5]  # L2 regularization term
        }
        self.param_dist = {
            'num_leaves': randint(20, 150),  # number of leaves in one tree
            'learning_rate': uniform(0.01, 0.3),  # shrinkage rate
            'n_estimators': randint(100, 300),  # number of boosting rounds
            'boosting_type': ['gbdt', 'dart'],  # type of boosting algorithm
            'subsample': uniform(0.7, 0.3),  # fraction of data used to train each base learner
            'colsample_bytree': uniform(0.7, 0.3),  # fraction of features used for each base learner
            'reg_alpha': uniform(0, 1),  # L1 regularization term
            'reg_lambda': uniform(0, 1)  # L2 regularization term
        }


class DNNModel(BaseModel):
    """Deep Neural Network model with parameter grid and distribution."""

    def __init__(self):
        super().__init__()
        self.model = KerasRegressor(build_fn=self.create_model, verbose=0)
        self.param_grid = {
            'batch_size': [16, 32, 64],
            'epochs': [10, 50, 100],
            'optimizer': ['adam', 'rmsprop'],
            'neurons': [32, 64, 128],
            'dropout_rate': [0.2, 0.3, 0.4]
        }
        self.param_dist = {
            'batch_size': [16, 32, 64],
            'epochs': randint(10, 100),
            'optimizer': ['adam', 'rmsprop'],
            'neurons': randint(32, 128),
            'dropout_rate': uniform(0.2, 0.2)
        }

    def create_model(self, optimizer='adam', neurons=32, dropout_rate=0.2):
        model = Sequential()
        model.add(Dense(neurons, input_dim=self.input_dim, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(neurons, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model


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

