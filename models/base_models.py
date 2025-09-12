from env.imports import *
from models.train_val import train_model

class BaseModel(nn.Module):
    """Base class for PyTorch models with default fit and predict logic."""
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    @staticmethod
    def predict(self, loader, binarize=None):
        self.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for batch_X, batch_y, *_ in loader:
                batch_X = batch_X.to(self.device)
                batch_preds = self(batch_X).cpu().numpy()
                predictions.append(batch_preds)
                targets.append(batch_y.numpy())
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        return predictions, targets
    
    @staticmethod
    def fit(self, dataset, train_indices, test_indices, criterion, optimizer, scheduler=None, epochs=100, batch_size=512, verbose=True):
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, self.scheduler, verbose=verbose)

class LinearModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()
        self.param_grid = {
            'fit_intercept': [True, False],
            'positive': [True, False]
        }
        self.param_dist = {
            'fit_intercept': Categorical([True, False]),
            'positive': Categorical([True, False])
        }
    
    def get_model(self):
        """Return the sklearn model for GridSearchCV compatibility"""
        return self.model

class RidgeModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = Ridge()
        self.param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000],
            'solver': ['auto'] #, 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }
        self.param_dist = {
            'alpha': Real(1e-6, 1e2, prior='log-uniform'),  # Log-uniform to explore a wide range of alphas
            'solver': Categorical(['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])  # Different solvers to try
        }
    
    def get_model(self):
        """Return the sklearn model for GridSearchCV compatibility"""
        return self.model

class PLSModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = PLSRegression()
        self.param_grid = {
            'n_components': [1, 2, 3, 4], # [1, 2, 3, 4, 5, 7, 10], # 7, 9, 15],
            'max_iter': [1000],
            'tol': [1e-07],
            'scale': [True] #, False]
        }
        self.param_dist = {
            'n_components': Categorical([1, 2, 3, 4, 5, 7, 10]),  # Integer range for number of components
            'scale': Categorical([True, False])  # Whether to scale the data
        }
    
    def get_model(self):
        """Return the sklearn model for GridSearchCV compatibility"""
        return self.model

class XGBRegressorModel(nn.Module):
    def __init__(self, input_dim=None, n_estimators=100, max_depth=5, learning_rate=0.1, 
                 subsample=0.8, colsample_bytree=0.8, gamma=0, reg_lambda=0.1, reg_alpha=0.01,
                 min_child_weight=3, batch_size=512, epochs=1, **kwargs):
        super().__init__()
        
        # Torch-style parameters
        self.batch_size = batch_size
        self.epochs = epochs  # For XGBoost, this is just 1 epoch since it's not iterative like neural nets
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # XGBoost handles its own device management, so we don't move parameters
        
        # XGBoost parameters
        xgb_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'reg_lambda': reg_lambda,
            'reg_alpha': reg_alpha,
            'min_child_weight': min_child_weight,
            'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
            'random_state': 42,
            'verbosity': 0
        }
        xgb_params.update(kwargs)
        
        self.model = XGBRegressor(**xgb_params)
        self.criterion = nn.MSELoss()  # For compatibility with torch interface
    
    def to(self, device):
        """Override to() method since XGBoost handles GPU internally"""
        # Call parent's to() method to maintain nn.Module behavior
        super().to(device)
        self.device = device
        return self
        
    def forward(self, x):
        """Forward pass - not used for XGBoost but required for nn.Module"""
        # Convert torch tensor to numpy if needed
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return torch.tensor(self.model.predict(x), dtype=torch.float32)
    
    def fit(self, dataset, train_indices, test_indices, save_model=None, verbose=True):
        """Torch-style fit method compatible with your training pipeline"""
        # Extract data from dataset
        if hasattr(dataset, 'X_expanded') and hasattr(dataset, 'Y_expanded'):
            X_train = dataset.X_expanded[train_indices]
            y_train = dataset.Y_expanded[train_indices]
            X_test = dataset.X_expanded[test_indices] 
            y_test = dataset.Y_expanded[test_indices]
        else:
            # Fallback for different dataset structures
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            test_dataset = torch.utils.data.Subset(dataset, test_indices)
            
            # Extract all training data
            X_train_list, y_train_list = [], []
            for x, y, *_ in train_dataset:
                X_train_list.append(x.numpy() if isinstance(x, torch.Tensor) else x)
                y_train_list.append(y.numpy() if isinstance(y, torch.Tensor) else y)
            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)
            
            # Extract all test data
            X_test_list, y_test_list = [], []
            for x, y, *_ in test_dataset:
                X_test_list.append(x.numpy() if isinstance(x, torch.Tensor) else x)
                y_test_list.append(y.numpy() if isinstance(y, torch.Tensor) else y)
            X_test = np.vstack(X_test_list)
            y_test = np.concatenate(y_test_list)
        
        # Convert torch tensors to numpy
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.cpu().numpy()
            y_train = y_train.cpu().numpy()
            X_test = X_test.cpu().numpy()
            y_test = y_test.cpu().numpy()
        
        # Fit XGBoost model
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=verbose)
        
        # Calculate losses for torch-style return
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_loss = np.mean((train_pred - y_train) ** 2)
        val_loss = np.mean((test_pred - y_test) ** 2)
        
        return {'train_loss': [train_loss], 'val_loss': [val_loss]}
    
    def predict(self, loader):
        """Torch-style predict method compatible with your evaluation pipeline"""
        predictions = []
        targets = []
        
        for batch_data in loader:
            batch_X, batch_y = batch_data[0], batch_data[1]
            
            # Convert to numpy
            if isinstance(batch_X, torch.Tensor):
                batch_X = batch_X.cpu().numpy()
            if isinstance(batch_y, torch.Tensor):
                batch_y = batch_y.cpu().numpy()
            
            # Make predictions
            batch_preds = self.model.predict(batch_X)
            
            predictions.append(batch_preds)
            targets.append(batch_y)
        
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        return predictions, targets


class XGBClassifierModel(nn.Module):
    def __init__(self, input_dim=None, n_estimators=100, max_depth=5, learning_rate=0.1,
                 subsample=0.8, colsample_bytree=0.8, gamma=0, reg_lambda=0.1, reg_alpha=0.01,
                 min_child_weight=3, scale_pos_weight=1, batch_size=512, epochs=1, **kwargs):
        super().__init__()
        
        # Torch-style parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # XGBoost parameters
        xgb_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'reg_lambda': reg_lambda,
            'reg_alpha': reg_alpha,
            'min_child_weight': min_child_weight,
            'scale_pos_weight': scale_pos_weight,
            'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
            'random_state': 42,
            'verbosity': 0,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        xgb_params.update(kwargs)
        
        self.model = XGBClassifier(**xgb_params)
        self.criterion = nn.CrossEntropyLoss()  # For compatibility with torch interface
    
    def to(self, device):
        """Override to() method since XGBoost handles GPU internally"""
        # Call parent's to() method to maintain nn.Module behavior
        super().to(device)
        self.device = device
        return self
        
    def forward(self, x):
        """Forward pass - not used for XGBoost but required for nn.Module"""
        # Convert torch tensor to numpy if needed
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        # Return probabilities as torch tensor
        probs = self.model.predict_proba(x)
        return torch.tensor(probs, dtype=torch.float32)
    
    def fit(self, dataset, train_indices, test_indices, save_model=None, verbose=True):
        """Torch-style fit method compatible with your training pipeline"""
        # Extract data from dataset
        if hasattr(dataset, 'X_expanded') and hasattr(dataset, 'Y_expanded'):
            X_train = dataset.X_expanded[train_indices]
            y_train = dataset.Y_expanded[train_indices]
            X_test = dataset.X_expanded[test_indices]
            y_test = dataset.Y_expanded[test_indices]
        else:
            # Fallback for different dataset structures
            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            test_dataset = torch.utils.data.Subset(dataset, test_indices)
            
            # Extract all training data
            X_train_list, y_train_list = [], []
            for x, y, *_ in train_dataset:
                X_train_list.append(x.numpy() if isinstance(x, torch.Tensor) else x)
                y_train_list.append(y.numpy() if isinstance(y, torch.Tensor) else y)
            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)
            
            # Extract all test data
            X_test_list, y_test_list = [], []
            for x, y, *_ in test_dataset:
                X_test_list.append(x.numpy() if isinstance(x, torch.Tensor) else x)
                y_test_list.append(y.numpy() if isinstance(y, torch.Tensor) else y)
            X_test = np.vstack(X_test_list)
            y_test = np.concatenate(y_test_list)
        
        # Convert torch tensors to numpy
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.cpu().numpy()
            y_train = y_train.cpu().numpy()
            X_test = X_test.cpu().numpy()
            y_test = y_test.cpu().numpy()
        
        # Fit XGBoost model
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=verbose)
        
        # Calculate losses for torch-style return
        train_pred_proba = self.model.predict_proba(X_train)[:, 1]  # Get positive class probabilities
        test_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Binary cross-entropy loss
        train_loss = -np.mean(y_train * np.log(train_pred_proba + 1e-15) + 
                             (1 - y_train) * np.log(1 - train_pred_proba + 1e-15))
        val_loss = -np.mean(y_test * np.log(test_pred_proba + 1e-15) + 
                           (1 - y_test) * np.log(1 - test_pred_proba + 1e-15))
        
        return {'train_loss': [train_loss], 'val_loss': [val_loss]}
    
    def predict(self, loader):
        """Torch-style predict method compatible with your evaluation pipeline"""
        predictions = []
        targets = []
        
        for batch_data in loader:
            batch_X, batch_y = batch_data[0], batch_data[1]
            
            # Convert to numpy
            if isinstance(batch_X, torch.Tensor):
                batch_X = batch_X.cpu().numpy()
            if isinstance(batch_y, torch.Tensor):
                batch_y = batch_y.cpu().numpy()
            
            # Make predictions (probabilities)
            batch_preds = self.model.predict_proba(batch_X)[:, 1]  # Positive class probabilities
            
            predictions.append(batch_preds)
            targets.append(batch_y)
        
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        return predictions, targets

class SVCModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = SVC()
        
        self.param_grid = {
            'C': [0.1, 1, 10, 20],                # Regularization parameter
            'kernel': ['linear'],
            # 'rbf'           # Kernel type 
            # 'gamma': ['scale', 'auto', 0.1, 1],    # Kernel coefficient
            'class_weight': ['balanced'],     # Class weights
            'random_state': [42]                    # Random seed
        }
        
        self.param_dist = {
            'C': Categorical([0.1, 1, 10]),
            'kernel': Categorical(['linear']),
            'gamma': Categorical(['scale', 'auto']),
            'class_weight': Categorical(['balanced']),
            # 'random_state': [42]
        }
    
    def get_model(self):
        """Return the sklearn model for GridSearchCV compatibility"""
        return self.model

class RandomForestModel(BaseModel):
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
    
    def get_model(self):
        """Return the sklearn model for GridSearchCV compatibility"""
        return self.model

class ModelBuild:
    """Factory class to create base models based on the given model type."""
    @staticmethod
    def init_model(model_type, binarize=None, **kwargs):
        if binarize:
            model_mapping = {
                'svm': SVCModel,
                'xgboost': XGBClassifierModel
            }
        else:
            model_mapping = {
                'pls': PLSModel,
                'ridge': RidgeModel,
                'linear': LinearModel,
                'xgboost': XGBRegressorModel,
                'random_forest': RandomForestModel
            }

        if model_type in model_mapping:
            model_class = model_mapping[model_type]
            # XGBoost models now have torch-style constructors
            if model_type == 'xgboost':
                return model_class(**kwargs)
            else:
                return model_class()
        else:
            raise ValueError(f"Model type '{model_type}' is not recognized.")