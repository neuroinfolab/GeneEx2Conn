from env.imports import *
from data.data_utils import create_data_loader
from models.train_val import train_model
# from models.base_models import BaseModel

class BilinearLoss(nn.Module):
    """MSE loss with optional L1/L2 regularization for bilinear models."""
    def __init__(self, parameters, regularization='l1', lambda_reg=1.0):
        super().__init__()
        self.param_list = list(parameters)
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)
        if self.lambda_reg > 0:
            params = torch.cat([p.view(-1) for p in self.param_list if p.requires_grad])
            if self.regularization == 'l1':
                reg_loss = torch.linalg.norm(params, ord=1)
            elif self.regularization == 'l2':
                reg_loss = torch.linalg.norm(params, ord=2)
            return mse_loss + self.lambda_reg * reg_loss
        return mse_loss

class BilinearLowRank(nn.Module):
    def __init__(self, input_dim, binarize=None, reduced_dim=10, activation='none', learning_rate=0.01, epochs=100, 
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
        print(f"Number of learnable parameters in Bilinear low rank model: {num_params}")
        print(f"  Linear layer 1: {self.linear.weight.numel()} parameters")
        if not self.shared_weights:
            print(f"  Linear layer 2: {self.linear2.weight.numel()} parameters")
        
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        else:  # 'none'
            self.activation = nn.Identity()

        self.criterion = BilinearLoss(self.parameters(), regularization=regularization, lambda_reg=lambda_reg)
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        
        self.patience = 40
        self.scheduler = ReduceLROnPlateau( 
            self.optimizer, 
            mode='min', 
            factor=0.3,  # Reduce LR by 70%
            patience=25,  # Reduce LR after patience epochs of no improvement
            threshold=0.05,  # Threshold to detect stagnation
            cooldown=1,  # Reduce cooldown period
            min_lr=1e-6,  # Prevent LR from going too low
            verbose=True
        )
    
    def forward(self, x):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        out1 = self.activation(self.linear(x_i))
        out2 = self.activation(self.linear(x_j) if self.shared_weights else self.linear2(x_j))
        return torch.sum(out1 * out2, dim=1) # dot product for paired samples
    
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
        return predictions, targets
    
    def fit(self, dataset, train_indices, test_indices, verbose=True):
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, self.scheduler, verbose=verbose)

class BilinearCM(nn.Module):
    def __init__(self, input_dim, binarize=None,learning_rate=0.01, epochs=100, 
                 batch_size=128, regularization='l2', lambda_reg=1.0, bias=True, closed_form=True):
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.closed_form = closed_form
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bilinear = nn.Bilinear(input_dim//2, input_dim//2, 1, bias=bias)

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total number of learnable parameters in BilinearCMmodel: {num_params}")
        
        self.criterion = BilinearLoss(self.parameters(), regularization=regularization, lambda_reg=lambda_reg)
        self.optimizer = Adam(self.parameters(), lr=learning_rate)

        self.patience = 30
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
    
    def forward(self, x):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        return self.bilinear(x_i, x_j).squeeze()

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
        return predictions, targets
    
    def fit(self, dataset, train_indices, test_indices, verbose=True):
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        if self.closed_form:
            return self.fit_closed_form(dataset, train_indices, test_indices, train_loader, test_loader)
        else: 
            return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.patience, self.scheduler, self.optimizer)

    def fit_full(self, dataset, verbose=True):
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False)
        return train_model(self, train_loader, None, self.epochs, self.criterion, self.patience, self.scheduler, self.optimizer)
    
    def fit_closed_form(self, dataset, train_indices, test_indices, train_loader, test_loader):
        """
        Fit model using closed form solution in original matrix form.
        Maps expanded indices back to valid region pairs for efficient computation.
        """
        # Get original full matrices
        X_full = dataset.X.to(self.device)
        Y_full = dataset.Y.to(self.device)
        
        # Map expanded indices to valid region pairs and get unique region indices
        train_pairs = [dataset.expanded_idx_to_valid_pair[idx] for idx in train_indices]
        test_pairs = [dataset.expanded_idx_to_valid_pair[idx] for idx in test_indices]
        train_indices_set = sorted(list(set(idx for pair in train_pairs for idx in pair)))
        test_indices_set = sorted(list(set(idx for pair in test_pairs for idx in pair)))

        # Partition matrices based on region sets
        X = X_full[train_indices_set]
        Y = Y_full[train_indices_set][:,train_indices_set]
        X_test = X_full[test_indices_set]
        Y_test = Y_full[test_indices_set][:,test_indices_set]

        # Compute inverse of regularized Gram matrix
        A = torch.mm(X.T, X) + self.lambda_reg * torch.eye(X.shape[1], device=self.device)
        A_inv = torch.linalg.inv(A)
        
        # Estimate O from closed-form solution
        O = torch.mm(A_inv, torch.mm(X.T, torch.mm(Y, torch.mm(X, A_inv))))
        
        # Compute bilinear prediction
        Y_lin = torch.mm(torch.mm(X, O), X.T)
        Y_lin_test = torch.mm(torch.mm(X_test, O), X_test.T)
        
        # Learn scalar bias to minimize mean residual
        b = torch.mean(Y - Y_lin)
        
        # Map parameters to bilinear layer
        O_reshaped = O.reshape(1, X.shape[1], X.shape[1])
        self.bilinear.weight.data = O_reshaped
        if self.bilinear.bias is not None:
            self.bilinear.bias.data = b.reshape(1)
        
        # Get predictions using dataloaders
        train_predictions = []
        train_targets = []
        for batch_X, batch_y, _, _ in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_preds = self.forward(batch_X)
            train_predictions.append(batch_preds)
            train_targets.append(batch_y)
            
        test_predictions = []
        test_targets = []
        for batch_X, batch_y, _, _ in test_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device) 
            batch_preds = self.forward(batch_X)
            test_predictions.append(batch_preds)
            test_targets.append(batch_y)
            
        Y_train_pred = torch.cat(train_predictions)
        Y_train = torch.cat(train_targets)
        Y_test_pred = torch.cat(test_predictions)
        Y_test = torch.cat(test_targets)
        
        # Compute losses
        train_loss = self.criterion(Y_train_pred, Y_train).item()
        val_loss = self.criterion(Y_test_pred, Y_test).item()
        
        return {'train_loss': [train_loss], 'val_loss': [val_loss]}