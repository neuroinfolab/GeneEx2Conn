from env.imports import *
from models.train_val import train_model
from models.base_models import BaseModel
from data.data_load import load_transcriptome, load_connectome
from data.data_utils import expand_X_symmetric, expand_Y_symmetric

class PLSEncoder(nn.Module):
    def __init__(self, train_indices, test_indices, region_pair_dataset, n_components=10, max_iter=1000, scale=True, optimize_encoder=False, device=None):
        super().__init__()
        
        self.n_components = n_components
        self.max_iter = max_iter
        self.scale = scale
        self.optimize_encoder = optimize_encoder
        self.region_pair_dataset = region_pair_dataset
        self.device = device
        self.X = region_pair_dataset.X
        self.Y = region_pair_dataset.Y

        # Subset to train indices
        X_train = self.X[train_indices]
        Y_train = self.Y[train_indices][:, train_indices]
        print(f"X_train shape: {X_train.shape}")
        print(f"Y_train shape: {Y_train.shape}")

        # Fit PLS model only on train data
        self.pls_model = PLSRegression(n_components=n_components, max_iter=max_iter, scale=scale)
        self.pls_model.fit(X_train, Y_train)
        
        # Save projection matrices as torch parameters
        # x_weights_ for predictive importance, x_loadings_ for correlation-based interpretability, x_rotations_ for learned orthonormal projection
        if self.optimize_encoder == True:
            # Optimize on x_weights_ since more aligned with biological interpretation and prediction task
            self.x_projector = nn.Parameter(torch.FloatTensor(self.pls_model.x_weights_).to(self.device), requires_grad=True)
            self.x_loadings = nn.Parameter(torch.FloatTensor(self.pls_model.x_loadings_).to(self.device), requires_grad=True)
        else: 
            # This is fixed like what is done in the literature and how sklearn projects, https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73/sklearn/cross_decomposition/_pls.py#L564
            self.x_projector = nn.Parameter(torch.FloatTensor(self.pls_model.x_rotations_).to(self.device), requires_grad=False)
            self.x_loadings = nn.Parameter(torch.FloatTensor(self.pls_model.x_loadings_).to(self.device), requires_grad=False)
            X_tensor = torch.FloatTensor(self.X).to(self.device)
            self.cached_projection = torch.matmul(X_tensor, self.x_projector) # shape: full dataset x num components
        
    def forward(self, x, expanded_idx):
        if self.optimize_encoder == True:
            x_i, x_j = torch.chunk(x, 2, dim=1)
            x_scores_i = torch.matmul(x_i, self.x_projector)
            x_scores_j = torch.matmul(x_j, self.x_projector)
        else:
            # precomputed projection - slightly faster on a100/h100
            idxs = expanded_idx.view(-1).tolist()
            region_pairs = np.array([self.region_pair_dataset.expanded_idx_to_valid_pair[idx] for idx in idxs])
            region_i = region_pairs[:, 0]
            region_j = region_pairs[:, 1]
            x_scores_i = self.cached_projection[region_i]
            x_scores_j = self.cached_projection[region_j]

            # project each forward batch - slightly faster on v100 
            # x_i, x_j = torch.chunk(x, 2, dim=1)
            # x_scores_i = torch.matmul(x_i, self.x_projector)
            # x_scores_j = torch.matmul(x_j, self.x_projector)
        
        return x_scores_i, x_scores_j

class PLS_BilinearDecoderModel(BaseModel):
    def __init__(self, input_dim, train_indices, test_indices, region_pair_dataset,
                 binarize=False, n_components=10, max_iter=1000, scale=True, optimize_encoder=False,
                 learning_rate=0.0001, weight_decay=0.0001, batch_size=512, epochs=100, closed_form=True):
        super().__init__()
        
        self.binarize = binarize
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimize_encoder = optimize_encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_reg = weight_decay
        self.closed_form = closed_form
        
        # Initialize PLS encoder
        self.encoder = PLSEncoder(
            train_indices=train_indices,
            test_indices=test_indices,
            region_pair_dataset=region_pair_dataset,
            n_components=n_components,
            max_iter=max_iter,
            scale=scale,
            optimize_encoder=optimize_encoder, 
            device=self.device)
            
        self.criterion = nn.MSELoss() # nn.BCEWithLogitsLoss() if binarize else 

        # Initialize decoder
        self.bilinear = nn.Bilinear(n_components, n_components, 1, bias=True)
        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters())}")

        self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)            
        self.patience = 30
        self.scheduler = ReduceLROnPlateau( 
            self.optimizer, 
            mode='min', 
            factor=0.3,  # Reduce LR by 70%
            patience=20,  # Reduce LR after patience epochs of no improvement
            threshold=0.05,  # Threshold to detect stagnation
            cooldown=1,  # Reduce cooldown period
            min_lr=1e-6,  # Prevent LR from going too low
            verbose=True)
        
    def forward(self, x, idx):
        encoded_i, encoded_j = self.encoder(x, idx)
        output = self.bilinear(encoded_i, encoded_j)
        return output.squeeze()
    
    def predict(self, loader):
        self.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for batch_X, batch_y, _, batch_idx in loader:
                batch_X = batch_X.to(self.device)
                batch_preds = self(batch_X, batch_idx).cpu().numpy()
                predictions.append(batch_preds)
                targets.append(batch_y.numpy())
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        return ((predictions > 0.5).astype(int) if self.binarize else predictions), targets
    
    def fit(self, dataset, expanded_train_indices, expanded_test_indices, verbose=True):
        train_dataset = Subset(dataset, expanded_train_indices)
        test_dataset = Subset(dataset, expanded_test_indices)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        if self.closed_form:
            return self.fit_closed_form(dataset, expanded_train_indices, expanded_test_indices, train_loader, test_loader)
        else:
            return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, self.scheduler, verbose=verbose)

    def fit_closed_form(self, dataset, train_indices, test_indices, train_loader, test_loader):
        """
        Fit model using closed form solution in original matrix form.
        First encodes data using PLS then solves for bilinear weights.
        """
        # Map expanded indices to valid region pairs and get unique region indices
        train_pairs = [dataset.expanded_idx_to_valid_pair[idx] for idx in train_indices]
        test_pairs = [dataset.expanded_idx_to_valid_pair[idx] for idx in test_indices]
        train_indices_set = sorted(list(set(idx for pair in train_pairs for idx in pair)))
        test_indices_set = sorted(list(set(idx for pair in test_pairs for idx in pair)))

        # Get encoded matrices using PLS encoder
        with torch.no_grad():
            X_full = self.encoder.cached_projection  # Already encoded and cached during PLS init
            X = X_full[train_indices_set]
            X_test = X_full[test_indices_set]
            
            # Get original Y matrices
            Y = dataset.Y[train_indices_set][:,train_indices_set].to(self.device)
            Y_test = dataset.Y[test_indices_set][:,test_indices_set].to(self.device)

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
        for batch_X, batch_y, _, batch_idx in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            batch_preds = self(batch_X, batch_idx)
            train_predictions.append(batch_preds)
            train_targets.append(batch_y)
            
        test_predictions = []
        test_targets = []
        for batch_X, batch_y, _, batch_idx in test_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device) 
            batch_preds = self(batch_X, batch_idx)
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


class PLS_MLPDecoderModel(BaseModel):
    def __init__(self, input_dim, train_indices, test_indices, region_pair_dataset,
                 binarize=False, n_components=10, max_iter=1000, scale=True, optimize_encoder=False, hidden_dims=[128, 64],
                 dropout_rate=0.2, learning_rate=0.0001, weight_decay=0.0001, batch_size=512, epochs=100):
        super().__init__()
        
        self.binarize = binarize
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimize_encoder = optimize_encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize PLS encoder
        self.encoder = PLSEncoder(
            train_indices=train_indices,
            test_indices=test_indices,
            region_pair_dataset=region_pair_dataset,
            n_components=n_components,
            max_iter=max_iter,
            scale=scale,
            optimize_encoder=optimize_encoder, 
            device=self.device)
            
        self.criterion = nn.MSELoss() # nn.BCEWithLogitsLoss() if binarize else 

        # Initialize decoder
        prev_dim = n_components * 2  # Doubled because we concatenate two region encodings
        deep_layers = []
        for hidden_dim in hidden_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        self.deep_layers = nn.Sequential(*deep_layers)
        self.output_layer = nn.Linear(prev_dim, 1)
    
        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters())}")

        self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)            
        self.patience = 30
        self.scheduler = ReduceLROnPlateau( 
            self.optimizer, 
            mode='min', 
            factor=0.3,  # Reduce LR by 70%
            patience=20,  # Reduce LR after patience epochs of no improvement
            threshold=0.05,  # Threshold to detect stagnation
            cooldown=1,  # Reduce cooldown period
            min_lr=1e-6,  # Prevent LR from going too low
            verbose=True)
        
    def forward(self, x, idx):
        encoded_i, encoded_j = self.encoder(x, idx)
        concatenated_embedding = torch.cat((encoded_i, encoded_j), dim=1)
        deep_output = self.deep_layers(concatenated_embedding)
        output = self.output_layer(deep_output)
        return output.squeeze()
    
    # def predict(self, loader):
    #     self.eval()
    #     predictions = []
    #     targets = []
    #     with torch.no_grad():
    #         for batch_X, batch_y, _, batch_idx in loader:
    #             batch_X = batch_X.to(self.device)
    #             batch_preds = self(batch_X, batch_idx).cpu().numpy()
    #             predictions.append(batch_preds)
    #             targets.append(batch_y.numpy())
    #     predictions = np.concatenate(predictions)
    #     targets = np.concatenate(targets)
    #     return ((predictions > 0.5).astype(int) if self.binarize else predictions), targets
    
    # def fit(self, dataset, expanded_train_indices, expanded_test_indices, verbose=True):
    #     train_dataset = Subset(dataset, expanded_train_indices)
    #     test_dataset = Subset(dataset, expanded_test_indices)
    #     train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
    #     test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
    #     return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, self.scheduler, verbose=verbose)



class PLSGene2Conn(nn.Module):
    def __init__(self, train_indices, test_indices, X, Y, n_components=10, max_iter=1000, scale=True):
        super().__init__()
        
        self.n_components = n_components
        self.max_iter = max_iter
        self.scale = scale
        
        X_train = X[train_indices]
        Y_train = Y[train_indices][:, train_indices]

        self.pls_model = PLSRegression(n_components=n_components, max_iter=max_iter, scale=scale)
        self.pls_model.fit(X_train, Y_train)
    
    def forward(self, X):
        Y_train_pred = self.pls_model.predict(X)
        return Y_train_pred

class PLSConn2Conn(nn.Module):
    def __init__(self, train_indices, test_indices, Y, n_components=10, max_iter=1000, scale=True):
        super().__init__()
        
        self.n_components = n_components
        self.max_iter = max_iter
        self.scale = scale
        
        Y_train = Y[train_indices][:, train_indices]
        Y_intermediate = Y[train_indices][:, test_indices]

        self.pls_model = PLSRegression(n_components=n_components, max_iter=max_iter, scale=scale)
        self.pls_model.fit(Y_train, Y_intermediate)
    
    def forward(self, Y):
        Y_test_pred = self.pls_model.predict(Y)
        return Y_test_pred

class PLSTwoStepModel(nn.Module):
    def __init__(self, input_dim, train_indices, test_indices, region_pair_dataset, n_components_l=10, n_components_k=10, max_iter=1000, scale=True, binarize=False,):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.X = region_pair_dataset.X.numpy()
        self.Y = region_pair_dataset.Y.numpy()
        
        # fit PLS between X train and Y train
        self.pls_step1 = PLSGene2Conn(train_indices, test_indices, self.X, self.Y, n_components_l, max_iter, scale)
        # fit PLS between Y_train and Y_test
        self.pls_step2 = PLSConn2Conn(train_indices, test_indices, self.Y, n_components_k, max_iter, scale)

        self.criterion = nn.MSELoss()
    
    def forward(self, X):
        Y_train_pred = self.pls_step1(X)
        Y_test_pred = self.pls_step2(Y_train_pred)
        return Y_test_pred
    
    def predict(self, X, indices, train=False):
        if train:
            Y_pred = self.pls_step1(self.X[indices])
        else:
            Y_pred = self(X[indices])
        
        Y_true = self.Y[indices][:, indices]
        
        # Expand Y matrices symmetrically
        Y_pred_no_diag = expand_Y_symmetric(Y_pred)
        Y_true_no_diag = expand_Y_symmetric(Y_true)
        
        return Y_pred_no_diag, Y_true_no_diag
    
    def fit(self, dataset, train_indices, test_indices, verbose=True):        
        # Note: Model is already fit during initialization
        Y_train_pred, Y_train_true = self.predict(self.X, train_indices, train=True)
        Y_test_pred, Y_test_true = self.predict(self.X, test_indices)
        
        train_loss = self.criterion(torch.tensor(Y_train_pred), torch.tensor(Y_train_true)).item()
        test_loss = self.criterion(torch.tensor(Y_test_pred), torch.tensor(Y_test_true)).item()
    
        train_pearson = pearsonr(Y_train_pred, Y_train_true)[0]
        test_pearson = pearsonr(Y_test_pred, Y_test_true)[0]
        
        if verbose:
            print(f"Train Loss: {train_loss:.4f}, Train Pearson: {train_pearson:.4f}")
            print(f"Test Loss: {test_loss:.4f}, Test Pearson: {test_pearson:.4f}")
            
        train_history = {
            "train_loss": [train_loss],
            "val_loss": [test_loss],
            "train_pearson": [train_pearson],
            "val_pearson": [test_pearson]
        }
        return train_history