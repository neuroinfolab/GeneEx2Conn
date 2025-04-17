from env.imports import *
from models.train_val import train_model
from data.data_load import load_transcriptome, load_connectome
from data.data_utils import expand_X_symmetric, expand_Y_symmetric


class PLSEncoder(nn.Module):
    def __init__(self, train_indices, test_indices, X, Y, n_components=10, max_iter=1000, scale=True, optimize_encoder=False):
        super().__init__()
        
        self.n_components = n_components
        self.max_iter = max_iter
        self.scale = scale
        
        # Subset to train indices
        X_train = X[train_indices]
        Y_train = Y[train_indices][:, train_indices]
        print(f"X_train shape: {X_train.shape}")
        print(f"Y_train shape: {Y_train.shape}")

        # Fit PLS model
        self.pls_model = PLSRegression(n_components=n_components, max_iter=max_iter, scale=scale)
        self.pls_model.fit(X_train, Y_train)
        
        # Save projection matrices as torch parameters
        # x_weights_ for predictive importance, x_loadings_ for correlation-based interpretability, x_rotations_ for learned orthonormal projection
        if optimize_encoder == True:
            # Optimize on x_weights_ since more aligned with biological interpretation and prediction task
            self.x_projector = nn.Parameter(torch.FloatTensor(self.pls_model.x_weights_), requires_grad=True)
            print(f"x_projector shape: {self.x_projector.shape}")
            self.x_loadings = nn.Parameter(torch.FloatTensor(self.pls_model.x_loadings_), requires_grad=True)
            print(f"x_loadings shape: {self.x_loadings.shape}")
        else: 
            # This is fixed like what is done in the literature and how sklearn projects, https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73/sklearn/cross_decomposition/_pls.py#L564
            self.x_projector = nn.Parameter(torch.FloatTensor(self.pls_model.x_rotations_), requires_grad=False)
            print(f"x_projector shape: {self.x_projector.shape}")
            self.x_loadings = nn.Parameter(torch.FloatTensor(self.pls_model.x_loadings_), requires_grad=False)
            print(f"x_loadings shape: {self.x_loadings.shape}")
    
    def forward(self, x):
        x_scores = torch.matmul(x, self.x_projector)
        return x_scores

class PLSDecoderModel(nn.Module):
    def __init__(self, input_dim, train_indices, test_indices, region_pair_dataset, decoder='mlp',
                 binarize=False, n_components=10, max_iter=1000, scale=True, optimize_encoder=False, hidden_dims=[128, 64],
                 dropout_rate=0.2, learning_rate=0.0001, weight_decay=0.0001, batch_size=512, epochs=100):
        super().__init__()
        
        self.binarize = binarize
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.X = region_pair_dataset.X
        self.Y = region_pair_dataset.Y
        
        # Initialize PLS encoder
        self.encoder = PLSEncoder(
            train_indices=train_indices,
            test_indices=test_indices,
            X=self.X,
            Y=self.Y,
            n_components=n_components,
            max_iter=max_iter,
            scale=scale,
            optimize_encoder=optimize_encoder)
            
        self.criterion = nn.MSELoss() # nn.BCEWithLogitsLoss() if binarize else 

        # Initialize decoder
        if decoder == 'mlp':
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
        elif decoder == 'linear':
            self.linear = nn.Linear(n_components * 2, 1)
        elif decoder == 'bilinear':
            self.bilinear = nn.Bilinear(n_components, n_components, 1)

        self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)            
        self.patience = 20
        self.scheduler = ReduceLROnPlateau( 
            self.optimizer, 
            mode='min', 
            factor=0.3,  # Reduce LR by 70%
            patience=20,  # Reduce LR after patience epochs of no improvement
            threshold=0.05,  # Threshold to detect stagnation
            cooldown=1,  # Reduce cooldown period
            min_lr=1e-6,  # Prevent LR from going too low
            verbose=True)
        
    def forward(self, x):
        x_i, x_j = torch.chunk(x, chunks=2, dim=1)
        
        encoded_i = self.encoder(x_i)
        encoded_j = self.encoder(x_j)

        if hasattr(self, 'deep_layers'):
            concatenated_embedding = torch.cat((encoded_i, encoded_j), dim=1)
            deep_output = self.deep_layers(concatenated_embedding)
            output = self.output_layer(deep_output)
        elif hasattr(self, 'bilinear'):
            output = self.bilinear(encoded_i, encoded_j)
        elif hasattr(self, 'linear'):
            concatenated_embedding = torch.cat((encoded_i, encoded_j), dim=1)
            output = self.linear(concatenated_embedding)
        else:
            raise ValueError(f"Decoder {self.decoder} not supported")
        
        return output.squeeze()
    
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
    
    def fit(self, dataset, expanded_train_indices, expanded_test_indices, verbose=True):
        train_dataset = Subset(dataset, expanded_train_indices)
        test_dataset = Subset(dataset, expanded_test_indices)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, self.scheduler, verbose=verbose)


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