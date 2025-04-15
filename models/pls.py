from env.imports import *
from models.train_val import train_model
from data.data_load import load_transcriptome, load_connectome
from models.bilinear import BilinearLoss


class PLSEncoder(nn.Module):
    def __init__(self, train_indices, n_components=10, max_iter=1000, scale=True, optimize_encoder=False):
        super(PLSEncoder, self).__init__()
        
        self.n_components = n_components
        self.max_iter = max_iter
        self.scale = scale
        
        # Load full data
        parcellation, omit_subcortical, hemisphere = 'S400', True, 'both'
        X_full = load_transcriptome(parcellation=parcellation, omit_subcortical=omit_subcortical, hemisphere=hemisphere)
        Y_full = load_connectome(parcellation=parcellation, omit_subcortical=omit_subcortical, measure='FC', hemisphere=hemisphere)
        
        # Remove any NaN rows
        valid_indices = ~np.isnan(X_full).all(axis=1)
        X_full = X_full[valid_indices]
        Y_full = Y_full[valid_indices][:, valid_indices]
        print(f"X_full shape: {X_full.shape}")
        print(f"Y_full shape: {Y_full.shape}")

        # Subset to train indices
        X_train = X_full[train_indices]
        Y_train = Y_full[train_indices][:, train_indices]
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


class PLSModel(nn.Module):
    def __init__(self, input_dim, encoder_indices,
                 decoder='mlp',binarize=False, n_components=10, max_iter=1000, scale=True, optimize_encoder=False, hidden_dims=[128, 64],
                 dropout_rate=0.2, learning_rate=0.0001, weight_decay=0.0001, batch_size=512, epochs=100):
        super().__init__()
        
        self.binarize = binarize
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize PLS encoder
        self.encoder = PLSEncoder(
            train_indices=encoder_indices,
            n_components=n_components,
            max_iter=max_iter,
            scale=scale,
            optimize_encoder=optimize_encoder)
            
        self.criterion = nn.BCEWithLogitsLoss() if binarize else nn.MSELoss()
        self.optimizer = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = None
        self.patience = 20

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
            self.patience = 20
            self.scheduler = ReduceLROnPlateau( 
                self.optimizer, 
                mode='min', 
                factor=0.3,  # Reduce LR by 70%
                patience=20,  # Reduce LR after patience epochs of no improvement
                threshold=0.1,  # Threshold to detect stagnation
                cooldown=1,  # Reduce cooldown period
                min_lr=1e-6,  # Prevent LR from going too low
                verbose=True)
        elif decoder == 'linear':
            self.linear = nn.Linear(n_components * 2, 1)
        elif decoder == 'bilinear':
            self.bilinear = nn.Bilinear(n_components, n_components, 1)            

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
    
    def fit(self, dataset, train_indices, test_indices, verbose=True):
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return train_model(self, train_loader, test_loader, self.epochs, self.criterion, self.optimizer, self.patience, self.scheduler, verbose=verbose)
