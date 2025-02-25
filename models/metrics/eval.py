from env.imports import *
from data.data_utils import reconstruct_connectome
import models.metrics.distance_FC
from models.metrics.distance_FC import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from matplotlib.colors import ListedColormap

# Evaluation funcs that preserve connectome properties 
def connectome_correlation(Y_pred, Y_ground_truth, include_diag=False, output=False):
    pred_corrs = []
    for i in range(Y_pred.shape[0]):
        if include_diag:
            pred_corrs.append(pearsonr(Y_pred[i, :], Y_ground_truth[i, :])[0])
        else:
            pred_no_diag = np.concatenate((Y_pred[i, :i], Y_pred[i, i + 1:]))
            ground_truth_no_diag = np.concatenate((Y_ground_truth[i, :i], Y_ground_truth[i, i + 1:]))
            pred_corrs.append(pearsonr(pred_no_diag, ground_truth_no_diag)[0])
    if output:
        print(pred_corrs)
        print(np.mean(pred_corrs))
    return np.mean(pred_corrs)

def connectome_r2(Y_pred, Y_ground_truth, include_diag=False, output=False):
    pred_r2s = []
    for i in range(Y_pred.shape[0]):
        if include_diag:
            pred_r2s.append(r2_score(Y_ground_truth[i, :], Y_pred[i, :]))
        else:
            pred_no_diag = np.concatenate((Y_pred[i, :i], Y_pred[i, i + 1:]))
            ground_truth_no_diag = np.concatenate((Y_ground_truth[i, :i], Y_ground_truth[i, i + 1:]))
            pred_r2s.append(r2_score(ground_truth_no_diag, pred_no_diag))
    if output:
        print(pred_r2s)
        print(np.mean(pred_r2s))
    return np.mean(pred_r2s)

# Custom numpy and cupy scorers for inner-CV
def pearson_numpy(y_true, y_pred):
    """Compute Pearson correlation coefficient between true and predicted values using numpy."""
    return pearsonr(y_true, y_pred)[0]

def mse_numpy(y_true, y_pred):
    """Compute mean squared error between true and predicted values using numpy."""
    return np.mean(np.square(y_true - y_pred))

def r2_numpy(y_true, y_pred):
    """Compute R-squared score between true and predicted values using numpy."""
    y_true_mean = np.mean(y_true)
    total_ss = np.sum(np.square(y_true - y_true_mean))
    residual_ss = np.sum(np.square(y_true - y_pred))
    return 1 - (residual_ss / total_ss)

def accuracy_numpy(y_true, y_pred):
    """Compute accuracy score between true and predicted values using numpy."""
    y_pred_labels = np.round(y_pred)  # Convert probabilities to binary labels (0 or 1)
    return accuracy_score(y_true, y_pred_labels)
def logloss_numpy(y_true, y_pred):
    """Compute log loss between true and predicted values using numpy."""
    return log_loss(y_true, y_pred)

def pearson_cupy(y_true, y_pred):
    """Compute Pearson correlation coefficient between true and predicted values using cupy."""
    y_pred = cp.asarray(y_pred)
    y_true = cp.asarray(y_true).ravel()
    y_pred = y_pred.ravel()
    corr_matrix = cp.corrcoef(y_true, y_pred)
    cp.cuda.Stream.null.synchronize()
    return corr_matrix[0, 1]

def mse_cupy(y_true, y_pred):
    """Compute mean squared error between true and predicted values using cupy."""
    y_pred = cp.asarray(y_pred)
    mse = cp.mean(cp.square(y_pred - y_true))
    cp.cuda.Stream.null.synchronize()
    return mse
    
def r2_cupy(y_true, y_pred):
    """Compute R-squared score between true and predicted values using cupy."""
    y_pred = cp.asarray(y_pred)
    y_true_mean = cp.mean(y_true)
    total_ss = cp.sum(cp.square(y_true - y_true_mean))
    residual_ss = cp.sum(cp.square(y_true - y_pred))
    return 1 - (residual_ss / total_ss)

def accuracy_cupy(y_true, y_pred):
    """Compute accuracy score between true and predicted values using cupy."""
    y_pred = cp.asarray(y_pred)
    y_true = cp.asarray(y_true)

    y_pred_labels = cp.round(y_pred)  # Convert probabilities to binary labels (0 or 1)
    accuracy = cp.mean(y_pred_labels == y_true)
    
    cp.cuda.Stream.null.synchronize()
    return accuracy.item()

def logloss_cupy(y_true, y_pred):
    """Compute log loss between true and predicted values using cupy."""
    # Convert inputs to cupy arrays
    y_pred = cp.asarray(y_pred)
    y_true = cp.asarray(y_true)
    
    # Clip predictions to avoid log(0) 
    eps = 1e-15
    y_pred = cp.clip(y_pred, eps, 1 - eps)
    
    # Calculate log loss
    losses = -(y_true * cp.log(y_pred) + (1 - y_true) * cp.log(1 - y_pred))
    loss = cp.mean(losses)
    
    cp.cuda.Stream.null.synchronize()
    return loss.item()


class Metrics:
    def __init__(self, Y, indices, Y_true, Y_pred, square=False, binarize=False, network_labels=None):
        self.Y = Y
        self.indices = indices
        self.network_labels = network_labels

        # convert cupy arrays to numpy arrays if necessary
        self.Y_true = getattr(Y_true, 'get', lambda: Y_true)()
        self.Y_pred = getattr(Y_pred, 'get', lambda: Y_pred)()

        self.Y_true_flat = self.Y_true.flatten()
        self.Y_pred_flat = self.Y_pred.flatten()
        self.square = square
        self.binarize = binarize

        self.compute_metrics()
        self.visualize_predictions()

    def compute_metrics(self):
        if self.binarize:    
            # Compute classification metrics
            self.accuracy = accuracy_score(self.Y_true_flat, self.Y_pred_flat)
            self.precision = precision_score(self.Y_true_flat, self.Y_pred_flat)
            self.recall = recall_score(self.Y_true_flat, self.Y_pred_flat)
            self.f1 = f1_score(self.Y_true_flat, self.Y_pred_flat)
            self.auc_roc = roc_auc_score(self.Y_true_flat, self.Y_pred_flat)
        else:
            # Compute regression metrics
            self.mse = mean_squared_error(self.Y_true_flat, self.Y_pred_flat)
            self.mae = mean_absolute_error(self.Y_true_flat, self.Y_pred_flat)
            self.r2 = r2_score(self.Y_true_flat, self.Y_pred_flat)
            self.pearson_corr = pearsonr(self.Y_true_flat, self.Y_pred_flat)[0]

    def visualize_predictions(self):
        if self.square: # Compute geodesic distance if test data is a square connectome
            Y_pred_connectome = reconstruct_connectome(self.Y_pred, symmetric=True)
            Y_pred_connectome_asymmetric = reconstruct_connectome(self.Y_pred, symmetric=False)
            Y_true_connectome = reconstruct_connectome(self.Y_true)
            self.geodesic_distance = distance_FC(Y_true_connectome, Y_pred_connectome).geodesic()

            # Visualize true and predicted connectomes
            plt.figure(figsize=(16, 4))

            plt.subplot(131)
            plt.imshow(Y_true_connectome, cmap='viridis', vmin=Y_true_connectome.min(), vmax=Y_true_connectome.max())
            plt.colorbar(shrink=0.5)
            plt.title('True Connectome')
            
            plt.subplot(132) 
            plt.imshow(Y_pred_connectome_asymmetric, cmap='viridis', vmin=Y_true_connectome.min(), vmax=Y_true_connectome.max())
            plt.colorbar(shrink=0.5)
            plt.title('Predicted Connectome (non-symmetrized)')

            plt.subplot(133)
            plt.imshow(abs(Y_true_connectome - Y_pred_connectome_asymmetric), cmap='RdYlGn_r')
            plt.colorbar(shrink=0.5)
            plt.title('Prediction Difference')
            
            plt.tight_layout()
            plt.show()
    
        # Create a mask showing prediction differences for training regions
        n = int(self.Y.shape[0])  # Get dimensions of square connectome
        split_mask = np.zeros((n, n))
        
        # Calculate prediction differences for region pairs
        diff = abs(self.Y_true - self.Y_pred)
        
        # For each pair of regions in indices
        for idx, (i, j) in enumerate(combinations(self.indices, 2)):
            # Each pair appears twice in diff - once in each direction
            # First direction: i->j is at 2*idx
            # Second direction: j->i is at 2*idx + 1
            split_mask[i,j] = diff[2*idx]
            split_mask[j,i] = diff[2*idx + 1]
                
        plt.figure(figsize=(16, 6))
        
        # Plot full connectome with network labels
        plt.subplot(121)
        plt.imshow(self.Y, cmap='viridis')
        plt.colorbar(shrink=0.5)
        plt.title('Full Connectome', fontsize=14)
        
        if self.network_labels is not None:
            # Create tick positions and labels
            tick_positions = []
            tick_labels = []
            start_idx = 0
            prev_label = self.network_labels[0]
            
            for i in range(1, len(self.network_labels)):
                if self.network_labels[i] != prev_label:
                    tick_positions.append((start_idx + i - 1) / 2)
                    tick_labels.append(prev_label)
                    start_idx = i
                    prev_label = self.network_labels[i]
            
            # Add the last group
            tick_positions.append((start_idx + len(self.network_labels) - 1) / 2)
            tick_labels.append(prev_label)

            plt.xticks(tick_positions, tick_labels, rotation=45, ha='right', fontsize=8)
            plt.yticks(tick_positions, tick_labels, fontsize=8)
        
        # Plot prediction differences with network labels
        plt.subplot(122)
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, 256))
        colors[0] = [0.1, 0, 0.2, 1]  # Set zero values to dark purple
        custom_cmap = ListedColormap(colors)
        plt.imshow(split_mask, cmap=custom_cmap, interpolation='none')
        plt.colorbar(shrink=0.5)
        plt.title('Prediction Differences for Training Pairs', fontsize=14)
        
        if self.network_labels is not None:
            plt.xticks(tick_positions, tick_labels, rotation=45, ha='right', fontsize=8)
            plt.yticks(tick_positions, tick_labels, fontsize=8)
            
        plt.tight_layout()
        plt.show()

        
        
    def get_metrics(self):
        if self.binarize:
            metrics = {
                'accuracy': self.accuracy,
                'precision': self.precision,
                'recall': self.recall,
                'f1': self.f1,
                'auc_roc': self.auc_roc
            }
        else:
            metrics = {
                'mse': self.mse,
                'mae': self.mae,
                'r2': self.r2,
                'pearson_corr': self.pearson_corr
            }

        if self.square:
            metrics['geodesic_distance'] = self.geodesic_distance

        return metrics


class ModelEvaluator:
    def __init__(self, model, Y, train_indices, test_indices, network_labels, X_train, Y_train, X_test, Y_test, train_shared_regions, test_shared_regions):        
        self.Y = Y
        self.train_indices = train_indices
        self.test_indices = test_indices

        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train 
        self.X_test = X_test
        self.Y_test = Y_test
        self.network_labels = network_labels

        self.train_shared_regions = train_shared_regions
        self.test_shared_regions = test_shared_regions

        self.binarize = len(np.unique(Y_train)) == 2 and len(np.unique(Y_test)) == 2
        
        self.train_metrics = self.evaluate(X_train, Y_train, train_indices, not self.train_shared_regions)
        self.test_metrics = self.evaluate(X_test, Y_test, test_indices, not self.test_shared_regions)

    def evaluate(self, X, Y, indices, square):
        self.Y_pred = self.model.predict(X)
        return Metrics(self.Y, indices, Y, self.Y_pred, square, self.binarize, self.network_labels).get_metrics()

    def get_train_metrics(self):
        return self.train_metrics

    def get_test_metrics(self):
        return self.test_metrics