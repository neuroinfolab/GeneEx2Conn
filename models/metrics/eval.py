# Gene2Conn/models/metrics/eval.py

from imports import *
from data.data_utils import reconstruct_connectome
import models.metrics.distance_FC
from models.metrics.distance_FC import *


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

# Custom cupy scorers for GPU acceleration
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


class Metrics:
    def __init__(self, Y_true, Y_pred, square=False):
        # convert cupy arrays to numpy arrays if necessary
        self.Y_true = getattr(Y_true, 'get', lambda: Y_true)()
        self.Y_pred = getattr(Y_pred, 'get', lambda: Y_pred)()    
        self.Y_true_flat = self.Y_true.flatten()
        self.Y_pred_flat = self.Y_pred.flatten()

    
        self.square = square
        self.compute_metrics()

    def compute_metrics(self):
        # Compute standard metrics on flattened data
        self.mse = mean_squared_error(self.Y_true_flat, self.Y_pred_flat)
        self.mae = mean_absolute_error(self.Y_true_flat, self.Y_pred_flat)
        self.r2 = r2_score(self.Y_true_flat, self.Y_pred_flat)
        self.pearson_corr = pearsonr(self.Y_true_flat, self.Y_pred_flat)[0]

        # Compute geodesic distance if data is square (connectome)
        if self.square:
            Y_pred_connectome = reconstruct_connectome(self.Y_pred, symmetric=True)
            Y_pred_connectome_asymmetric = reconstruct_connectome(self.Y_pred, symmetric=False)
            Y_true_connectome = reconstruct_connectome(self.Y_true)
            self.geodesic_distance = distance_FC(Y_true_connectome, Y_pred_connectome).geodesic()

            # Visualize true and predicted connectomes
            plt.figure(figsize=(16, 4))
            
            plt.subplot(141)
            plt.imshow(Y_true_connectome, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(shrink=0.5)
            plt.title('True Connectome')
            
            plt.subplot(142) 
            plt.imshow(Y_pred_connectome_asymmetric, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(shrink=0.5)
            plt.title('Predicted Connectome (asymmetric)')

            plt.subplot(143) 
            plt.imshow(Y_pred_connectome, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(shrink=0.5)
            plt.title('Predicted Connectome (symmetric)')
            
            plt.subplot(144)
            plt.imshow(abs(Y_true_connectome - Y_pred_connectome), cmap='RdYlGn_r')
            plt.colorbar(shrink=0.5)
            plt.title('Prediction Difference')
            
            plt.tight_layout()
            plt.show()
    
    def get_metrics(self):
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
    def __init__(self, model, X_train, Y_train, X_test, Y_test, train_shared_regions, test_shared_regions):
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train 
        self.X_test = X_test
        self.Y_test = Y_test
        
        self.train_shared_regions = train_shared_regions
        self.test_shared_regions = test_shared_regions
        self.train_metrics = self.evaluate(self.X_train, self.Y_train, not self.train_shared_regions)
        self.test_metrics = self.evaluate(self.X_test, self.Y_test, not self.test_shared_regions)

    def evaluate(self, X, Y, square):
        Y_pred = self.model.predict(X) # this outputs a np array
        return Metrics(Y, Y_pred, square).get_metrics()

    def get_train_metrics(self):
        return self.train_metrics

    def get_test_metrics(self):
        return self.test_metrics