# Gene2Conn/metrics/eval.py

from imports import *
from data.data_utils import reconstruct_connectome
import metrics.distance_FC
from metrics.distance_FC import *

# connectome evaluation funcs
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


class Metrics:
    def __init__(self, Y_true, Y_pred):
        self.Y_true = Y_true
        self.Y_pred = Y_pred
        self.compute_metrics()

    def compute_metrics(self):
        Y_true = self.Y_true
        Y_pred = self.Y_pred
        
        try:
            Y_true = Y_true.get()
            Y_pred = Y_pred.get()
        except:
            pass
            
        self.mse = mean_squared_error(Y_true, Y_pred)
        self.mae = mean_absolute_error(Y_true, Y_pred)
        self.r2 = r2_score(Y_true, Y_pred)
        self.pearson_corr = pearsonr(Y_true.flatten(), Y_pred.flatten())[0]
            

    def get_metrics(self):
        return {
            'mse': self.mse,
            'mae': self.mae,
            'r2': self.r2,
            'pearson_corr': self.pearson_corr
        }

class ConnectomeMetrics(Metrics):
    def __init__(self, Y_true, Y_pred, include_diag=False):
        super().__init__(Y_true, Y_pred)
        self.include_diag = include_diag
        self.compute_connectome_metrics()

    def compute_connectome_metrics(self):
        self.connectome_corr = connectome_correlation(self.Y_pred, self.Y_true, self.include_diag)
        self.connectome_r2 = connectome_r2(self.Y_pred, self.Y_true, self.include_diag)
        self.geodesic_distance = distance_FC(self.Y_true, self.Y_pred).geodesic()

    def get_metrics(self):
        base_metrics = super().get_metrics()
        connectome_metrics = {
            'connectome_corr': self.connectome_corr,
            'connectome_r2': self.connectome_r2,
            'geodesic_distance': self.geodesic_distance
        }
        return {**base_metrics, **connectome_metrics}

class ModelEvaluator:
    def __init__(self, model, X_train, Y_train, X_test, Y_test, split_params):
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.use_shared_regions, _, self.test_shared_regions = split_params
        self.train_metrics = self.evaluate(self.X_train, self.Y_train, self.use_shared_regions)
        self.test_metrics = self.evaluate(self.X_test, self.Y_test, self.test_shared_regions)

    def evaluate(self, X, Y, non_square):
        Y_pred = self.model.predict(X)
        if non_square == False:
            Y_true_connectome = reconstruct_connectome(Y)
            Y_pred_connectome = reconstruct_connectome(Y_pred)
            return ConnectomeMetrics(Y_true_connectome, Y_pred_connectome).get_metrics()
        else:
            return Metrics(Y, Y_pred).get_metrics()

    def get_train_metrics(self):
        return self.train_metrics

    def get_test_metrics(self):
        return self.test_metrics


# Custom scorers for inner-CV 
def pearson_numpy(y_true, y_pred):
    corr, _ = pearsonr(y_true, y_pred)
    return corr

def mse_numpy(y_true, y_pred):
    # Compute the squared differences
    squared_diff = np.square(np.subtract(y_true, y_pred))
    
    # Compute the mean of the squared differences
    mse = np.mean(squared_diff)
    return mse

def r2_numpy(y_true, y_pred):
    # Compute the mean of the true values
    y_true_mean = np.mean(y_true)
    
    # Total sum of squares
    total_sum_of_squares = np.sum(np.square(np.subtract(y_true, y_true_mean)))
    
    # Residual sum of squares
    residual_sum_of_squares = np.sum(np.square(np.subtract(y_true, y_pred)))
    
    # Compute R² using true division
    r2 = 1 - np.true_divide(residual_sum_of_squares, total_sum_of_squares)
    return r2

# Can create a bunch of custom scorers for cupy 
def pearson_cupy(y_true, y_pred):
    # Compute the correlation matrix
    corr_matrix = cp.corrcoef(y_true, y_pred)
    
    # Extract the Pearson correlation coefficient (off-diagonal element)
    corr = corr_matrix[0, 1]
    return corr

def mse_cupy(y_true, y_pred): # try treating as np? 
    print(type(y_pred))
    # Compute the squared differences
    squared_diff = cp.square(cp.subtract(y_true, y_pred))
    
    # Compute the mean of the squared differences
    mse = cp.mean(squared_diff)
    return mse
    
def r2_cupy(y_true, y_pred):
    # Compute the mean of the true values
    y_true_mean = cp.mean(y_true)
    
    # Total sum of squares
    total_sum_of_squares = cp.sum(cp.square(cp.subtract(y_true, y_true_mean)))
    
    # Residual sum of squares
    residual_sum_of_squares = cp.sum(cp.square(cp.subtract(y_true, y_pred)))
    
    # Compute R² using true division
    r2 = 1 - cp.divide(residual_sum_of_squares, total_sum_of_squares)
    return r2
