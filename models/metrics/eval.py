from env.imports import *
from data.data_utils import reconstruct_connectome
import models.metrics.distance_FC
from models.metrics.distance_FC import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from matplotlib.colors import ListedColormap
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from IPython import get_ipython


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

def in_jupyter_notebook():
    try:
        if 'IPKernelApp' in get_ipython().config:
            return True
    except:
        pass
    return False

class Metrics:
    def __init__(self, Y, indices, Y_true, Y_pred, square=False, binarize=False, network_labels=None, distances=None):
        self.Y = Y
        self.distances = distances
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
        
        if in_jupyter_notebook():
            plt.show()
            try:
                self.visualize_predictions_full()
                self.visualize_predictions_subset()
            except: 
                print('No full or subset visualizations for this model')
            try: 
                if not self.binarize:
                    self.visualize_predictions_scatter()
            except: 
                print('No scatter plot visualization for this model')
        else:
            plt.close()

    def get_distance_based_correlations(self, distances, y_true, y_pred):
        """
        Calculate correlations for different distance ranges.
        """
        # Calculate fixed distance cutoffs for short/mid/long range
        dist_min = 0
        dist_max = 175
        dist_range = dist_max - dist_min
        dist_33 = dist_min + (dist_range / 3)
        dist_67 = dist_min + (2 * dist_range / 3)
        
        # Create masks for different ranges
        short_mask = distances <= dist_33
        mid_mask = (distances > dist_33) & (distances <= dist_67)
        long_mask = distances > dist_67
        
        # Calculate correlations for each range
        overall_corr = pearsonr(y_true, y_pred)[0]
        short_range_corr = pearsonr(y_true[short_mask], y_pred[short_mask])[0]
        mid_range_corr = pearsonr(y_true[mid_mask], y_pred[mid_mask])[0]
        long_range_corr = pearsonr(y_true[long_mask], y_pred[long_mask])[0]
        
        return overall_corr, short_range_corr, mid_range_corr, long_range_corr
    
    def compute_metrics(self):
        if self.binarize: # Compute classification metrics
            self.accuracy = accuracy_score(self.Y_true_flat, self.Y_pred_flat)
            self.precision = precision_score(self.Y_true_flat, self.Y_pred_flat)
            self.recall = recall_score(self.Y_true_flat, self.Y_pred_flat)
            self.f1 = f1_score(self.Y_true_flat, self.Y_pred_flat)
            self.auc_roc = roc_auc_score(self.Y_true_flat, self.Y_pred_flat)
        else: # Compute regression metrics
            self.mse = mean_squared_error(self.Y_true_flat, self.Y_pred_flat)
            self.mae = mean_absolute_error(self.Y_true_flat, self.Y_pred_flat)
            self.r2 = r2_score(self.Y_true_flat, self.Y_pred_flat)
            self.pearson_corr = pearsonr(self.Y_true_flat, self.Y_pred_flat)[0]
            try: 
                self.overall_r, self.short_r, self.mid_r, self.long_r = self.get_distance_based_correlations(
                self.distances, 
                self.Y_true_flat, 
                self.Y_pred_flat)
            except: 
                print('No distance-based correlations for this model')
            
            if self.square: # Compute geodesic distance if test data is a square connectome
                Y_pred_connectome = reconstruct_connectome(self.Y_pred, symmetric=True)
                Y_pred_connectome_asymmetric = reconstruct_connectome(self.Y_pred, symmetric=False)
                Y_true_connectome = reconstruct_connectome(self.Y_true)
                self.geodesic_distance = distance_FC(Y_true_connectome, Y_pred_connectome).geodesic()
        
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
                'pearson_r': self.pearson_corr,
                'short_r': self.short_r,
                'mid_r': self.mid_r,
                'long_r': self.long_r
            
            }
        if self.square:
            metrics['geodesic_distance'] = self.geodesic_distance
        return metrics

    
    def visualize_predictions_scatter(self):
        # Set global visualization parameters
        TITLE_SIZE = 14
        LABEL_SIZE = 22
        LEGEND_SIZE = 20
        TEXT_SIZE = 20
        TICK_SIZE = 20
        
        plt.figure(figsize=(10, 7))        
        
        min_val = min(self.Y_true_flat.min(), self.Y_pred_flat.min())
        max_val = max(self.Y_true_flat.max(), self.Y_pred_flat.max())

        # Get correlations for different distance ranges using helper function
        overall_r, short_r, mid_r, long_r = self.get_distance_based_correlations(
            self.distances, 
            self.Y_true_flat, 
            self.Y_pred_flat
        )
        # Create distance-based colormap
        norm = plt.Normalize(self.distances.min(), self.distances.max())
        cmap = plt.cm.viridis  # Using viridis colormap for distance gradient
        
        # Create scatter plot with distance-based colors
        scatter = plt.scatter(self.Y_true_flat, self.Y_pred_flat, 
                            c=self.distances, cmap=cmap,
                            alpha=0.5, s=4)
        
        # Add line of best fit
        z = np.polyfit(self.Y_true_flat, self.Y_pred_flat, 1)
        p = np.poly1d(z)
        plt.plot(self.Y_true_flat, p(self.Y_true_flat), "r:", alpha=0.5)
        
        # Set equal axes ranges
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        
        # Add legend with correlations
        legend_text = f'Overall r = {overall_r:.3f}\nShort-range r = {short_r:.3f}\nMid-range r = {mid_r:.3f}\nLong-range r = {long_r:.3f}'
        plt.text(0.05, 0.95, legend_text, transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top',
                fontsize=LEGEND_SIZE)
        
        plt.xlabel('True Values', fontsize=LABEL_SIZE)
        plt.ylabel('Predicted Values', fontsize=LABEL_SIZE)
        #plt.title('Scatter Plot of True vs Predicted Values', fontsize=TITLE_SIZE)
        
        # Set tick label sizes
        plt.xticks(fontsize=TICK_SIZE)
        plt.yticks(fontsize=TICK_SIZE)
        
        # Add colorbar with consistent font sizes
        cbar = plt.colorbar(scatter)
        cbar.set_label('Distance (mm)', fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)  # Correct way to set colorbar tick label size
        
        plt.show()
        
    def visualize_predictions_subset(self):
        if self.square: # Compute geodesic distance if test data is a square connectome
            Y_pred_connectome = reconstruct_connectome(self.Y_pred, symmetric=True)
            Y_pred_connectome_asymmetric = reconstruct_connectome(self.Y_pred, symmetric=False)
            Y_true_connectome = reconstruct_connectome(self.Y_true)

            # Create high resolution figure
            plt.figure(figsize=(16, 4), dpi=300)
            
            plt.subplot(131)
            plt.imshow(Y_true_connectome, cmap='RdBu_r', vmin=-0.8, vmax=0.8, interpolation='nearest')
            plt.colorbar(shrink=0.5)
            plt.title('True Connectome', pad=10)
            
            plt.subplot(132) 
            plt.imshow(Y_pred_connectome_asymmetric, cmap='RdBu_r', vmin=-0.8, vmax=0.8, interpolation='nearest')
            plt.colorbar(shrink=0.5)
            plt.title('Predicted Connectome', pad=10) # (non-symmetrized)

            plt.subplot(133)
            plt.imshow(abs(Y_true_connectome - Y_pred_connectome_asymmetric), cmap='RdYlGn_r', interpolation='nearest')
            plt.colorbar(shrink=0.5)
            plt.title('Prediction Difference', pad=10)
            
            plt.tight_layout()
            plt.savefig('predictions_subset.png', dpi=300, bbox_inches='tight')
            plt.show()
        
    def visualize_predictions_full(self):
        n = int(self.Y.shape[0])  # Get dimensions of square connectome
        if self.binarize: 
            split_mask = np.zeros((n, n)) + 0.1
        else: 
            split_mask = np.zeros((n, n))
            
        # Calculate prediction differences for region pairs
        diff = abs(self.Y_true - self.Y_pred)
        
        # For each pair of regions in indices
        for idx, (i, j) in enumerate(combinations(self.indices, 2)): # Each pair appears twice in diff - once in each direction
            split_mask[i,j] = diff[2*idx]   # First direction: i->j is at 2*idx
            split_mask[j,i] = diff[2*idx + 1]   # Second direction: j->i is at 2*idx + 1
                
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

            plt.xticks(tick_positions, tick_labels, rotation=45, ha='right', fontsize=12)
            plt.yticks(tick_positions, tick_labels, fontsize=12)
        
        if self.binarize: 
            colors = [
                (0.0, "green"),  
                (0.1, "#2A0A4A"), 
                (1.0, "red"), 
            ]
            cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
        else: 
            colors = [
                (0.0, "#2A0A4A"),  # 0 -> dark purple
                (0.001, "green"),   # Near zero -> green
                (0.1, "green"),    # 0.1 -> still green
                (0.2, "yellow"),   # 0.2 -> yellow
                (1.0, "red"),      # 1.0 -> red
            ]
            cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
        
        # Plot prediction differences with network labels
        plt.subplot(122)
        plt.imshow(split_mask, cmap=cmap, interpolation='none', vmin=0, vmax=1)
        plt.colorbar(shrink=0.5)
        plt.title('Prediction Differences for Training Pairs', fontsize=14)
            
        plt.tight_layout()
        plt.show()

class ModelEvaluatorTorch:
    def __init__(self, region_pair_dataset, model, Y, train_loader, train_indices, train_indices_expanded, test_loader, test_indices, test_indices_expanded, network_labels, train_shared_regions, test_shared_regions):        
        self.region_pair_dataset = region_pair_dataset
        self.X = region_pair_dataset.X.numpy()
        self.train_distances_expanded = self.region_pair_dataset.distances_expanded[train_indices_expanded]
        self.test_distances_expanded = self.region_pair_dataset.distances_expanded[test_indices_expanded]    
        
        self.model = model
        self.train_loader = train_loader
        self.train_indices = train_indices
        self.test_loader = test_loader
        self.test_indices = test_indices
        self.network_labels = network_labels
        self.train_shared_regions = train_shared_regions
        self.test_shared_regions = test_shared_regions
        self.Y = Y

        self.binarize = len(np.unique(Y)) == 2
        
        self.train_metrics = self.evaluate(self.train_loader, self.train_indices, self.train_distances_expanded, not self.train_shared_regions, train=True)
        self.test_metrics = self.evaluate(self.test_loader, self.test_indices, self.test_distances_expanded, not self.test_shared_regions, train=False)

    def evaluate(self, loader, indices, distances, square, train=False):
        try: # for most deep learning models
            self.Y_pred, self.Y_true = self.model.predict(loader)
        except: # this is for PLS like models
            self.Y_pred, self.Y_true = self.model.predict(self.X, indices, train)

        return Metrics(self.Y, indices, self.Y_true, self.Y_pred, square, self.binarize, self.network_labels, distances).get_metrics()

    def get_train_metrics(self):
        return self.train_metrics

    def get_test_metrics(self):
        return self.test_metrics
    

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