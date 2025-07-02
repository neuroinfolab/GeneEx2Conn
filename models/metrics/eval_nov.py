from env.imports import *
from data.data_utils import reconstruct_connectome
import models.metrics.distance_FC
from models.metrics.distance_FC import *
from matplotlib.colors import ListedColormap
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from IPython import get_ipython

# Import all evaluation utilities
from models.metrics.eval_utils import (
    PlotConfig,
    in_jupyter_notebook,
    # Custom numpy scorers
    pearson_numpy,
    mse_numpy,
    r2_numpy,
    accuracy_numpy,
    logloss_numpy,
    # Custom cupy scorers
    pearson_cupy,
    mse_cupy,
    r2_cupy,
    accuracy_cupy,
    logloss_cupy,
    # Modern metric functions
    compute_basic_metrics,
    compute_distance_metrics,
    compute_hemispheric_metrics,
    compute_subnetwork_metrics,
    # Plotting functions
    plot_distance_scatter,
    plot_hemispheric_scatter,
    plot_subnetwork_scatter,
    plot_connectome_predictions_subset,
    plot_connectome_predictions_full,
    # Formatting functions
    format_metrics_output
)


class ModelEvaluator:
    """
    Unified model evaluator for torch-based models with comprehensive evaluation capabilities
    """
    def __init__(self, region_pair_dataset=None, model=None, Y=None,
                 train_loader=None, train_indices=None, train_indices_expanded=None,
                 test_loader=None, test_indices=None, test_indices_expanded=None,
                 network_labels=None, train_shared_regions=None, test_shared_regions=None,
                 plot_mode='basic'):
        """        
        Args:
            region_pair_dataset: Dataset for region pairs
            model: Torch model to evaluate
            Y: Full target connectome matrix
            train_loader/test_loader: Data loaders
            train_indices/test_indices: Region indices for train/test splits
            train_indices_expanded/test_indices_expanded: Expanded indices for region pairs
            network_labels: Network labels for each region
            train_shared_regions/test_shared_regions: Whether using shared regions
            plot_mode: 'metrics' (no plots), 'basic' (key plots), 'verbose' (all plots)
        """
        # Full dataset components
        self.region_pair_dataset = region_pair_dataset
        self.X = region_pair_dataset.X.numpy()
        self.Y = Y
        self.coords = region_pair_dataset.coords.numpy()
        self.network_labels = network_labels
        
        # Model, data loaders, indices
        self.model = model
        self.train_loader = train_loader
        self.train_indices = train_indices
        self.train_indices_expanded = train_indices_expanded 
        self.test_loader = test_loader
        self.test_indices = test_indices
        self.test_indices_expanded = test_indices_expanded

        # Config parameters 
        self.binarize = len(np.unique(Y)) == 2 if Y is not None else False
        self.train_shared_regions = train_shared_regions
        self.test_shared_regions = test_shared_regions
        self.config = PlotConfig(plot_mode=plot_mode)

        # Set up distance data if available
        self.train_distances = region_pair_dataset.distances_expanded[train_indices_expanded]
        self.test_distances = region_pair_dataset.distances_expanded[test_indices_expanded]
        
        # Run evaluation
        print("Running train evaluation...")
        self.train_metrics = self._evaluate_split('train')
        print("Running test evaluation...")
        self.test_metrics = self._evaluate_split('test')
        
    def evaluate(self, y_true, y_pred, indices, distances=None, mode='test', square=False):
        """
        Core evaluation method for computing metrics and generating plots
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            indices: Region indices for this evaluation subset
            distances: Distance matrix for distance-based metrics
            mode: 'train' or 'test' for proper labeling
            square: Whether connectome is square (for geodesic metrics)
        """
        print(f"\n=== {mode.upper()} EVALUATION ===")
        self.config.mode_title = f'{mode.capitalize()} Set'

        # Convert cupy arrays to numpy if necessary
        y_true = getattr(y_true, 'get', lambda: y_true)()
        y_pred = getattr(y_pred, 'get', lambda: y_pred)()
        
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        print(f"Evaluating {len(indices)} regions, {len(y_true_flat)} connections")
        
        # Data structure for all metrics
        metrics = {}
        metrics.update(compute_basic_metrics(y_true_flat, y_pred_flat, self.binarize))
        
        # Distance-based metrics
        if distances is not None and not self.binarize:
            try:
                distance_metrics = compute_distance_metrics(y_true_flat, y_pred_flat, distances)
                metrics.update(distance_metrics)
            except Exception as e:
                print(f"Warning: Could not compute distance metrics: {e}")
        
        # Hemispheric metrics
        if not self.binarize:
            try:
                hemispheric_metrics = compute_hemispheric_metrics(y_true_flat, y_pred_flat, indices, self.coords)
                metrics.update(hemispheric_metrics)
            except Exception as e:
                print(f"Warning: Could not compute hemispheric metrics: {e}")
        
        # Functional subnetwork metrics
        if self.network_labels is not None and not self.binarize:
            try:
                network_metrics = compute_subnetwork_metrics(y_true_flat, y_pred_flat, indices, self.network_labels)
                metrics.update(network_metrics)
            except Exception as e:
                print(f"Warning: Could not compute network metrics: {e}")
        
        # Geodesic metrics for square connectomes
        if square and not self.binarize:
            try:
                Y_pred_connectome = reconstruct_connectome(y_pred, symmetric=True)
                Y_true_connectome = reconstruct_connectome(y_true)
                geodesic_distance = distance_FC(Y_true_connectome, Y_pred_connectome).geodesic()
                metrics['geodesic_distance'] = geodesic_distance
            except Exception as e:
                print(f"Warning: Could not compute geodesic distance: {e}")
        
        return metrics
    
    def _evaluate_split(self, mode):
        """Evaluate either train or test split using torch model"""
        if mode == 'train':
            loader = self.train_loader
            indices = self.train_indices
            distances = self.train_distances
            square = not self.train_shared_regions if self.train_shared_regions is not None else False
        else:  # test
            loader = self.test_loader
            indices = self.test_indices  
            distances = self.test_distances
            square = not self.test_shared_regions if self.test_shared_regions is not None else False
            
        # Get predictions
        try:
            Y_pred, Y_true = self.model.predict(loader)
        except: # Fallback for PLS-like models
            Y_pred, Y_true = self.model.predict(self.X, indices, mode == 'train')
        
        # Compute metrics
        metrics = self.evaluate(Y_true, Y_pred, indices, distances, mode=mode, square=square)
        
        # Generate plots
        self.plot(Y_true, Y_pred, indices, distances, mode=mode, square=square)
        
        return metrics

    def plot(self, y_true, y_pred, indices, distances=None, mode='test', square=False):
        """
        Generate plots based on plot_mode configuration
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            indices: Region indices for this evaluation subset
            distances: Distance matrix for distance-based metrics
            mode: 'train' or 'test' for proper labeling
            square: Whether connectome is square (for geodesic metrics)
        """
        if not self.config.show_plots or self.config.plot_mode == 'metrics':
            return
            
        self.config.mode_title = f'{mode.capitalize()} Set'
        
        # Convert cupy arrays to numpy if necessary
        y_true = getattr(y_true, 'get', lambda: y_true)()
        y_pred = getattr(y_pred, 'get', lambda: y_pred)()
        
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        try:
            if self.config.plot_mode == 'basic':
                # Basic plots: connectome predictions and distance scatter
                if self.Y is not None:
                    plot_connectome_predictions_full(self.Y, y_true, y_pred, indices, 
                                                   self.network_labels, self.binarize, self.config)
                
                if square:
                    plot_connectome_predictions_subset(y_true, y_pred, self.config)
                
                if distances is not None and not self.binarize:
                    plot_distance_scatter(y_true_flat, y_pred_flat, distances, mode, self.config)
                    
            elif self.config.plot_mode == 'verbose':
                # All possible plots
                if self.Y is not None:
                    plot_connectome_predictions_full(self.Y, y_true, y_pred, indices, 
                                                   self.network_labels, self.binarize, self.config)
                
                if square:
                    plot_connectome_predictions_subset(y_true, y_pred, self.config)
                
                if not self.binarize:
                    if distances is not None:
                        plot_distance_scatter(y_true_flat, y_pred_flat, distances, mode, self.config)
                    
                    plot_hemispheric_scatter(y_true_flat, y_pred_flat, indices, self.coords, mode, self.config)
                    
                    if self.network_labels is not None:
                        plot_subnetwork_scatter(y_true_flat, y_pred_flat, indices, self.network_labels, mode, self.config)
                        
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")

    def plot_manual(self, mode='test'):
        """Convenience method for manual plotting calls"""
        # Get the appropriate data based on mode
        if mode == 'train':
            loader = self.train_loader
            indices = self.train_indices
            distances = self.train_distances
            square = not self.train_shared_regions if self.train_shared_regions is not None else False
        else:  # test
            loader = self.test_loader
            indices = self.test_indices  
            distances = self.test_distances
            square = not self.test_shared_regions if self.test_shared_regions is not None else False
            
        # Get predictions
        try:
            Y_pred, Y_true = self.model.predict(loader)
        except: # Fallback for PLS-like models
            Y_pred, Y_true = self.model.predict(self.X, indices, mode == 'train')
        
        self.plot(Y_true, Y_pred, indices, distances, mode=mode, square=square)

    def get_train_metrics(self):
        """Get training metrics"""
        if self.train_metrics is None:
            raise ValueError("No train metrics available. Either run full evaluation or call evaluate() directly.")
        format_metrics_output(self.train_metrics, 'train')
        return self.train_metrics

    def get_test_metrics(self):
        """Get test metrics"""
        if self.test_metrics is None:
            raise ValueError("No test metrics available. Either run full evaluation or call evaluate() directly.")
        format_metrics_output(self.test_metrics, 'test')
        return self.test_metrics

# Backwards compatibility alias
ModelEvaluatorTorch = ModelEvaluator