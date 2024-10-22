# Gene2Conn/sim/sim.py

# imports
from imports import * 

# data load
from data.data_load import load_transcriptome, load_connectome, load_coords
import data.data_load
importlib.reload(data.data_load)

# data utils
from data.data_utils import (
    reconstruct_connectome,
    reconstruct_upper_triangle,
    make_symmetric,
    expand_X_symmetric,
    expand_Y_symmetric,
    expand_X_symmetric_shared,
    expand_X_Y_symmetric_conn_only,
    expand_shared_matrices,
    expand_X_symmetric_w_conn, 
    process_cv_splits, 
    process_cv_splits_conn_only_model, 
    expanded_inner_folds_combined_plus_indices,
    expanded_inner_folds_combined_plus_indices_connectome
)
import data.data_utils
importlib.reload(data.data_utils)

# cross-validation classes
from data.cv_split import (
    RandomCVSplit, 
    SchaeferCVSplit, 
    CommunityCVSplit, 
    SubnetworkCVSplit
)
import data.cv_split
importlib.reload(data.cv_split)

# prebuilt model classes
from models.prebuilt_models import ModelBuild
import models.prebuilt_models
importlib.reload(models.prebuilt_models)

# metric classes
from metrics.eval import (
    ModelEvaluator,
    pearson_numpy,
    pearson_cupy,
    mse_numpy,
    mse_cupy,
    r2_numpy,
    r2_cupy
)
import metrics.eval
importlib.reload(metrics.eval)

# sim utility functions
import sim.sim_utils
from sim.sim_utils import bayes_search_init, grid_search_init, random_search_init, drop_test_network, find_best_params
from sim.sim_utils import bytes2human, print_system_usage
importlib.reload(sim.sim_utils)

from skopt.plots import plot_objective, plot_histogram

class Simulation:
    def __init__(self, feature_type, cv_type, model_type, gpu_acceleration, predict_connectome_from_connectome, summary_measure=None, euclidean=False, structural=False, resolution=1.0,random_seed=42,
                 use_shared_regions=False, include_conn_feats=False, test_shared_regions=False):        
        """
        Initialization of simulation parameters
        """
        self.cv_type = cv_type
        self.model_type = model_type
        self.gpu_acceleration = gpu_acceleration
        self.feature_type = feature_type
        self.predict_connectome_from_connectome = predict_connectome_from_connectome
        self.summary_measure = summary_measure
        self.euclidean = euclidean
        self.structural = structural
        self.resolution = resolution
        self.random_seed=random_seed
        self.use_shared_regions = use_shared_regions
        self.include_conn_feats = include_conn_feats
        self.test_shared_regions = test_shared_regions
        self.results = []

    
    def load_data(self):
        """
        Load transcriptome and connectome data
        """
        # load everything but apply logic to expand data for different data types!!!
        # reformatted needs to go in this way first
        omit_subcortical=False
        self.X = load_transcriptome()
        self.Y = load_connectome(measure='FC')
        self.X_pca = load_transcriptome(run_PCA=True)
        self.Y_sc = load_connectome(measure='SC')
        self.coords = load_coords()
    
    def select_cv(self):
        """
        Select cross-validation strategy
        """
        if self.cv_type == 'random':
            self.cv_obj = RandomCVSplit(self.X, self.Y, num_splits=4, shuffled=True, use_random_state=True, random_seed=self.random_seed)
        elif self.cv_type == 'schaefer':
            self.cv_obj = SchaeferCVSplit()
        elif self.cv_type == 'community':
            self.cv_obj = CommunityCVSplit(self.X, self.Y, resolution=self.resolution, random_seed=self.random_seed)

    
    def expand_data(self):
        """
        Expand data based on feature type and prediction type
        """
        # create a dict to map inputted feature types to data array
        feature_dict = {'transcriptome': self.X, 
                        'transcriptomePCA': self.X_pca,
                        'functional':self.Y, 
                        'structural':self.Y_sc, 
                        'euclidean':self.coords}

        X = []
        for feature in self.feature_type:
            feature_X = feature_dict[feature]
            X.append(feature_X)

        X = np.hstack(X)
        self.X = X
        print('self X shape', self.X.shape)
        
        if 'transcriptomePCA' in self.feature_type:
            feature_X = feature_dict['transcriptomePCA']
            self.PC_dim = int(feature_X.shape[1])
        else:
            self.PC_dim = None # save dimensionality for kronecker for region-wise expansion
        
        if self.summary_measure == 'kronecker':
            kron = True
            print('PC dim', self.PC_dim )
         
        self.fold_splits = process_cv_splits(self.X, self.Y, self.cv_obj, 
                          self.use_shared_regions, 
                          self.include_conn_feats, 
                          self.test_shared_regions,
                          kron=(True if self.summary_measure == 'kronecker' else False),
                          kron_input_dim = self.PC_dim)
        
        '''
        if self.predict_connectome_from_connectome:
            self.fold_splits = process_cv_splits_conn_only_model(self.Y, self.Y,
                                                                 self.cv_obj,
                                                                 self.use_shared_regions,
                                                                 self.test_shared_regions)
        else:
            self.fold_splits = process_cv_splits(self.X, self.Y, self.cv_obj, 
                                                 self.use_shared_regions, 
                                                 self.include_conn_feats, 
                                                 self.test_shared_regions)
        '''
            
    
    def run_innercv(self, train_indices, test_indices, train_network_dict, search_method='grid', n_iter=100):
        """
        Inner cross-validation with option for Grid, Bayesian, or Randomized Search
        """
        
        # Create inner CV object (just indices) for X_train and Y_train
        inner_cv_obj = SubnetworkCVSplit(train_indices, train_network_dict)
    
        if self.predict_connectome_from_connectome:
            inner_fold_splits = process_cv_splits_conn_only_model(self.Y, self.Y, inner_cv_obj,
                                                                  self.use_shared_regions,
                                                                  self.test_shared_regions)
        else:
            inner_fold_splits = process_cv_splits(self.X, self.Y, inner_cv_obj, 
                                                  self.use_shared_regions, 
                                                  self.test_shared_regions,
                                                  kron=(True if self.summary_measure == 'kronecker' else False),
                                                  kron_input_dim = self.PC_dim
                                                 )
        '''
        if self.predict_connectome_from_connectome or self.include_conn_feats:
            grid_search_cv_results, grid_search_best_scores, grid_search_best_params = [], [], []
            
            for X_train, X_test, Y_train, Y_test in inner_fold_splits:
                # Create single fold object for inner CV  
                X_combined, Y_combined, train_test_indices = expanded_inner_folds_combined_plus_indices_connectome(X_train, X_test, Y_train, Y_test)
            
                # Initialize model
                model = ModelBuild.init_model(self.model_type)
                param_grid = model.get_param_grid()
                param_dist = model.get_param_dist()

                # Initialize grid search and return cupy converted array if necessary
                if search_method == 'grid':
                    grid_search, X_combined, Y_combined = grid_search_init(self.gpu_acceleration, model, X_combined, Y_combined, param_grid, train_test_indices)
                elif search_method == 'random':
                    grid_search, X_combined, Y_combined = random_search_init(self.gpu_acceleration, model, X_combined, Y_combined, param_dist, train_test_indices, n_iter)
                elif search_method == 'bayes':
                    grid_search, X_combined, Y_combined = bayes_search_init(self.gpu_acceleration, model, X_combined, Y_combined, param_dist, train_test_indices, n_iter)
                    
                # Fit GridSearchCV on the current fold
                grid_search.fit(X_combined, Y_combined)
                
                # Display comprehensive results
                print("\nGrid Search CV Results:")
                print("=======================")
                print("Best Cross-Validation Score: ", grid_search.best_score_)
                print("Best Parameters: ", grid_search.best_params_)

                _ = plot_objective(grid_search.optimizer_results_[0])
                plt.show()
                
                grid_search_cv_results.append(grid_search.cv_results_)
                grid_search_best_scores.append(grid_search.best_score_)
                grid_search_best_params.append(grid_search.best_params_)

            # this is done automatically in true gridsearch
            best_params = find_best_params(grid_search_cv_results) 
            
            model = ModelBuild.init_model(self.model_type)
            model = model.get_model()
            best_estimator = model.set_params(**best_params)
            
            return best_estimator
        else:
            
        '''
        X_combined, Y_combined, train_test_indices = expanded_inner_folds_combined_plus_indices(inner_fold_splits)
        
        # Initialize model
        model = ModelBuild.init_model(self.model_type, X_combined.shape[1])
        param_grid = model.get_param_grid()
        param_dist = model.get_param_dist()

        # Initialize grid search and return cupy converted array if necessary
        if search_method == 'grid':
            grid_search, X_combined, Y_combined = grid_search_init(self.gpu_acceleration, model, X_combined, Y_combined, param_grid, train_test_indices)
        elif search_method == 'random':
            grid_search, X_combined, Y_combined = random_search_init(self.gpu_acceleration, model, X_combined, Y_combined, param_dist, train_test_indices, n_iter=n_iter)
        elif search_method == 'bayes':
            grid_search, X_combined, Y_combined = bayes_search_init(self.gpu_acceleration, model, X_combined, Y_combined, param_dist, train_test_indices, n_iter=n_iter)

        
        # Fit GridSearchCV on the combined data
        grid_search.fit(X_combined, Y_combined)
        
        # Display comprehensive results
        print("\nGrid Search CV Results:")
        print("=======================")
        print("Best Parameters: ", grid_search.best_params_)
        print("Best Cross-Validation Score: ", grid_search.best_score_)

        _ = plot_objective(grid_search.optimizer_results_[0], size=5)
        plt.show()

        best_model = model.get_model()
        best_model.set_params(**grid_search.best_params_)
        return best_model


    def run_sim(self, search_method='random'):
        """
        Main simulation method
        """
        self.load_data()           # Step 1: Load data
        self.select_cv()           # Step 2: Select CV strategy
        self.expand_data()         # Step 3: Expand data based on CV strategy

        # Step 4: Outer CV on folds
        for i, (X_train, X_test, Y_train, Y_test) in enumerate(self.fold_splits):
            print('\n', f'Test fold num: {i+1}')
            print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
            
            train_indices = self.cv_obj.folds[i][0]
            test_indices = self.cv_obj.folds[i][1]
   
            network_dict = self.cv_obj.networks
            train_network_dict = drop_test_network(self.cv_type, network_dict, test_indices, i+1)

            # Step 5: Inner CV on training data
            # search_method options: random, grid, bayes
            best_model = self.run_innercv(train_indices, test_indices, train_network_dict, search_method=search_method, n_iter=100)

            # convert to cupy if necessary
            if self.gpu_acceleration:
                X_train = cp.array(X_train)
                Y_train = cp.array(Y_train)
                X_test = cp.array(X_test)
                Y_test = cp.array(Y_test)
            
            # Step 6: Retrain the best parameter model on training data and test on testing data
            best_model.fit(X_train, Y_train)
 
            evaluator = ModelEvaluator(best_model, X_train, Y_train, X_test, Y_test, [self.use_shared_regions, self.include_conn_feats, self.test_shared_regions])
            
            train_metrics = evaluator.get_train_metrics()
            test_metrics = evaluator.get_test_metrics()

            print("\nTrain Metrics:", train_metrics)
            print("Test Metrics:", test_metrics)
            print('BEST MODEL PARAMS', best_model.get_params())

            # Implement function to grab feature importances here - can do for ridge too
            if self.model_type == 'pls': 
                feature_importances_ = best_model.x_weights_[:, 0]  # Weights for the first component
            elif self.model_type == 'xgboost': 
                feature_importances_ = best_model.feature_importances_
            elif self.model_type == 'ridge': 
                feature_importances_ = best_model.coef_
            else: 
                feature_importances_ = None
            
            self.results.append({
                'model_parameters': best_model.get_params(),
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'y_true': Y_test.get() if self.gpu_acceleration else Y_test,
                'y_pred': best_model.predict(X_test) if self.gpu_acceleration else best_model.predict(X_test), 
                'feature_importances': feature_importances_
            })
            
            # Display CPU and RAM utilization 
            print_system_usage()
            
            # Display GPU utilization
            GPUtil.showUtilization()
            

        
