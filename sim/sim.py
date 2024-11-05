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
from models.base_models import ModelBuild
import models.base_models
importlib.reload(models.base_models)

# custom models
from models.dynamic_nn import DynamicNN

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
                 use_shared_regions=False, include_conn_feats=False, test_shared_regions=False, connectome_target='FC'):        
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
        self.connectome_target = connectome_target.upper()
        self.results = []

    
    def load_data(self):
        """
        Load transcriptome and connectome data
        """
        # load everything but apply logic to expand data for different data types!!!
        # reformatted needs to go in this way first
        omit_subcortical=False
        self.X = load_transcriptome()
        self.Y_fc = load_connectome(measure='FC')
        self.Y_sc = load_connectome(measure='SC')
        self.Y = self.Y_fc if self.connectome_target == 'FC' else self.Y_sc
        self.X_pca = load_transcriptome(run_PCA=True)
        self.coords = load_coords()
    
    def select_cv(self):
        """
        Select cross-validation strategy
        """
        if self.cv_type == 'random':
            self.cv_obj = RandomCVSplit(self.X, self.Y, num_splits=4, shuffled=True, use_random_state=True, random_seed=self.random_seed)
        elif self.cv_type == 'schaefer':
            self.cv_obj = SchaeferCVSplit()
        elif self.cv_type == 'community': # for comparability to SC as target the splits should be based on the functional connectome
            self.cv_obj = CommunityCVSplit(self.X, self.Y_fc, resolution=self.resolution, random_seed=self.random_seed) 

    
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
        
        if 'transcriptomePCA' in self.feature_type:
            feature_X = feature_dict['transcriptomePCA']
            self.PC_dim = int(feature_X.shape[1])
        else:
            self.PC_dim = None # save dimensionality for kronecker for region-wise expansion
        
        if self.summary_measure == 'kronecker':
            kron = True

        self.fold_splits = process_cv_splits(self.X, self.Y, self.cv_obj, 
                          self.use_shared_regions, 
                          self.include_conn_feats, 
                          self.test_shared_regions,
                          struct_summ=(True if self.summary_measure == 'strength_and_corr' and 'structural' in self.feature_type else False),
                          kron=(True if self.summary_measure == 'kronecker' else False),
                          kron_input_dim = self.PC_dim)

                        
    def run_innercv_wandb(self, train_indices, test_indices, train_network_dict, search_method=('wandb', 'mse'), n_iter=100):
        """Inner cross-validation with W&B support for neural networks"""
        # Create inner CV object for X_train and Y_train
        inner_cv_obj = SubnetworkCVSplit(train_indices, train_network_dict)
        
        # Get all inner fold splits
        if self.predict_connectome_from_connectome:
            inner_fold_splits = process_cv_splits_conn_only_model(
                self.Y, self.Y, 
                inner_cv_obj,
                self.use_shared_regions,
                self.test_shared_regions
            )
        else:
            inner_fold_splits = process_cv_splits(
                self.X, self.Y, 
                inner_cv_obj,
                self.use_shared_regions,
                self.test_shared_regions,
                struct_summ=(True if self.summary_measure == 'strength_and_corr' and 'structural' in self.feature_type else False),
                kron=(True if self.summary_measure == 'kronecker' else False),
                kron_input_dim=self.PC_dim
            )
        
        print('inner fold splits', len(inner_fold_splits))
        print('X train shape', inner_fold_splits[0][0].shape)
        print('X test shape', inner_fold_splits[0][1].shape)
        print('Y train shape', inner_fold_splits[0][2].shape)
        print('Y test shape', inner_fold_splits[0][3].shape)    
        input_dim = inner_fold_splits[0][0].shape[1]
        
        sweep_config = {
            'method': 'random',
            'metric': {
                'name': 'val_loss',
                'goal': 'minimize'
            },
            'parameters': {
                'hidden_dims': {
                    'values': [
                        [64, 32],
                        [128, 64],
                        [256, 128, 64]
                    ]
                },
                'learning_rate': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-4,
                    'max': 1e-2
                },
                'batch_size': {
                    'distribution': 'q_log_uniform_values',
                    'q': 8,
                    'min': 32,
                    'max': 256
                },
                'dropout_rate': {
                    'distribution': 'uniform',
                    'min': 0.1,
                    'max': 0.5
                },
                'weight_decay': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-5,
                    'max': 1e-3
                },
                'input_dim': {'value': input_dim},
                'epochs': {'value': 10} # 100
            }
        }
        
        device = torch.device("cuda")

        def train_sweep(config=None):
            # Loop through all inner CV splits
            # for fold_idx, (X_train, y_train, X_val, y_val) in enumerate(inner_fold_splits):
            
            #fold_train_losses = []
            #fold_val_losses = []

            # Temporarily test with one fold
            fold_idx = 0
            X_train, X_val, y_train, y_val = inner_fold_splits[fold_idx]
            
            print('sweep id on inner call:', sweep_id)
            run = wandb.init(
                project="gx2conn_dynamicNN", # should already be set in sweep config
                name=f"fold_{fold_idx}",
                group=f"sweep_{sweep_id}",
                tags=["cross_validation", f"fold_{fold_idx}"],
                config=config,
                reinit=False
                )
            
            sweep_config = wandb.config

            model=DynamicNN(input_dim=sweep_config['input_dim'], hidden_dims=sweep_config['hidden_dims']).to(device)
            
            train_loader = model.create_data_loader(X_train, y_train, batch_size=sweep_config['batch_size'], shuffle=False)
            val_loader = model.create_data_loader(X_val, y_val, batch_size=sweep_config['batch_size'], shuffle=False)

            # Train full model on this fold
            fold_results = model.fit(train_loader, val_loader, epochs=sweep_config['epochs'], mode='learn', verbose=True)
            
            train_loss = fold_results['final_train_loss']
            val_loss = fold_results['final_val_loss']

            print('train loss', train_loss)
            print('val loss', val_loss)

            wandb.log({'train_loss': train_loss, 'val_loss': val_loss})

            #fold_train_losses.append(train_loss)
            #fold_val_losses.append(val_loss)
            # wandb.log({'fold_train_losses': fold_train_losses, 'fold_val_losses': fold_val_losses})
            # mean_train_loss = np.mean(fold_train_losses)
            # mean_val_loss = np.mean(fold_val_losses)
            # wandb.log({'mean_train_losses': mean_train_loss, 'mean_val_losses': mean_val_loss})

            run.finish()

            return val_loss

        sweep_id = wandb.sweep(sweep=sweep_config, project="gx2conn_dynamicNN")
        print('sweep id on outer call:', sweep_id)

        # Run sweep
        print('n_iter', n_iter)
        wandb.agent(sweep_id, function=train_sweep, count=n_iter)
        print('SWEEP COMPLETE')
        
        # Get best run from sweep
        api = wandb.Api()
        sweep = api.sweep(f"alexander-ratzan-new-york-university/gx2conn_dynamicNN/{sweep_id}")
        best_run = sweep.best_run()
        best_val_loss = best_run.summary.val_loss
        best_config = best_run.config
        print('best val loss', best_run.summary.val_loss)
        print('best run config', best_config)
            
        # Initialize final model with best config
        best_model = DynamicNN(input_dim=best_config['input_dim'], 
                               hidden_dims=best_config['hidden_dims'],
                               batch_size=best_config['batch_size'],
                               dropout_rate=best_config['dropout_rate'],
                               learning_rate=best_config['learning_rate'],
                               weight_decay=best_config['weight_decay']).to(device)
        
        wandb.finish()
        return best_model, best_val_loss


    def run_innercv(self, train_indices, test_indices, train_network_dict, search_method=('random', 'mse'), n_iter=100):
        """
        Inner cross-validation with option for Grid, Bayesian, or Randomized hyperparamter search
        """
        # Create inner CV object (just indices) for X_train and Y_train for any strategy 
        inner_cv_obj = SubnetworkCVSplit(train_indices, train_network_dict)

        if self.predict_connectome_from_connectome:
            inner_fold_splits = process_cv_splits_conn_only_model(self.Y, self.Y, inner_cv_obj,
                                                                  self.use_shared_regions,
                                                                  self.test_shared_regions)
        else:
            inner_fold_splits = process_cv_splits(self.X, self.Y, inner_cv_obj, 
                                                  self.use_shared_regions, 
                                                  self.test_shared_regions,
                                                  struct_summ=(True if self.summary_measure == 'strength_and_corr' and 'structural' in self.feature_type else False),
                                                  kron=(True if self.summary_measure == 'kronecker' else False),
                                                  kron_input_dim = self.PC_dim
                                                 )

        # Inner CV data packaged into a large matrix with indices for individual folds
        X_combined, Y_combined, train_test_indices = expanded_inner_folds_combined_plus_indices(inner_fold_splits)
        
        # Initialize model
        model = ModelBuild.init_model(self.model_type, X_combined.shape[1])
        param_grid = model.get_param_grid()
        param_dist = model.get_param_dist()

        # Unpack search method and metric
        search_type, metric = search_method
        # Initialize grid search and return cupy converted array if necessary
        if search_type == 'grid':
            param_search, X_combined, Y_combined = grid_search_init(self.gpu_acceleration, model, X_combined, Y_combined, param_grid, train_test_indices, metric=metric)
        elif search_type == 'random':
            param_search, X_combined, Y_combined = random_search_init(self.gpu_acceleration, model, X_combined, Y_combined, param_dist, train_test_indices, n_iter=n_iter, metric=metric)
        elif search_type == 'bayes':
            param_search, X_combined, Y_combined = bayes_search_init(self.gpu_acceleration, model, X_combined, Y_combined, param_dist, train_test_indices, n_iter=n_iter, metric=metric)

        # Fit GridSearchCV on the combined data
        param_search.fit(X_combined, Y_combined)
        
        # Display comprehensive results
        print("\nParameter Search CV Results:")
        print("=============================")
        print("Best Parameters: ", param_search.best_params_)
        print("Best Cross-Validation Score: ", param_search.best_score_)
        
        if search_type == 'bayes':
            _ = plot_objective(param_search.optimizer_results_[0], size=5)
            plt.show()

        best_model = model.get_model()
        best_model.set_params(**param_search.best_params_)

        return best_model, param_search.best_score_


    def run_sim(self, search_method=('random', 'mse')):
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
            print('SEARCH METHOD', search_method)
            if search_method[0] == 'wandb':
                wandb.login()
                best_model, best_val_score = self.run_innercv_wandb(train_indices, test_indices, train_network_dict, search_method=search_method, n_iter=2)
            else: # search_method options: random, grid, bayes
                best_model, best_val_score = self.run_innercv(train_indices, test_indices, train_network_dict, search_method=search_method, n_iter=100)

            # convert to cupy if necessary
            if self.gpu_acceleration:
                X_train = cp.array(X_train)
                Y_train = cp.array(Y_train)
                X_test = cp.array(X_test)
                Y_test = cp.array(Y_test)
            
            # Step 6: Retrain the best parameter model on training data and test on testing data
            best_model.fit(X_train, Y_train) # NEED TO CREATE COMPARABLE METHOD FOR EVALUATION 

            evaluator = ModelEvaluator(best_model, X_train, Y_train, X_test, Y_test, [self.use_shared_regions, self.include_conn_feats, self.test_shared_regions])
            
            train_metrics = evaluator.get_train_metrics()
            test_metrics = evaluator.get_test_metrics()

            print("\nTrain Metrics:", train_metrics)
            print("Test Metrics:", test_metrics)
            print('BEST VAL SCORE', best_val_score)
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
            
            self.results.append({ # Can we add validation accuracy here? 
                'model_parameters': best_model.get_params(),
                'train_metrics': train_metrics,
                'best_val_score': best_val_score,
                'test_metrics': test_metrics,
                'y_true': Y_test.get() if self.gpu_acceleration else Y_test,
                'y_pred': best_model.predict(X_test) if self.gpu_acceleration else best_model.predict(X_test), 
                'feature_importances': feature_importances_
            })

            # Display CPU and RAM utilization 
            print_system_usage()
            
            # Display GPU utilization
            GPUtil.showUtilization()