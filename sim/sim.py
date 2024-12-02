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
MODEL_CLASSES = {
    'dynamic_nn': DynamicNN,
    # Add other neural network models here as they're implemented
    # 'transformer_nn': TransformerNN
}

# metric classes
from models.metrics.eval import (
    ModelEvaluator,
    pearson_numpy,
    pearson_cupy,
    mse_numpy,
    mse_cupy,
    r2_numpy,
    r2_cupy
)
import models.metrics.eval
importlib.reload(models.metrics.eval)

# sim utility functions
import sim.sim_utils
from sim.sim_utils import bayes_search_init, grid_search_init, random_search_init, drop_test_network, find_best_params, load_sweep_config
from sim.sim_utils import bytes2human, print_system_usage, validate_inputs, train_sweep, log_wandb_metrics
importlib.reload(sim.sim_utils)

class Simulation:
    def __init__(self, feature_type, cv_type, model_type, gpu_acceleration, predict_connectome_from_connectome=False, feature_interactions=None, resolution=1.0,random_seed=42,
                 use_shared_regions=False, include_conn_feats=False, test_shared_regions=False, connectome_target='FC', save_model_json=False):        
        """
        Initialization of simulation parameters
        """
        validate_inputs(
            cv_type=cv_type,
            model_type=model_type,
            connectome_target=connectome_target
        )
        
        # consider storing in a config
        self.cv_type = cv_type
        self.model_type = model_type
        self.gpu_acceleration = gpu_acceleration
        self.feature_type = feature_type
        self.feature_interactions = feature_interactions
        self.predict_connectome_from_connectome = predict_connectome_from_connectome
        self.resolution = resolution
        self.random_seed=random_seed
        self.use_shared_regions = use_shared_regions
        self.include_conn_feats = include_conn_feats
        self.test_shared_regions = test_shared_regions
        self.connectome_target = connectome_target.upper()
        self.save_model_json = save_model_json
        self.results = []

    
    def load_data(self):
        """
        Load transcriptome and connectome data
        """
        self.X = load_transcriptome()
        self.X_pca = load_transcriptome(run_PCA=True)
        self.Y_sc = load_connectome(measure='SC', spectral=None)
        self.Y_sc_spectralL = load_connectome(measure='SC', spectral='L')
        self.Y_sc_spectralA = load_connectome(measure='SC', spectral='A')
        self.Y_fc = load_connectome(measure='FC')
        self.Y = self.Y_fc if self.connectome_target == 'FC' else self.Y_sc
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
        # create a list of features to be expanded into edge-wise dataset
        features = []
        for feature_dict in self.feature_type:
            for feature_name, processing_type in feature_dict.items():
                print('feature_name', feature_name)
                print('processing_type', processing_type)
                if processing_type is None:
                    features.append(feature_name)
                else:
                    features.append(feature_name + '_' + processing_type)

        # create a dict to map inputted feature types to data array - more can be addded to this
        feature_dict = {'transcriptome': self.X, 
                        'transcriptome_PCA': self.X_pca,
                        'structural':self.Y_sc,
                        'structural_spectral_L':self.Y_sc_spectralL,
                        'structural_spectral_A':self.Y_sc_spectralA, 
                        'functional':self.Y_fc,
                        'euclidean':self.coords
                        }
        validate_inputs(features=features, feature_dict=feature_dict)

        # append feature data into a horizontal stack indexed by node
        X = []

        for feature in features:
            if 'spectral' in feature:
                feature_type = '_'.join(feature.split('_')[:-1])  # Take everything before the number
                feature_X = feature_dict[feature_type]
                num_latents = int(feature.split('_')[-1])
                if num_latents < 0:
                    feature_X = feature_X[:, num_latents:] # Take columns from num_latents to end
                else:
                    feature_X = feature_X[:, :num_latents]    # Take columns from start up to num_latents
            else:
                feature_X = feature_dict[feature]
            
            X.append(feature_X)
        
        self.X = np.hstack(X)
        print('X shape', self.X.shape)

        '''        
        # temporarily suspend feature interactions for now
        # to add back in need to loop through feature_interactions and cleverly save indices from X to apply interaction from there
        if 'transcriptome_PCA' in self.feature_type:
            feature_X = feature_dict['transcriptome_PCA']
            self.PC_dim = int(feature_X.shape[1])
        else:
            self.PC_dim = None # save dimensionality for kronecker for region-wise expansion
 
        if self.feature_interactions == 'kronecker':
            kron = True
        '''

        self.fold_splits = process_cv_splits(self.X, self.Y, self.cv_obj, 
                          self.use_shared_regions, 
                          self.include_conn_feats, 
                          self.test_shared_regions
                          )
                          #struct_summ=(True if self.summary_measure == 'strength_and_corr' and 'structural' in self.feature_type else False),
                          #kron=(True if self.summary_measure == 'kronecker' else False),
                          #kron_input_dim = self.PC_dim) # adjust this syntax to be more general

                        
    def run_innercv_wandb(self, X_train, Y_train, X_test, Y_test,train_indices, test_indices, train_network_dict, outer_fold_idx, search_method=('wandb', 'mse'), n_iter=10):
        """Inner cross-validation with W&B support for deep learning models"""
        
        # Create inner CV object for X_train and Y_train
        inner_cv_obj = SubnetworkCVSplit(train_indices, train_network_dict)
        
        inner_fold_splits = process_cv_splits(
            self.X, self.Y, 
            inner_cv_obj,
            self.use_shared_regions,
            self.test_shared_regions
        )
        
        # Load sweep config
        sweep_config_path = os.path.join(os.getcwd(), 'models', f'{self.model_type}_sweep_config.yml')
        sweep_config = load_sweep_config(sweep_config_path, input_dim=inner_fold_splits[0][0].shape[1]) # take num features from a fold
        
        device = torch.device("cuda")

        def train_sweep_wrapper(config=None):
            return train_sweep(
                config=config,
                model_type=self.model_type,
                feature_type=self.feature_type,
                connectome_target=self.connectome_target,
                cv_type=self.cv_type,
                outer_fold_idx=outer_fold_idx,
                inner_fold_splits=inner_fold_splits,
                device=device,
                sweep_id=sweep_id,
                model_classes=MODEL_CLASSES
            )
        
        # Initialize sweep
        sweep_id = wandb.sweep(sweep=sweep_config, project="gx2conn")

        # Run sweep
        wandb.agent(sweep_id, function=train_sweep_wrapper, count=n_iter)

        # Get best run from sweep
        api = wandb.Api()
        sweep = api.sweep(f"alexander-ratzan-new-york-university/gx2conn/{sweep_id}")
        best_run = sweep.best_run()
        best_val_loss = best_run.summary.mean_val_loss
        best_config = best_run.config
        
        # Initialize final model with best config
        ModelClass = MODEL_CLASSES[self.model_type]
        best_model = ModelClass(**best_config).to(device)
            
        return best_model, best_val_loss


    def run_innercv(self, train_indices, test_indices, train_network_dict, search_method=('random', 'mse'), n_iter=100):
        """
        Inner cross-validation with option for Grid, Bayesian, or Randomized hyperparameter search for sklearn-like models
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
                                                  self.test_shared_regions
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
        
        # Display objective plots for hyperparameter search
        # if search_type == 'bayes': _ = plot_objective(param_search.optimizer_results_[0], size=5); plt.show()
        
        best_model = model.get_model()
        best_model.set_params(**param_search.best_params_)

        return best_model, param_search.best_score_


    def run_sim(self, search_method=('random', 'mse'), track_wandb=False):
        """
        Main simulation method
        """
        self.load_data()           # Step 1: Load data
        self.select_cv()           # Step 2: Select CV strategy
        self.expand_data()         # Step 3: Expand data based on CV strategy

        # Step 4: Outer CV on folds
        for fold_idx, (X_train, X_test, Y_train, Y_test) in enumerate(self.fold_splits):
            print('\n', f'Test fold num: {fold_idx+1}')
            print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
            
            train_indices = self.cv_obj.folds[fold_idx][0]
            test_indices = self.cv_obj.folds[fold_idx][1]
   
            network_dict = self.cv_obj.networks
            train_network_dict = drop_test_network(self.cv_type, network_dict, test_indices, fold_idx+1)

            if self.gpu_acceleration:
                X_train = cp.array(X_train)
                Y_train = cp.array(Y_train)
                X_test = cp.array(X_test)
                Y_test = cp.array(Y_test)

            # Step 5: Inner CV on training data
            print('SEARCH METHOD', search_method)
            if track_wandb and self.model_type in ['dynamic_nn']: # need to change this to run for any epoch based model
                wandb.login()
                best_model, best_val_score = self.run_innercv_wandb(X_train, Y_train, X_test, Y_test, train_indices, test_indices, train_network_dict, fold_idx, search_method=search_method, n_iter=5)                
                train_history = best_model.fit(X_train, Y_train, val_data=(X_test, Y_test))
            else:
                best_model, best_val_score = self.run_innercv(train_indices, test_indices, train_network_dict, search_method=search_method, n_iter=100)
                best_model.fit(X_train, Y_train)
                train_history = None
            
            # Step 6: Evaluate on the test fold
            evaluator = ModelEvaluator(
                best_model, X_train, Y_train, X_test, Y_test,
                [self.use_shared_regions, self.include_conn_feats, self.test_shared_regions]
            )
            train_metrics = evaluator.get_train_metrics()
            test_metrics = evaluator.get_test_metrics()
            
            # Display final evaluation metrics
            print("\nTrain Metrics:", train_metrics)
            print("Test Metrics:", test_metrics)
            print('BEST VAL SCORE', best_val_score)
            print('BEST MODEL PARAMS', best_model.get_params())

            # Step 7: Log final evaluation metrics
            if track_wandb:
                log_wandb_metrics(self.feature_type, self.model_type, self.connectome_target, self.cv_type, fold_idx, train_metrics, test_metrics, best_val_score, best_model, train_history)
            
            # Set model specific parameters to be saved in pickle file (such as feature importances or model JSON)
            model_json = None
            if self.model_type == 'pls':
                feature_importances_ = best_model.x_weights_[:, 0]  # Weights for the first component
            elif self.model_type == 'xgboost': 
                feature_importances_ = best_model.feature_importances_
                if self.save_model_json:
                    booster = best_model.get_booster()
                    model_json = booster.save_raw("json").decode("utf-8")   # Save JSON as string in memory
            elif self.model_type == 'ridge': 
                feature_importances_ = best_model.coef_
            else: 
                feature_importances_ = None

            # Step 8: Save results to pickle file
            self.results.append({
                'model_parameters': best_model.get_params(),
                'train_metrics': train_metrics,
                'best_val_score': best_val_score,
                'test_metrics': test_metrics,
                'y_true': Y_test.get() if self.gpu_acceleration else Y_test,
                'y_pred': best_model.predict(X_test) if self.gpu_acceleration else best_model.predict(X_test),
                'feature_importances': feature_importances_,
                'model_json': model_json
            })

            # Display CPU and RAM utilization 
            print_system_usage()
            
            # Display GPU utilization
            GPUtil.showUtilization()