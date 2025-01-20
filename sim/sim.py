# GeneEx2Conn/sim/sim.py

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
    expand_shared_matrices,
    process_cv_splits, 
    expanded_inner_folds_combined_plus_indices,
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
from models.dynamic_mlp import DynamicMLP
from models.bilinear import BilinearLowRank, BilinearSCM
from models.shared_encoder_model import SharedMLPEncoderModel, SharedSelfAttentionModel
MODEL_CLASSES = {
    'dynamic_mlp': DynamicMLP,
    'bilinear_lowrank': BilinearLowRank,
    'bilinear_SCM': BilinearSCM,
    'shared_mlp_encoder': SharedMLPEncoderModel,
    'shared_transformer': SharedSelfAttentionModel
    # Add other deep learning models here as they're implemented
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
from sim.sim_utils import bayes_search_init, grid_search_init, random_search_init, drop_test_network, find_best_params, load_sweep_config, load_best_parameters, extract_feature_importances
from sim.sim_utils import bytes2human, print_system_usage, validate_inputs, train_sweep, log_wandb_metrics
importlib.reload(sim.sim_utils)


class Simulation:
    def __init__(self, feature_type, cv_type, model_type, gpu_acceleration, feature_interactions=None, resolution=1.0,random_seed=42,
                 omit_subcortical=False, parcellation='S100', gene_list='0.2', hemisphere='both',
                 use_shared_regions=False, test_shared_regions=False, connectome_target='FC', save_model_json=False, skip_cv=False):        
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
        self.resolution = resolution
        self.random_seed=random_seed
        self.omit_subcortical, self.parcellation, self.gene_list, self.hemisphere = omit_subcortical, parcellation, gene_list, hemisphere
        self.use_shared_regions = use_shared_regions
        self.test_shared_regions = test_shared_regions
        self.connectome_target = connectome_target.upper()
        self.skip_cv = skip_cv
        self.save_model_json = save_model_json
        self.results = []

    
    def load_data(self):
        """
        Load transcriptome and connectome data
        """
        self.X = load_transcriptome(parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, gene_list=self.gene_list, hemisphere=self.hemisphere)        
        self.X_pca = load_transcriptome(parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, gene_list=self.gene_list, run_PCA=True, hemisphere=self.hemisphere)
        self.Y_sc = load_connectome(parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, measure='SC', spectral=None, hemisphere=self.hemisphere)
        self.Y_sc_spectralL = load_connectome(parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, measure='SC', spectral='L', hemisphere=self.hemisphere)
        self.Y_sc_spectralA = load_connectome(parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, measure='SC', spectral='A', hemisphere=self.hemisphere)
        self.Y_fc = load_connectome(parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, measure='FC', hemisphere=self.hemisphere)
        self.coords = load_coords(parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, hemisphere=self.hemisphere)

        # Find rows that are not all NaN - necessary for gene expression data with unsampled regions
        valid_indices = ~np.isnan(self.X).all(axis=1)
        
        # Subset all data using valid indices
        self.X = self.X[valid_indices]
        self.X_pca = self.X_pca[valid_indices]
        self.Y_sc = self.Y_sc[valid_indices][:, valid_indices]
        self.Y_sc_spectralL = self.Y_sc_spectralL[valid_indices]
        self.Y_sc_spectralA = self.Y_sc_spectralA[valid_indices]
        self.Y_fc = self.Y_fc[valid_indices][:, valid_indices]
        self.coords = self.coords[valid_indices]

        print(f"X shape: {self.X.shape}")
        print(f"X_pca shape: {self.X_pca.shape}")
        print(f"Y_sc shape: {self.Y_sc.shape}")
        print(f"Y_sc_spectralL shape: {self.Y_sc_spectralL.shape}")
        print(f"Y_sc_spectralA shape: {self.Y_sc_spectralA.shape}")
        print(f"Y_fc shape: {self.Y_fc.shape}")
        print(f"Coordinates shape: {self.coords.shape}")

        # Define target connectome
        self.Y = self.Y_fc if self.connectome_target == 'FC' else self.Y_sc
        print('Y shape', self.Y.shape)
    
    
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
                print('feature_name: ', feature_name)
                print('processing_type: ', processing_type)
                features.append(feature_name if processing_type is None else f"{feature_name}_{processing_type}")

        # create a dict to map inputted feature types to data array - more can be addded to this
        # THIS IS THE NODE-WISE FEATURE DICT
        feature_dict = {'transcriptome': self.X,
                        'transcriptome_PCA': self.X_pca,
                        # add PLS embedding of transcriptome here
                        'structural': self.Y_sc,
                        'structural_spectral_L': self.Y_sc_spectralL,
                        'structural_spectral_A': self.Y_sc_spectralA,
                        'functional': self.Y_fc,
                        'euclidean': self.coords, 
                        'structural_spatial_null': np.hstack((self.coords, self.Y_sc)), # cannot be combined with other feats
                        'transcriptome_spatial_autocorr_null': np.hstack((self.coords, self.Y_sc, self.X_pca, self.X)), # cannot be combined with other feats
                        }
        
        validate_inputs(features=features, feature_dict=feature_dict)
        
        # append feature data into a horizontal stack indexed by node
        X = []
        for feature in features:
            if 'spectral' in feature:
                feature_type = '_'.join(feature.split('_')[:-1])  # take provided number of spectral components
                feature_X = feature_dict[feature_type]
                num_latents = int(feature.split('_')[-1])
                feature_X = feature_X[:, num_latents:] if num_latents < 0 else feature_X[:, :num_latents] # take first num_latents components if positive, last if negative
            else:
                if feature == 'structural_spatial_null':
                    spatial_null=True
                elif feature == 'transcriptome_spatial_autocorr_null':
                    transcriptome_spatial_null=True
                else:
                    spatial_null=False
                    transcriptome_spatial_null=False
    
                feature_X = feature_dict[feature]
            
            X.append(feature_X)
        
        print(X)
        self.X = np.hstack(X)
        print('X shape', self.X.shape)

        self.fold_splits = process_cv_splits(
                          self.X, self.Y, self.cv_obj, 
                          self.use_shared_regions,
                          self.test_shared_regions, 
                          spatial_null=spatial_null, 
                          transcriptome_spatial_null=transcriptome_spatial_null
                          )

                        
    def run_innercv_wandb(self, X_train, Y_train, X_test, Y_test,train_indices, test_indices, train_network_dict, outer_fold_idx, search_method=('random', 'mse', 3)):
        """Inner cross-validation with W&B support for deep learning models"""
        
        # Create inner CV object for X_train and Y_train
        inner_cv_obj = SubnetworkCVSplit(train_indices, train_network_dict)
        inner_fold_splits = process_cv_splits(
            self.X, self.Y, inner_cv_obj,
            self.use_shared_regions,
            self.test_shared_regions)
        
        # Load sweep config
        sweep_config_path = os.path.join(os.getcwd(), 'models', 'configs', f'{self.model_type}_sweep_config.yml')
        input_dim = inner_fold_splits[0][0].shape[1]
        sweep_config = load_sweep_config(sweep_config_path, input_dim=input_dim) # take num features from a fold
        device = torch.device("cuda")

        if self.skip_cv: 
            best_config = load_best_parameters(sweep_config_path, input_dim=input_dim)
        else:
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
                    model_classes=MODEL_CLASSES,
                    parcellation=self.parcellation, 
                    hemisphere=self.hemisphere, 
                    omit_subcortical=self.omit_subcortical, 
                    gene_list=self.gene_list, 
                )
            
            # Initialize sweep
            sweep_id = wandb.sweep(sweep=sweep_config, project="gx2conn")

            # Run sweep
            wandb.agent(sweep_id, function=train_sweep_wrapper, count=search_method[2])

            # Get best run from sweep
            api = wandb.Api()
            sweep = api.sweep(f"alexander-ratzan-new-york-university/gx2conn/{sweep_id}")
            best_run = sweep.best_run()
            wandb.teardown()

            best_val_loss = best_run.summary.mean_val_loss # this can be changed to another metric
            best_config = best_run.config

        print('BEST CONFIG', best_config)

        # Initialize final model with best config
        ModelClass = MODEL_CLASSES[self.model_type]
        best_model = ModelClass(**best_config).to(device)
        best_val_loss = 0.0
            
        return best_model, best_val_loss


    def run_innercv(self, train_indices, test_indices, train_network_dict, search_method=('random', 'mse', 10)):
        """
        Inner cross-validation with option for Grid, Bayesian, or Randomized hyperparameter search for sklearn-like models
        """
        # Create inner CV object (just indices) for X_train and Y_train for any strategy
        inner_cv_obj = SubnetworkCVSplit(train_indices, train_network_dict)

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

        search_type, metric, n_iter = search_method        
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
        
        # if search_type == 'bayes': _ = plot_objective(param_search.optimizer_results_[0], size=5); plt.show() # Display objective plots for hyperparameter search
        
        best_model = model.get_model()
        best_model.set_params(**param_search.best_params_)

        return best_model, param_search.best_score_


    def run_sim(self, search_method=('random', 'mse', 5), track_wandb=False):
        """
        Main simulation method
        """
        self.load_data()
        self.select_cv()
        self.expand_data()

        # Outer CV
        for fold_idx, (X_train, X_test, Y_train, Y_test) in enumerate(self.fold_splits):
            print('\n', f'Test fold num: {fold_idx+1}', f'X_train shape: {X_train.shape}', f'Y_train shape: {Y_train.shape}', f'X_test shape: {X_test.shape}', f'Y_test shape: {Y_test.shape}')
            
            train_indices = self.cv_obj.folds[fold_idx][0]
            test_indices = self.cv_obj.folds[fold_idx][1]
            network_dict = self.cv_obj.networks
            train_network_dict = drop_test_network(self.cv_type, network_dict, test_indices, fold_idx+1)
            
            if self.gpu_acceleration:
                X_train, Y_train, X_test, Y_test = map(cp.array, [X_train, Y_train, X_test, Y_test])
            if search_method[0] == 'wandb' or track_wandb:
                wandb.login()

            # Inner CV on current training fold
            if search_method[0] == 'wandb':
                best_model, best_val_score = self.run_innercv_wandb(X_train, Y_train, X_test, Y_test, train_indices, test_indices, train_network_dict, fold_idx, search_method=search_method)                
                train_history = best_model.fit(X_train, Y_train, X_test, Y_test)
            else:
                best_model, best_val_score = self.run_innercv(train_indices, test_indices, train_network_dict, search_method=search_method)
                best_model.fit(X_train, Y_train)
                train_history = None
            
            # Evaluate on the test fold                
            evaluator = ModelEvaluator(best_model, X_train, Y_train, X_test, Y_test, self.use_shared_regions, self.test_shared_regions)
            train_metrics = evaluator.get_train_metrics()
            test_metrics = evaluator.get_test_metrics()
            
            # Display final evaluation metrics
            print("\nTRAIN METRICS:", train_metrics)
            print("TEST METRICS:", test_metrics)
            print('BEST VAL SCORE', best_val_score)
            print('BEST MODEL PARAMS', best_model.get_params())

            # Log final evaluation metrics
            if track_wandb:
                log_wandb_metrics(
                    self.feature_type, self.model_type, self.connectome_target, self.cv_type, 
                    fold_idx,
                    train_metrics, 
                    test_metrics, 
                    best_val_score, 
                    best_model, 
                    train_history, 
                    model_classes=MODEL_CLASSES,
                    parcellation=self.parcellation, 
                    hemisphere=self.hemisphere, 
                    omit_subcortical=self.omit_subcortical, 
                    gene_list=self.gene_list
                )

            # Extract feature importances and model JSON
            feature_importances_, model_json = extract_feature_importances(
                self.model_type, 
                best_model, 
                self.save_model_json
            )

            # Save results to pickle file
            self.results.append({
                'model_parameters': best_model.get_params(),
                'train_metrics': train_metrics,
                'best_val_score': best_val_score,
                'test_metrics': test_metrics,
                'y_true': Y_test.get() if self.gpu_acceleration else Y_test,
                'y_pred': best_model.predict(X_test),
                'feature_importances': feature_importances_,
                'model_json': model_json
            })

            print_system_usage() # Display CPU and RAM utilization 
            GPUtil.showUtilization() # Display GPU utilization

            break