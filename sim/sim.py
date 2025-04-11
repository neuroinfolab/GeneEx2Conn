from env.imports import *

from data.data_load import load_transcriptome, load_connectome, load_coords, load_network_labels

from data.cv_split import (
    RandomCVSplit, 
    SchaeferCVSplit, 
    CommunityCVSplit, 
    SubnetworkCVSplit,
    SpatialCVSplit
)

from data.data_utils import (
    process_cv_splits, 
    process_cv_splits_coords, 
    expanded_inner_folds_combined_plus_indices, 
    expand_X_symmetric,
    expand_Y_symmetric,
    RegionPairDataset
)

from models.base_models import ModelBuild, BaseModel
from models.dynamic_mlp import DynamicMLP
from models.bilinear import BilinearLowRank, BilinearSCM
from models.shared_encoder_models import SharedMLPEncoderModel, SharedLinearEncoderModel
from models.transformer_models import SharedSelfAttentionModel, SharedSelfAttentionCLSModel, CrossAttentionModel

MODEL_CLASSES = {
    'dynamic_mlp': DynamicMLP,
    'bilinear_lowrank': BilinearLowRank,
    'bilinear_SCM': BilinearSCM,
    'shared_mlp_encoder': SharedMLPEncoderModel,
    'shared_linear_encoder': SharedLinearEncoderModel,
    'shared_transformer': SharedSelfAttentionModel,
    'shared_transformer_cls': SharedSelfAttentionCLSModel,
    'cross_attention': CrossAttentionModel
}

from models.metrics.eval import (
    ModelEvaluator,
    ModelEvaluatorTorch
)

from sim.sim_utils import (
    bytes2human, 
    print_system_usage, 
    extract_model_params
)

from sim.sim_utils import (
    set_seed,
    bayes_search_init, 
    grid_search_init, 
    random_search_init, 
    drop_test_network, 
    find_best_params,  
    extract_feature_importances
)

from sim.sim_utils import (
    train_sweep,
    train_sweep_torch,
    load_sweep_config, 
    load_best_parameters
    # log_wandb_metrics
)


class Simulation:
    def __init__(self, feature_type, cv_type, model_type, gpu_acceleration, resolution=1.0, random_seed=42,
                 omit_subcortical=False, parcellation='S100', impute_strategy='mirror_interpolate', sort_genes='expression', 
                 gene_list='0.2', hemisphere='both', use_shared_regions=False, test_shared_regions=False, 
                 connectome_target='FC', binarize=False, skip_cv=False, species="human"):        
        """
        Initialization of simulation parameters
        """
        self.feature_type = feature_type
        self.cv_type = cv_type
        self.model_type = model_type
        self.gpu_acceleration = gpu_acceleration
        self.resolution = resolution
        self.random_seed = random_seed
        self.omit_subcortical, self.parcellation, self.impute_strategy, self.sort_genes, self.gene_list, self.hemisphere = omit_subcortical, parcellation, impute_strategy, sort_genes, gene_list, hemisphere
        self.use_shared_regions = use_shared_regions
        self.test_shared_regions = test_shared_regions
        self.connectome_target = connectome_target.upper()
        self.binarize = binarize
        self.skip_cv = skip_cv
        self.results = []
        self.species = species

    
    def load_data(self):
        """
        Load transcriptome and connectome data
        """
        dataset = "AHBA"
        network_dataset = "HCP"
        if self.species == "c_elegans":
            dataset = "c_elegans"
            network_dataset = "c_elegans"
        
        self.X = load_transcriptome(dataset=dataset, parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, gene_list=self.gene_list, hemisphere=self.hemisphere, impute_strategy=self.impute_strategy, sort_genes=self.sort_genes)        
        self.X_pca = load_transcriptome(dataset=dataset, parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, gene_list=self.gene_list, run_PCA=True, hemisphere=self.hemisphere)
        self.Y_sc = load_connectome(dataset=dataset, parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, measure='SC', spectral=None, hemisphere=self.hemisphere)
        self.Y_sc_binary = load_connectome(dataset=dataset, parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, measure='SC', binarize=True, hemisphere=self.hemisphere)
        self.Y_sc_spectralL = load_connectome(dataset=dataset, parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, measure='SC', spectral='L', hemisphere=self.hemisphere)
        self.Y_sc_spectralA = load_connectome(dataset=dataset, parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, measure='SC', spectral='A', hemisphere=self.hemisphere)
        self.Y_fc = load_connectome(dataset=dataset, parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, measure='FC', hemisphere=self.hemisphere)
        self.Y_fc_binary = load_connectome(dataset=dataset, parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, measure='FC', binarize=True, hemisphere=self.hemisphere)
        self.coords = load_coords(dataset=dataset, parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, hemisphere=self.hemisphere)
        self.labels, self.network_labels = load_network_labels(dataset=network_dataset, parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, hemisphere=self.hemisphere)

        # Find rows that are not all NaN - necessary for gene expression data with unsampled regions
        valid_indices = ~np.isnan(self.X).all(axis=1)

        # Create index map so we know true indices of valid data
        valid_indices_values = np.where(valid_indices)[0]
        valid2true_index_mapping = dict(enumerate(valid_indices_values))
        self.valid2true_index_mapping = valid2true_index_mapping

      
        # Subset all data using valid indices
        self.X = self.X[valid_indices]
        self.X_pca = self.X_pca[valid_indices]
        self.Y_sc = self.Y_sc[valid_indices][:, valid_indices]
        self.Y_sc_binary = self.Y_sc_binary[valid_indices][:, valid_indices]
        self.Y_sc_spectralL = self.Y_sc_spectralL[valid_indices]
        self.Y_sc_spectralA = self.Y_sc_spectralA[valid_indices]
        self.Y_fc = self.Y_fc[valid_indices][:, valid_indices]
        self.Y_fc_binary = self.Y_fc_binary[valid_indices][:, valid_indices]
        self.coords = self.coords[valid_indices]
        self.labels = [self.labels[i] for i in range(len(self.labels)) if valid_indices[i]]
        self.network_labels = self.network_labels[valid_indices]

        print(f"X shape: {self.X.shape}")
        print(f"X_pca shape: {self.X_pca.shape}")
        print(f"Y_sc shape: {self.Y_sc.shape}")
        print(f"Y_sc_spectralL shape: {self.Y_sc_spectralL.shape}")
        print(f"Y_sc_spectralA shape: {self.Y_sc_spectralA.shape}")
        print(f"Y_fc shape: {self.Y_fc.shape}")
        print(f"Coordinates shape: {self.coords.shape}")

        # Define target connectome
        if self.binarize:
            self.Y = self.Y_fc_binary if self.connectome_target == 'FC' else self.Y_sc_binary
        else:
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
        elif self.cv_type == 'spatial':
            self.cv_obj = SpatialCVSplit(self.X, self.Y, self.coords, num_splits=4, random_seed=self.random_seed)
    
    def expand_data_torch(self):
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

        feature_dict = {'transcriptome': self.X,
                        'transcriptome_PCA': self.X_pca,
                        'structural': self.Y_sc,
                        'structural_spectral_L': self.Y_sc_spectralL,
                        'structural_spectral_A': self.Y_sc_spectralA,
                        'functional': self.Y_fc,
                        'euclidean': self.coords, 
                        'structural_spatial_null': np.hstack((self.coords, self.Y_sc)), # cannot be combined with other feats
                        'transcriptome_spatial_autocorr_null': np.hstack((self.coords, self.Y_sc, self.X_pca, self.X)), # cannot be combined with other feats
                        }
        
        self.features = features 
        print('features', self.features)

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
                    transcriptome_spatial_null=None
                elif feature == 'transcriptome_spatial_autocorr_null':
                    spatial_null=False
                    transcriptome_spatial_null=[self.coords.shape[1], self.Y_sc.shape[1], self.X_pca.shape[1], self.X.shape[1]]
                else:
                    spatial_null=False
                    transcriptome_spatial_null=None
    
                feature_X = feature_dict[feature]
            
            X.append(feature_X)
        
        self.X = np.hstack(X)
        print('X generated... expanding to pairwise dataset')

        # Create custom dataset for region pair data
        self.region_pair_dataset = RegionPairDataset(
            self.X, 
            self.Y,
            self.coords,
            self.valid2true_index_mapping
        )


    def run_innercv_wandb_torch(self, input_dim, train_indices, train_network_dict, outer_fold_idx, search_method=('random', 'mse', 3)):
        """Inner cross-validation with W&B support for deep learning models"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sweep_config_path = os.path.join(os.getcwd(), 'models', 'configs', f'{self.model_type}_sweep_config.yml')
        sweep_config = load_sweep_config(sweep_config_path, input_dim=input_dim, binarize=self.binarize)
        
        inner_cv_obj = SubnetworkCVSplit(train_indices, train_network_dict)

        if self.skip_cv: 
            best_config = load_best_parameters(sweep_config_path, input_dim=input_dim, binarize=self.binarize)
            best_val_loss = 0.0 # no CV --> no best val loss
        else:
            def train_sweep_wrapper(config=None):
                return train_sweep_torch(
                    config=config,
                    model_type=self.model_type,
                    feature_type=self.feature_type,
                    connectome_target=self.connectome_target,
                    dataset=self.region_pair_dataset,
                    cv_type=self.cv_type,
                    cv_obj=inner_cv_obj,
                    outer_fold_idx=outer_fold_idx,
                    device = device,
                    sweep_id=sweep_id,
                    model_classes=MODEL_CLASSES,
                    parcellation=self.parcellation, 
                    hemisphere=self.hemisphere, 
                    omit_subcortical=self.omit_subcortical, 
                    gene_list=self.gene_list, 
                    seed=self.random_seed,
                    binarize=self.binarize,
                    impute_strategy=self.impute_strategy,
                    sort_genes=self.sort_genes
                )
            
            # Initialize sweep
            sweep_id = wandb.sweep(sweep=sweep_config, project="gx2conn")
            print('sweep_id', sweep_id)

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
            
        return best_model, best_val_loss


    def run_sim_torch(self, search_method=('random', 'mse', 5), track_wandb=False):
        """
        Main simulation method
        """
        set_seed(self.random_seed)
        self.load_data()
        self.select_cv()

        self.expand_data_torch()
    
        if search_method[0] == 'wandb' or track_wandb:
            wandb.login()
        
        network_dict = self.cv_obj.networks

        for fold_idx, (train_indices, test_indices) in enumerate(self.cv_obj.split(self.X, self.Y)):
        
            train_region_pairs = expand_X_symmetric(train_indices.reshape(-1, 1)).astype(int)
            test_region_pairs = expand_X_symmetric(test_indices.reshape(-1, 1)).astype(int)

            train_indices_expanded = np.array([self.region_pair_dataset.valid_pair_to_expanded_idx[tuple(pair)] for pair in train_region_pairs])
            test_indices_expanded = np.array([self.region_pair_dataset.valid_pair_to_expanded_idx[tuple(pair)] for pair in test_region_pairs])
            
            innercv_network_dict = drop_test_network(self.cv_type, network_dict, test_indices, fold_idx+1)
            input_dim = self.region_pair_dataset.X_expanded[0].shape[0]
            best_model, best_val_score = self.run_innercv_wandb_torch(input_dim, train_indices, innercv_network_dict, fold_idx, search_method)
            
            if track_wandb:
                feature_str = "+".join(str(k) if v is None else f"{k}_{v}" 
                                for feat in self.feature_type 
                                for k,v in feat.items())
                run_name = f"{self.model_type}_{feature_str}_{self.connectome_target}_{self.cv_type}_fold{fold_idx}_final_eval"
                final_eval_run = wandb.init(project="gx2conn",
                                            name=run_name,
                                            tags=["final_eval", 
                                                  f'cv_type_{self.cv_type}', 
                                                  f'outerfold_{fold_idx}',
                                                  f'model_{self.model_type}',
                                                  f"split_{self.cv_type}{self.random_seed}", 
                                                  f'feature_type_{feature_str}',
                                                  f'target_{self.connectome_target}',
                                                  f'parcellation_{self.parcellation}',
                                                  f'hemisphere_{self.hemisphere}',
                                                  f'omit_subcortical_{self.omit_subcortical}',
                                                  f'gene_list_{self.gene_list}',
                                                  f"binarize_{self.binarize}",
                                                  f"impute_strategy_{self.impute_strategy}",
                                                  f"sort_genes_{self.sort_genes}"],
                                            reinit=True)

                if self.model_type in MODEL_CLASSES:
                    wandb.watch(best_model, log='all')
                    train_history = best_model.fit(self.region_pair_dataset, train_indices_expanded, test_indices_expanded)
                    for epoch, (train_loss, val_loss) in enumerate(zip(train_history['train_loss'], train_history['val_loss'])):
                        wandb.log({'train_mse_loss': train_loss, 'test_mse_loss': val_loss})
            else:
                train_history = best_model.fit(self.region_pair_dataset, train_indices_expanded, test_indices_expanded)

            train_dataset = Subset(self.region_pair_dataset, train_indices_expanded)
            test_dataset = Subset(self.region_pair_dataset, test_indices_expanded)
            train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, pin_memory=True)

            # Evaluate on the test fold
            evaluator = ModelEvaluatorTorch(region_pair_dataset=self.region_pair_dataset,
                                            model=best_model,
                                            Y=self.Y,
                                            train_loader=train_loader,
                                            train_indices=train_indices,
                                            train_indices_expanded=train_indices_expanded,
                                            test_loader=test_loader,
                                            test_indices=test_indices,
                                            test_indices_expanded=test_indices_expanded,
                                            network_labels=self.network_labels,
                                            train_shared_regions=self.use_shared_regions,
                                            test_shared_regions=self.test_shared_regions)

            train_metrics = evaluator.get_train_metrics()
            test_metrics = evaluator.get_test_metrics()
            print("\nTRAIN METRICS:", train_metrics)
            print("TEST METRICS:", test_metrics)
            print('BEST VAL SCORE', best_val_score)
            print('BEST MODEL HYPERPARAMS', best_model.get_params() if hasattr(best_model, 'get_params') else extract_model_params(best_model))

            if track_wandb:
                wandb.log({'final_train_metrics': train_metrics, 'final_test_metrics': test_metrics, 'best_val_loss': best_val_score, 'config': best_model.get_params() if hasattr(best_model, 'get_params') else extract_model_params(best_model)})
                final_eval_run.finish()
                wandb.finish()
                print("Final evaluation metrics logged successfully.")

        print_system_usage() # Display CPU and RAM utilization
        GPUtil.showUtilization() # Display GPU utilization

    


    #########################################################
    #### FUNCTIONALITY FOR NON-GPU SKLEARN BASED MODELS #####
    #########################################################
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

        # create a dict to map inputted feature types to data array - more can be added to this
        # THIS IS THE NODE-WISE FEATURE DICT
        feature_dict = {'transcriptome': self.X,
                        'transcriptome_PCA': self.X_pca,
                        # add PLS embedding of transcriptome here? 
                        'structural': self.Y_sc,
                        'structural_spectral_L': self.Y_sc_spectralL,
                        'structural_spectral_A': self.Y_sc_spectralA,
                        'functional': self.Y_fc,
                        'euclidean': self.coords, 
                        'structural_spatial_null': np.hstack((self.coords, self.Y_sc)), # cannot be combined with other feats
                        'transcriptome_spatial_autocorr_null': np.hstack((self.coords, self.Y_sc, self.X_pca, self.X)), # cannot be combined with other feats
                        }
        
        # validate_inputs(features=features, feature_dict=feature_dict)
        self.features = features 
        print('features', self.features)

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
                    transcriptome_spatial_null=None
                elif feature == 'transcriptome_spatial_autocorr_null':
                    spatial_null=False
                    transcriptome_spatial_null=[self.coords.shape[1], self.Y_sc.shape[1], self.X_pca.shape[1], self.X.shape[1]]
                else:
                    spatial_null=False
                    transcriptome_spatial_null=None
    
                feature_X = feature_dict[feature]
            
            X.append(feature_X)
        
        self.X = np.hstack(X)
        print('X shape', self.X.shape)
        
        self.fold_splits = process_cv_splits(
                          self.X, self.Y, self.cv_obj, 
                          self.use_shared_regions,
                          self.test_shared_regions,
                          spatial_null=spatial_null, 
                          transcriptome_spatial_null=transcriptome_spatial_null)

        self.fold_splits_coords = process_cv_splits_coords(
                                    self.X,
                                    self.Y,
                                    self.coords,
                                    self.cv_obj)
        
    def run_innercv(self, train_indices, test_indices, train_network_dict, search_method=('random', 'mse', 10)):
        """
        Inner cross-validation with option for Grid, Bayesian, or Randomized hyperparameter search for sklearn-like models
        """
        # Create inner CV object (just indices) for X_train and Y_train for any strategy
        inner_cv_obj = SubnetworkCVSplit(train_indices, train_network_dict)

        inner_fold_splits = process_cv_splits(self.X, self.Y, inner_cv_obj, 
                                                self.use_shared_regions, 
                                                self.test_shared_regions)

        # Inner CV data packaged into a large matrix with indices for individual folds
        X_combined, Y_combined, train_test_indices = expanded_inner_folds_combined_plus_indices(inner_fold_splits)
    
        model = ModelBuild.init_model(self.model_type, self.binarize)

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


    def run_innercv_wandb(self, input_dim, train_indices, test_indices, train_network_dict, outer_fold_idx, search_method=('random', 'mse', 3)):
        """Inner cross-validation with W&B support for deep learning models"""
        
        # Create inner CV object for X_train and Y_train
        inner_cv_obj = SubnetworkCVSplit(train_indices, train_network_dict)
        inner_fold_splits = process_cv_splits(
            self.X, self.Y, inner_cv_obj,
            self.use_shared_regions,
            self.test_shared_regions)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load sweep config
        sweep_config_path = os.path.join(os.getcwd(), 'models', 'configs', f'{self.model_type}_sweep_config.yml')
        sweep_config = load_sweep_config(sweep_config_path, input_dim=input_dim, binarize=self.binarize)
        
        if self.skip_cv: 
            best_config = load_best_parameters(sweep_config_path, input_dim=input_dim, binarize=self.binarize)
            best_val_loss = 0.0 # no CV --> no best val loss
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
                    device = device,
                    sweep_id=sweep_id,
                    model_classes=MODEL_CLASSES,
                    parcellation=self.parcellation, 
                    hemisphere=self.hemisphere, 
                    omit_subcortical=self.omit_subcortical, 
                    gene_list=self.gene_list, 
                    seed=self.random_seed,
                    binarize=self.binarize,
                    impute_strategy=self.impute_strategy,
                    sort_genes=self.sort_genes
                )
            
            # Initialize sweep
            sweep_id = wandb.sweep(sweep=sweep_config, project="gx2conn")
            # sweep_id = f"{self.model_type}_{sweep_id}" #  # figure out how to tag with {self.model_type}
            print('sweep_id', sweep_id)

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
            
        return best_model, best_val_loss


    def run_sim(self, search_method=('random', 'mse', 5), track_wandb=False):
        """
        Main simulation method
        """
        set_seed(self.random_seed)
        self.load_data()
        self.select_cv()
        self.expand_data()        
    
        if search_method[0] == 'wandb' or track_wandb:
            wandb.login()
    
        # Outer CV
        for fold_idx, (X_train, X_test, Y_train, Y_test) in enumerate(self.fold_splits):
            print('\n', f'Test fold num: {fold_idx+1}', f'X_train shape: {X_train.shape}', f'Y_train shape: {Y_train.shape}', f'X_test shape: {X_test.shape}', f'Y_test shape: {Y_test.shape}')
                        
            # folds of training and testing data for inner CV
            X_train_CV = self.cv_obj.folds[fold_idx][0]
            X_test_CV = self.cv_obj.folds[fold_idx][1]

            input_dim = X_train.shape[1]
            network_dict = self.cv_obj.networks
            train_network_dict = drop_test_network(self.cv_type, network_dict, X_test_CV, fold_idx+1)
            
            # Generate train and test indices from network dictionaries
            train_indices = np.concatenate([indices for indices in train_network_dict.values()])
            test_indices = network_dict[str(fold_idx+1)]

            if self.gpu_acceleration:
                X_train, Y_train, X_test, Y_test = map(cp.array, [X_train, Y_train, X_test, Y_test])

            # Inner CV on current training fold
            if search_method[0] == 'wandb':
                best_model, best_val_score = self.run_innercv_wandb(input_dim, X_train_CV, X_test_CV, train_network_dict, fold_idx, search_method=search_method)                
                train_history = best_model.fit(X_train, Y_train, X_test, Y_test)
            else:
                best_model, best_val_score = self.run_innercv(X_train_CV, X_test_CV, train_network_dict, search_method=search_method)
                best_model.fit(X_train, Y_train)
                train_history = None
            
            # Evaluate on the test fold                
            evaluator = ModelEvaluator(best_model, self.Y, train_indices, test_indices, self.network_labels, X_train, Y_train, X_test, Y_test, self.use_shared_regions, self.test_shared_regions)

            train_metrics = evaluator.get_train_metrics()
            test_metrics = evaluator.get_test_metrics()
            
            # Display final evaluation metrics
            print("\nTRAIN METRICS:", train_metrics)
            print("TEST METRICS:", test_metrics)
            print('BEST VAL SCORE', best_val_score)
            print('BEST MODEL HYPERPARAMS', best_model.get_params() if hasattr(best_model, 'get_params') else extract_model_params(best_model))

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
                    gene_list=self.gene_list,
                    binarize=self.binarize,
                    impute_strategy=self.impute_strategy,
                    sort_genes=self.sort_genes,
                    seed=self.random_seed
                )

            # Extract feature importances and model JSON
            feature_importances_, model_json = extract_feature_importances(
                self.model_type, 
                best_model, 
                self.save_model_json
            )

            # Save results to pickle file - consider removing this
            self.results.append({
                'model_parameters': best_model.get_params() if hasattr(best_model, 'get_params') else extract_model_params(best_model),
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