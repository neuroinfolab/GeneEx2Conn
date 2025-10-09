from env.imports import *

from data.data_load import load_transcriptome, load_cell_types, load_connectome, load_coords, load_network_labels, load_lobe_labels

from data.cv_split import (
    RandomCVSplit, 
    SchaeferCVSplit, 
    CommunityCVSplit, 
    SubnetworkCVSplit,
    SpatialCVSplit, 
    LobeCVSplit
)

from data.data_utils import (
    expand_X_symmetric,
    expand_Y_symmetric,
    RegionPairDataset
)

from models.base_models import ModelBuild, BaseModel, XGBRegressorModel
from models.feature_based import CGEModel, GaussianKernelModel, ExponentialDecayModel
from models.bilinear import BilinearLowRank, BilinearCM
from models.pls import PLSTwoStepModel, PLS_MLPDecoderModel, PLS_BilinearDecoderModel
from models.dynamic_mlp import DynamicMLP
from models.shared_encoder_models import SharedMLPEncoderModel, SharedLinearEncoderModel
from models.smt import SharedSelfAttentionModel, SharedSelfAttentionPoolingModel, SharedSelfAttentionCLSModel, SharedSelfAttentionCLSPoolingModel
from models.smt_advanced import SharedSelfAttentionPCAModel, SharedSelfAttentionPLSModel, SharedSelfAttentionConvModel
from models.smt_advanced import SharedSelfAttentionAEModel
from models.smt_advanced import SharedSelfAttentionCelltypeModel
from models.smt_advanced import SharedSelfAttentionGeneformerModel
from models.smt_advanced import SharedSelfAttentionGene2VecModel
from models.smt_cross import CrossAttentionGene2VecModel #, CrossAttentionModel

MODEL_CLASSES = {
    'cge': CGEModel,
    'gaussian_kernel': GaussianKernelModel,
    'exponential_decay': ExponentialDecayModel,
    'bilinear_lowrank': BilinearLowRank,
    'xgboost': XGBRegressorModel, 
    'bilinear_CM': BilinearCM,
    'pls_twostep': PLSTwoStepModel,
    'pls_mlpdecoder': PLS_MLPDecoderModel,
    'pls_bilineardecoder': PLS_BilinearDecoderModel,
    'dynamic_mlp': DynamicMLP,
    'shared_mlp_encoder': SharedMLPEncoderModel,
    'shared_linear_encoder': SharedLinearEncoderModel,
    'shared_transformer': SharedSelfAttentionModel,
    'shared_transformer_pool': SharedSelfAttentionPoolingModel,
    'shared_transformer_cls': SharedSelfAttentionCLSModel,
    'shared_transformer_cls_pool': SharedSelfAttentionCLSPoolingModel,
    'shared_transformer_conv': SharedSelfAttentionConvModel,
    'shared_transformer_pca': SharedSelfAttentionPCAModel,
    'shared_transformer_pls': SharedSelfAttentionPLSModel,
    'shared_transformer_ae': SharedSelfAttentionAEModel,
    'shared_transformer_celltype': SharedSelfAttentionCelltypeModel,
    'shared_transformer_geneformer': SharedSelfAttentionGeneformerModel,
    'shared_transformer_gene2vec': SharedSelfAttentionGene2VecModel,
    #'cross_attention': CrossAttentionModel,
    'cross_attention_gene2vec': CrossAttentionGene2VecModel
}

from models.metrics.eval import (
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
    find_best_params
    )

from sim.sim_utils import (
    train_sweep_torch,
    load_sweep_config, 
    load_best_parameters
)

absolute_root_path = '/scratch/asr655/neuroinformatics/GeneEx2Conn'


class Simulation:
    def __init__(self, feature_type, cv_type, model_type, gpu_acceleration, resolution=1.0, random_seed=42,
                 omit_subcortical=False, dataset='UKBB', parcellation='S456', impute_strategy='mirror_interpolate', sort_genes='expression', 
                 gene_list='0.2', hemisphere='both', train_shared_regions=False, test_shared_regions=False, 
                 connectome_target='FC', skip_cv=False, null_model=False, save_model=None):        
        """
        Initialization of simulation parameters
        """
        self.feature_type = feature_type
        self.cv_type = cv_type
        self.model_type = model_type
        self.gpu_acceleration = gpu_acceleration
        self.resolution = resolution
        self.random_seed = random_seed
        self.omit_subcortical, self.dataset, self.parcellation, self.impute_strategy, self.sort_genes, self.gene_list, self.hemisphere = omit_subcortical, dataset, parcellation, impute_strategy, sort_genes, gene_list, hemisphere
        self.train_shared_regions = train_shared_regions
        self.test_shared_regions = test_shared_regions
        self.connectome_target = connectome_target.upper()
        self.skip_cv = skip_cv
        self.null_model = null_model
        self.save_model = save_model
        self.results = []
    
    def load_data(self):
        """
        Load transcriptome and connectome data
        """
        self.X, self.valid_genes = load_transcriptome(parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, gene_list=self.gene_list, hemisphere=self.hemisphere, impute_strategy=self.impute_strategy, sort_genes=self.sort_genes, null_model=self.null_model, random_seed=self.random_seed, return_valid_genes=True)
        self.X_pca = load_transcriptome(parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, gene_list=self.gene_list, run_PCA='95var', hemisphere=self.hemisphere, null_model=self.null_model, random_seed=self.random_seed)
        self.X_pca_full = load_transcriptome(parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, gene_list=self.gene_list, run_PCA='full', hemisphere=self.hemisphere, null_model=self.null_model, random_seed=self.random_seed)
        self.X_cell_types_Jorstad = load_cell_types(parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, ref_dataset='Jorstad')
        self.X_cell_types_Lake_DFC = load_cell_types(parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, ref_dataset='LakeDFC')
        self.X_cell_types_Lake_VIS = load_cell_types(parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, ref_dataset='LakeVIS')
        self.Y_sc = load_connectome(parcellation=self.parcellation, dataset=self.dataset, omit_subcortical=self.omit_subcortical, measure='SC', spectral=None, hemisphere=self.hemisphere)
        self.Y_sc_binary = load_connectome(parcellation=self.parcellation, dataset=self.dataset, omit_subcortical=self.omit_subcortical, measure='SC', binarize=True, hemisphere=self.hemisphere)
        self.Y_sc_spectralL = load_connectome(parcellation=self.parcellation, dataset=self.dataset, omit_subcortical=self.omit_subcortical, measure='SC', spectral='L', hemisphere=self.hemisphere)
        self.Y_sc_spectralA = load_connectome(parcellation=self.parcellation, dataset=self.dataset, omit_subcortical=self.omit_subcortical, measure='SC', spectral='A', hemisphere=self.hemisphere)
        self.Y_fc = load_connectome(parcellation=self.parcellation, dataset=self.dataset, omit_subcortical=self.omit_subcortical, measure=self.connectome_target, hemisphere=self.hemisphere)
        self.Y_fc_binary = load_connectome(parcellation=self.parcellation, dataset=self.dataset, omit_subcortical=self.omit_subcortical, measure='FC', binarize=True, hemisphere=self.hemisphere)
        self.coords = load_coords(parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, hemisphere=self.hemisphere)
        self.labels, self.network_labels = load_network_labels(parcellation=self.parcellation, omit_subcortical=self.omit_subcortical, hemisphere=self.hemisphere)

        # Remove rows that are all NaN - necessary for gene expression data with unsampled regions
        valid_indices = ~np.isnan(self.X).all(axis=1)
        self.valid_indices = valid_indices
        # Create index map so we know original (i.e. true) indices of valid data after subsetting
        valid_indices_values = np.where(valid_indices)[0]
        valid2true_index_mapping = dict(enumerate(valid_indices_values))
        self.valid2true_index_mapping = valid2true_index_mapping
      
        # Subset all data using valid indices
        self.X = self.X[valid_indices]
        self.X_pca = self.X_pca[valid_indices]
        self.X_pca_full = self.X_pca_full[valid_indices]
        self.X_cell_types_Jorstad = self.X_cell_types_Jorstad[valid_indices]
        self.X_cell_types_Lake_DFC = self.X_cell_types_Lake_DFC[valid_indices]
        self.X_cell_types_Lake_VIS = self.X_cell_types_Lake_VIS[valid_indices]
        self.Y_sc = self.Y_sc[valid_indices][:, valid_indices]
        self.Y_sc_binary = self.Y_sc_binary[valid_indices][:, valid_indices]
        self.Y_sc_spectralL = self.Y_sc_spectralL[valid_indices]
        self.Y_sc_spectralA = self.Y_sc_spectralA[valid_indices]
        self.Y_fc = self.Y_fc[valid_indices][:, valid_indices]
        self.Y_fc_binary = self.Y_fc_binary[valid_indices][:, valid_indices]
        self.coords = self.coords[valid_indices]
        self.network_labels = self.network_labels[valid_indices]
        self.labels = [self.labels[i] for i in range(len(self.labels)) if valid_indices[i]]
        
        print(f"X shape: {self.X.shape}")
        print(f"X_pca shape: {self.X_pca.shape}")
        print(f"X_pca_full shape: {self.X_pca_full.shape}")
        print(f"X_cell_types_Jorstad shape: {self.X_cell_types_Jorstad.shape}") 
        print(f"X_cell_types_LakeDFC shape: {self.X_cell_types_Lake_DFC.shape}")
        print(f"X_cell_types_LakeVIS shape: {self.X_cell_types_Lake_VIS.shape}")
        print(f"Y_sc shape: {self.Y_sc.shape}")
        print(f"Y_sc_spectralL shape: {self.Y_sc_spectralL.shape}")
        print(f"Y_sc_spectralA shape: {self.Y_sc_spectralA.shape}")
        print(f"Y_fc shape: {self.Y_fc.shape}")
        print(f"Coordinates shape: {self.coords.shape}")
        
        # Define target connectome
        self.Y = self.Y_fc if 'FC' in self.connectome_target else self.Y_sc
        print('connectome target', self.connectome_target)
        print('Y shape', self.Y.shape)
    
    def select_cv(self):
        """
        Select cross-validation strategy
        """
        if self.cv_type == 'random':
            self.cv_obj = RandomCVSplit(self.X, self.Y, num_splits=4, shuffled=True, use_random_state=True, random_seed=self.random_seed)
        if self.cv_type == 'random_full':
            self.cv_obj = RandomCVSplit(self.X, self.Y, num_splits=100, shuffled=True, use_random_state=True, random_seed=self.random_seed)
        elif self.cv_type == 'schaefer':
            self.cv_obj = SchaeferCVSplit(self.X, self.Y, self.network_labels)
        elif self.cv_type == 'community':
            self.cv_obj = CommunityCVSplit(self.X, self.Y_fc, resolution=self.resolution, random_seed=self.random_seed) 
        elif self.cv_type == 'spatial':
            self.cv_obj = SpatialCVSplit(self.X, self.Y, self.coords, num_splits=4, random_seed=self.random_seed)
        elif self.cv_type == 'lobe':
            self.lobe_labels = load_lobe_labels(parcellation=self.parcellation, omit_subcortical=self.omit_subcortical)[self.valid_indices]
            self.cv_obj = LobeCVSplit(self.X, self.Y, self.lobe_labels)

    def expand_data_torch(self):
        """
        Expand data based on feature+processing type, and target
        """        
        # Create a list of features to be expanded into edge-wise dataset
        self.features = []
        for feature_dict in self.feature_type:
            for feature_name, processing_type in feature_dict.items():
                print(f'feature_name: {feature_name}, processing_type: {processing_type}')
                self.features.append(feature_name if processing_type is None else f"{feature_name}_{processing_type}")

        print('features', self.features)

        # Possible features types
        feature_dict = {'transcriptome': self.X,
                        'transcriptome_PCA_95var': self.X_pca,
                        'transcriptome_PCA_full': self.X_pca_full,
                        # cell type info 
                        'Jorstad': self.X_cell_types_Jorstad,
                        'LakeDFC': self.X_cell_types_Lake_DFC,
                        'LakeVIS': self.X_cell_types_Lake_VIS,
                        'structural':   self.Y_sc,
                        'structural_spectral_L': self.Y_sc_spectralL,
                        'structural_spectral_A': self.Y_sc_spectralA,
                        'functional': self.Y_fc,
                        'euclidean': self.coords, 
                        'structural_spatial_null': np.hstack((self.coords, self.Y_sc)), # cannot be combined with other feats
                        'transcriptome_spatial_autocorr_null': np.hstack((self.coords, self.Y_sc, self.X_pca, self.X)), # cannot be combined with other feats
                        }
    
        X_all = []
        for feature in self.features:
            if 'spectral' in feature:
                feature_type = '_'.join(feature.split('_')[:-1])  # subset provided number of spectral components
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
            X_all.append(feature_X)
        self.X_all = np.hstack(X_all)
        print('Feature matrix, X, generated... expanding to pairwise dataset')

        # Create custom dataset for region pair data
        self.region_pair_dataset = RegionPairDataset(
            self.X_all,
            self.Y,
            self.coords,
            self.valid2true_index_mapping, 
            self.dataset,
            self.parcellation,
            self.valid_genes
        )
    
    def run_innercv_wandb_torch(self, input_dim, train_indices, test_indices, train_network_dict, outer_fold_idx, search_method=('random', 'mse', 3), sweep_id=None):
        """
        Inner cross-validation with W&B support for torch models
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        inner_cv_obj = SubnetworkCVSplit(train_indices, train_network_dict)

        if self.skip_cv:
            sweep_config_path = os.path.join(absolute_root_path, 'models', 'configs', f'{self.model_type}_sweep_config.yml')
            best_config = load_best_parameters(sweep_config_path, input_dim=input_dim)
            best_val_loss = 0.0 # no CV --> no best val loss
        else:
            def train_sweep_wrapper(config=None):
                return train_sweep_torch(
                    config=config,
                    train_indices=train_indices,
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
                    impute_strategy=self.impute_strategy,
                    sort_genes=self.sort_genes,
                    null_model=self.null_model
                )
            
            # Run sweep
            wandb.agent(sweep_id, function=train_sweep_wrapper, count=search_method[2])

            # Get best run from sweep
            api = wandb.Api()
            sweep = api.sweep(f"alexander-ratzan-new-york-university/gx2conn/{sweep_id}")
            best_run = sweep.best_run()
            wandb.teardown()

            best_val_loss = best_run.summary.mean_val_loss # can change to mean_val_pearson
            best_config = best_run.config
        
        print('BEST CONFIG', best_config)

        # Initialize final model with best config
        ModelClass = MODEL_CLASSES[self.model_type]
        if 'pls' in self.model_type or 'pca' in self.model_type:
            best_model = ModelClass(**best_config, train_indices=train_indices, test_indices=test_indices, region_pair_dataset=self.region_pair_dataset).to(device)   
        elif 'celltype' in self.model_type or 'gene2vec' in self.model_type:
            best_model = ModelClass(**best_config, region_pair_dataset=self.region_pair_dataset).to(device)
        else:
            best_model = ModelClass(**best_config).to(device)
            
        return best_model, best_val_loss, best_config, sweep_id

    def run_sim_torch(self, search_method=('random', 'mse', 5), track_wandb=False, use_folds=[0, 1, 2, 3, 4, 5, 6, 7]):
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
            if fold_idx in use_folds: # use_folds set to 0-7 to accomodate Schaefer CV
                # Initialize sweep for each fold
                sweep_id = None
                if (search_method[0] == 'wandb' or track_wandb):
                    input_dim = self.region_pair_dataset.X_expanded[0].shape[0]
                    sweep_config_path = os.path.join(absolute_root_path, 'models', 'configs', f'{self.model_type}_sweep_config.yml')
                    sweep_config = load_sweep_config(sweep_config_path, input_dim=input_dim)
                    sweep_id = wandb.sweep(sweep=sweep_config, project="gx2conn")
                    print('Initialized sweep with ID:', sweep_id)
                
                train_region_pairs = expand_X_symmetric(train_indices.reshape(-1, 1)).astype(int)
                test_region_pairs = expand_X_symmetric(test_indices.reshape(-1, 1)).astype(int)
                
                train_indices_expanded = np.array([self.region_pair_dataset.valid_pair_to_expanded_idx[tuple(pair)] for pair in train_region_pairs])
                test_indices_expanded = np.array([self.region_pair_dataset.valid_pair_to_expanded_idx[tuple(pair)] for pair in test_region_pairs])    
                
                innercv_network_dict = drop_test_network(self.cv_type, network_dict, test_indices, fold_idx+1)
                input_dim = self.region_pair_dataset.X_expanded[0].shape[0]
                best_model, best_val_score, best_config, _ = self.run_innercv_wandb_torch(input_dim, train_indices, test_indices, innercv_network_dict, fold_idx, search_method, sweep_id)
                
                if track_wandb:
                    feature_str = "+".join(str(k) if v is None else f"{k}_{v}" 
                                    for feat in self.feature_type
                                    for k,v in feat.items())
                    run_name = f"{self.model_type}_{feature_str}_{self.connectome_target}_{self.cv_type}{self.random_seed}_fold{fold_idx}_final_eval"
                    final_eval_run = wandb.init(project="gx2conn",
                                                name=run_name,
                                                group=f"sweep_{sweep_id}" if sweep_id else None,
                                                config=best_config,
                                                tags=["final_eval", 
                                                    f"dataset_{self.dataset}",
                                                    f"cv_type_{self.cv_type}", 
                                                    f"outerfold_{fold_idx}",
                                                    f"model_{self.model_type}",
                                                    f"split_{self.cv_type}{self.random_seed}", 
                                                    f"feature_type_{feature_str}",
                                                    f"target_{self.connectome_target}",
                                                    f"parcellation_{self.parcellation}",
                                                    f"hemisphere_{self.hemisphere}",
                                                    f"omit_subcortical_{self.omit_subcortical}",
                                                    f"gene_list_{self.gene_list}",
                                                    f"impute_strategy_{self.impute_strategy}",
                                                    f"sort_genes_{self.sort_genes}",
                                                    f"null_model_{self.null_model}"],
                                                reinit=True)

                    if self.model_type in MODEL_CLASSES:
                        wandb.watch(best_model, log='all')
                        if self.model_type == 'pls_twostep':
                            train_history = best_model.fit(self.region_pair_dataset, train_indices, test_indices)
                        else:
                            train_history = best_model.fit(self.region_pair_dataset, train_indices_expanded, test_indices_expanded, save_model=self.save_model)
                        for epoch, (train_loss, val_loss) in enumerate(zip(train_history['train_loss'], train_history['val_loss'])):
                            wandb.log({'train_mse_loss': train_loss, 'test_mse_loss': val_loss})
                else:
                    if self.model_type == 'pls_twostep':
                        train_history = best_model.fit(self.region_pair_dataset, train_indices, test_indices)
                    else: 
                        train_history = best_model.fit(self.region_pair_dataset, train_indices_expanded, test_indices_expanded, save_model=self.save_model)
                
                # Evaluate on the test fold
                train_dataset = Subset(self.region_pair_dataset, train_indices_expanded)
                test_dataset = Subset(self.region_pair_dataset, test_indices_expanded)
                
                if self.train_shared_regions or self.test_shared_regions:
                    # get shared indices between train and test
                    # Get interconnections between train and test sets using numpy meshgrid
                    train_idx_grid, test_idx_grid = np.meshgrid(train_indices, test_indices)
                    train_test_pairs = np.column_stack((train_idx_grid.ravel(), test_idx_grid.ravel()))
                    
                    # Add reverse direction pairs
                    train_test_pairs = np.vstack((train_test_pairs, train_test_pairs[:, ::-1]))
                    train_test_indices_expanded = np.array([self.region_pair_dataset.valid_pair_to_expanded_idx[tuple(pair)] for pair in train_test_pairs])

                    if self.train_shared_regions: # update train dataset 
                        train_indices_expanded = np.concatenate((train_indices_expanded, train_test_indices_expanded)).astype(train_indices_expanded.dtype)
                        # train_indices = sorted(list(set(train_indices).union(set(test_indices))))
                        train_dataset = Subset(self.region_pair_dataset, train_indices_expanded) # viz not yet implemented for this path
                    elif self.test_shared_regions: # update test dataset
                        test_indices_expanded = np.concatenate((test_indices_expanded, train_test_indices_expanded)).astype(test_indices_expanded.dtype)
                        test_dataset = Subset(self.region_pair_dataset, test_indices_expanded)
                
                train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, pin_memory=True)
                test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, pin_memory=True)
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
                                                train_shared_regions=self.train_shared_regions,
                                                test_shared_regions=self.test_shared_regions)

                train_metrics = evaluator.get_train_metrics()
                test_metrics = evaluator.get_test_metrics()
                print('BEST VAL SCORE', best_val_score)
                print('BEST MODEL HYPERPARAMS', best_model.get_params() if hasattr(best_model, 'get_params') else extract_model_params(best_model))

                if track_wandb:
                    wandb.log({'final_train_metrics': train_metrics, 
                            'final_test_metrics': test_metrics,
                            'best_val_loss': best_val_score})
                    final_eval_run.finish()
                    wandb.finish()
                    print("Final evaluation metrics logged successfully.")
            
            #torch._dynamo.reset()
            torch.cuda.empty_cache()
            gc.collect()
            
        print_system_usage() # Display CPU and RAM utilization
        if self.gpu_acceleration: GPUtil.showUtilization() # Display GPU utilization
        time.sleep(5)
        print("Sim complete")