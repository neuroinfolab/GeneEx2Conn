from env.imports import *

import importlib
import data
import data.data_utils
from data.data_load import load_transcriptome, load_connectome, load_coords, load_network_labels
from models import *
from data import * 
from sim import *
import models
import models.metrics
from models.metrics import *

### SPATIAL AUTOCORRELATION ###
def load_and_plot_data(parcellation='S400', hemisphere='both', omit_subcortical=False, 
                      sort_genes='expression', impute_strategy='mirror_interpolate', 
                      null_model='none', fontsize=16, subset_indices=None):
    """
    Load gene expression, connectivity and coordinate data and create visualization plots.
    
    Args:
        parcellation (str): Brain parcellation scheme (default: 'S400')
        hemisphere (str): Which hemisphere to use - 'left', 'right' or 'both' (default: 'both')
        omit_subcortical (bool): Whether to exclude subcortical regions (default: False)
        sort_genes (str): Method for sorting genes - 'expression' or 'chromosome' (default: 'expression')
        impute_strategy (str): Strategy for imputing missing values (default: 'mirror_interpolate')
        null_model (str): Type of null model to use (default: 'none')
        fontsize (int): Font size for plot labels (default: 16)
        subset_indices (array-like): Optional indices to subset the data (default: None)
        
    Returns:
        tuple: Contains the following arrays:
            - X: Raw gene expression data
            - X_pca: PCA-transformed gene expression data  
            - Y: Connectivity matrix
            - coords: Region coordinates
            - labels: Region labels
            - network_labels: Network assignment labels
            - X_corr: Gene expression correlation matrix
            - X_pca_corr: PCA gene expression correlation matrix
            - dist_matrix: Euclidean distance matrix between regions
    """
    
    # Load data
    X = load_transcriptome(parcellation=parcellation, hemisphere=hemisphere, 
                          omit_subcortical=omit_subcortical, sort_genes=sort_genes,
                          impute_strategy=impute_strategy, null_model=null_model)
    Y = load_connectome(parcellation=parcellation, hemisphere=hemisphere,
                       omit_subcortical=omit_subcortical)
    coords = load_coords(parcellation=parcellation, hemisphere=hemisphere,
                        omit_subcortical=omit_subcortical)
    labels, network_labels = load_network_labels(parcellation=parcellation,
                                               hemisphere=hemisphere,
                                               omit_subcortical=omit_subcortical)

    # Filter valid data
    valid_indices = ~np.isnan(X).all(axis=1)
    X = X[valid_indices]

    X_pca = load_transcriptome(run_PCA=True, parcellation=parcellation,
                              hemisphere=hemisphere, omit_subcortical=omit_subcortical,
                              sort_genes=sort_genes, impute_strategy=impute_strategy, null_model=null_model)
    X_pca = X_pca[valid_indices]
    Y = Y[valid_indices][:, valid_indices]

    coords = coords[valid_indices]
    labels = [labels[i] for i in range(len(labels)) if valid_indices[i]]
    network_labels = network_labels[valid_indices]

    # Apply subset if provided
    if subset_indices is not None:
        X = X[subset_indices]
        X_pca = X_pca[subset_indices]
        Y = Y[subset_indices][:, subset_indices]
        coords = coords[subset_indices]
        labels = [labels[i] for i in subset_indices]
        network_labels = network_labels[subset_indices]

    # Compute correlation matrices
    X_corr = np.corrcoef(X, rowvar=True)
    X_pca_corr = np.corrcoef(X_pca, rowvar=True)

    # Compute distance matrix
    dist_matrix = cdist(coords, coords)

    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(40, 8))

    # Plot X correlations
    im0 = axes[0].imshow(X_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title('Gene Expression Correlations', fontsize=fontsize)
    plt.colorbar(im0, ax=axes[0])
    axes[0].tick_params(axis='both', which='major', labelsize=fontsize-2)

    # Plot X PCA correlations
    im1 = axes[1].imshow(X_pca_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title('PCA Gene Expression Correlations', fontsize=fontsize)
    plt.colorbar(im1, ax=axes[1])
    axes[1].tick_params(axis='both', which='major', labelsize=fontsize-2)

    # Plot distance matrix
    im2 = axes[2].imshow(dist_matrix, cmap='viridis')
    axes[2].set_title('Euclidean Distances', fontsize=fontsize)
    plt.colorbar(im2, ax=axes[2])
    axes[2].tick_params(axis='both', which='major', labelsize=fontsize-2)

    # Plot Y matrix
    im3 = axes[3].imshow(Y, cmap='RdBu_r', vmin=-0.8, vmax=0.8)
    axes[3].set_title('Connectivity', fontsize=fontsize)
    plt.colorbar(im3, ax=axes[3])
    axes[3].tick_params(axis='both', which='major', labelsize=fontsize-2)

    plt.tight_layout()
    plt.show()

    return X, X_pca, Y, coords, labels, network_labels, X_corr, X_pca_corr, dist_matrix

# Define exponential decay function
def exp_decay(x, SA_inf, SA_lambda):
    """    
    Parameters
    ----------
    x : array-like
        Distance values (in mm)
    SA_inf : float
        Asymptotic value of functional connectivity at infinite distance
    SA_lambda : float
        Decay constant controlling rate of exponential decay
        
    Returns
    -------
    float or array-like
        Predicted values following exponential decay
    """
    return SA_inf + (1 - SA_inf) * np.exp(-x / SA_lambda)

def plot_distance_decay(features, y_feature='FC', bin_size_mm=5, coverage='Full Brain'):
    """Plot distance decay relationship between a feature and distance.
    
    Parameters
    ----------
    features : dict
        Dictionary containing 'distances' and feature arrays (e.g., 'FC', 'CGE', 'PCA_CGE')
    y_feature : str, optional
        Name of feature to plot on y-axis, by default 'FC'
    bin_size_mm : float, optional
        Size of distance bins in mm, by default 5
    coverage : str, optional
        Region of brain being analyzed ('Full Brain', 'Cortex', or 'Subcortex'), by default 'Full Brain'
    """
    # Create distance bins
    bin_edges = np.arange(0, max(features['distances']) + bin_size_mm, bin_size_mm)

    # Calculate binned statistics
    bin_means, bin_edges, _ = stats.binned_statistic(features['distances'], features[y_feature], 
                                             statistic='mean', bins=bin_edges)
    bin_std, _, _ = stats.binned_statistic(features['distances'], features[y_feature], 
                                   statistic='std', bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit exponential decay
    popt, _ = curve_fit(exp_decay, bin_centers, bin_means,
                      p0=[0, 10], bounds=([-1, 0], [1, 100]), maxfev=5000)

    # Generate fit curve
    x_fit = np.linspace(min(features['distances']), max(features['distances']), 100)
    y_fit = exp_decay(x_fit, *popt)

    # Create plot
    plt.figure(figsize=(12,8))

    # Get y-axis label based on feature
    if y_feature == 'FC':
        y_label = 'Functional Connectivity (FC)'
    elif y_feature == 'CGE':
        y_label = 'Correlated Gene Expression (CGE)'
    elif y_feature == 'PCA_CGE':
        y_label = 'PCA CGE'
    else:
        y_label = y_feature.replace('_', ' ').title()

    plt.bar(bin_centers, bin_means, width=bin_size_mm, color='#808080', alpha=0.6)
    plt.scatter(features['distances'], features[y_feature], color='gray', 
               alpha=0.1, s=0.8)
    plt.errorbar(bin_centers, bin_means, yerr=bin_std, color='black', fmt='o',
                markersize=2, capsize=1, linewidth=0.5,
                label=f"Mean {y_label} ({bin_size_mm}mm bins)")
    plt.plot(x_fit, y_fit, "r-", linewidth=2.5, label="Exp Decay Fit")
    plt.axvline(x=popt[1], color='green', linestyle='--', alpha=0.2)

    # Add SA parameters text
    sa_text = f'SA-λ = {popt[1]:.3f}\nSA-∞ = {popt[0]:.3f}'
    plt.text(0.95, 0.75, sa_text, transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            horizontalalignment='right', verticalalignment='bottom',
            fontsize=18)

    plt.xlabel('Distance (mm)', fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.title(f'Distance vs {y_label} ({coverage})', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(fontsize=16)

    plt.tight_layout()
    plt.show()


def plot_distance_decay_poly3(features, y_feature='FC', bin_size_mm=5, coverage='Full Brain'):
    """Plot distance decay using a 3rd order polynomial fit.

    Parameters
    ----------
    features : dict
        Dictionary containing 'distances' and feature arrays (e.g., 'FC', 'CGE', 'PCA_CGE')
    y_feature : str, optional
        Name of feature to plot on y-axis, by default 'FC'
    bin_size_mm : float, optional
        Size of distance bins in mm, by default 5
    coverage : str, optional
        Region of brain being analyzed ('Full Brain', 'Cortex', or 'Subcortex'), by default 'Full Brain'
    """
    # Create distance bins
    bin_edges = np.arange(0, max(features['distances']) + bin_size_mm, bin_size_mm)

    # Calculate binned statistics
    bin_means, bin_edges, _ = stats.binned_statistic(features['distances'], features[y_feature], 
                                                     statistic='mean', bins=bin_edges)
    bin_std, _, _ = stats.binned_statistic(features['distances'], features[y_feature], 
                                           statistic='std', bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Remove NaNs (may occur if a bin is empty)
    valid = ~np.isnan(bin_means)
    bin_centers = bin_centers[valid]
    bin_means = bin_means[valid]
    bin_std = bin_std[valid]

    # Fit a 3rd order polynomial
    poly3 = Polynomial.fit(bin_centers, bin_means, deg=3)
    coefs = poly3.convert().coef  # Convert to standard basis to extract raw coefficients

    x_fit = np.linspace(min(features['distances']), max(features['distances']), 100)
    y_fit = poly3(x_fit)

    # Create plot
    plt.figure(figsize=(12, 8))

    if y_feature == 'FC':
        y_label = 'Functional Connectivity (FC)'
    elif y_feature == 'CGE':
        y_label = 'Correlated Gene Expression (CGE)'
    elif y_feature == 'PCA_CGE':
        y_label = 'PCA CGE'
    else:
        y_label = y_feature.replace('_', ' ').title()

    plt.bar(bin_centers, bin_means, width=bin_size_mm, color='#808080', alpha=0.6)
    plt.scatter(features['distances'], features[y_feature], color='gray', alpha=0.1, s=0.8)
    plt.errorbar(bin_centers, bin_means, yerr=bin_std, color='black', fmt='o',
                 markersize=2, capsize=1, linewidth=0.5,
                 label=f"Mean {y_label} ({bin_size_mm}mm bins)")
    plt.plot(x_fit, y_fit, "darkorange", linewidth=2.5, label="3rd Order Polynomial Fit")

    # Add polynomial coefficients to plot
    coef_text = '\n'.join([
        f"$a_3$ = {coefs[3]:.4e}",
        f"$a_2$ = {coefs[2]:.4e}",
        f"$a_1$ = {coefs[1]:.4e}",
        f"$a_0$ = {coefs[0]:.4e}",
    ])
    plt.text(0.05, 0.05, coef_text, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             horizontalalignment='left', verticalalignment='bottom',
             fontsize=14)

    plt.xlabel('Distance (mm)', fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.title(f'Distance vs {y_label} ({coverage})', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(fontsize=16)

    plt.tight_layout()
    plt.show()

def plot_distance_fits(features, y_feature, bin_size_mm=10, coverage='Full Brain', include_linear=False, fontsize=28):
    """
    Plot distance relationships with exponential decay, polynomial and linear fits
    
    Args:
        features (dict): Dictionary containing distances and feature values
        y_feature (str): Feature to plot ('FC', 'CGE', or 'PCA_CGE')
        bin_size_mm (float): Size of distance bins in mm
        coverage (str): Brain coverage description for title
        include_linear (bool): Whether to include linear fit (default False)
        fontsize (int): Base font size for plot text elements (default 18)
    """
    # Calculate binned statistics
    bins = np.arange(0, max(features['distances']) + bin_size_mm, bin_size_mm)
    bin_centers = bins[:-1] + bin_size_mm/2
    bin_means, _, _ = stats.binned_statistic(features['distances'], 
                                           features[y_feature], 
                                           statistic='mean', 
                                           bins=bins)
    bin_std, _, _ = stats.binned_statistic(features['distances'], 
                                          features[y_feature], 
                                          statistic='std', 
                                          bins=bins)
    valid = ~np.isnan(bin_means)
    bin_centers = bin_centers[valid]
    bin_means = bin_means[valid]
    bin_std = bin_std[valid]

    # Fit models
    # 3rd order polynomial
    poly3 = Polynomial.fit(bin_centers, bin_means, deg=3)
    poly_coefs = poly3.convert().coef
    # Exponential decay
    exp_params, _ = curve_fit(exp_decay, bin_centers, bin_means,
                           p0=[0, 10], bounds=([-1, 0], [1, 100]), maxfev=5000)
    
    # Linear fit if requested
    if include_linear:
        linear = np.polyfit(bin_centers, bin_means, 1)
    
    # Generate fit lines
    x_fit = np.linspace(min(features['distances']), max(features['distances']), 100)
    y_poly = poly3(x_fit)
    y_exp = exp_decay(x_fit, *exp_params)
    if include_linear:
        y_linear = linear[0] * x_fit + linear[1]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    # Set y-axis label based on feature type
    if y_feature == 'FC':
        y_label = 'Functional Connectivity (FC)'
    elif y_feature == 'CGE':
        y_label = 'Correlated Gene Expression (CGE)'
    elif y_feature == 'PCA_CGE':
        y_label = 'PCA CGE'
    else:
        y_label = y_feature.replace('_', ' ').title()

    # Plot raw data and binned statistics
    plt.bar(bin_centers, bin_means, width=bin_size_mm, color='#808080', alpha=0.5)
    plt.scatter(features['distances'], features[y_feature], color='gray', alpha=0.1, s=1)
    # plt.errorbar(bin_centers, bin_means, yerr=bin_std, color='black', fmt='o',
    #              markersize=2, capsize=2, linewidth=0.5,
    #              label=f"Mean {y_feature} ({bin_size_mm}mm bins)")

    # Plot fits
    plt.plot(x_fit, y_poly, "darkorange", linewidth=3, label="3rd Order Polynomial")
    plt.plot(x_fit, y_exp, "red", linewidth=3, label="Exponential Decay")
    if include_linear:
        plt.plot(x_fit, y_linear, "blue", linewidth=4, label="Linear Fit")

    plt.xlabel('Distance (mm)', fontsize=fontsize-2)
    plt.ylabel(y_label, fontsize=fontsize-2)
    # plt.title(f'Distance vs {y_label} ({coverage})', fontsize=fontsize+2)
    plt.tick_params(axis='both', which='major', labelsize=fontsize-2)
    # plt.legend(fontsize=fontsize-4, loc='upper right')

    # Add fit parameters text box in lower right corner
    param_text = (
        "Exponential:\n"
        f"SA-∞ = {exp_params[0]:.3f}\n"
        f"SA-λ = {exp_params[1]:.3f}\n\n"
        "Polynomial:\n"
        f"$a_3$ = {poly_coefs[3]:.2e}\n"
        f"$a_2$ = {poly_coefs[2]:.2e}\n"
        f"$a_1$ = {poly_coefs[1]:.2e}\n"
        f"$a_0$ = {poly_coefs[0]:.2e}"
    )
    
    if include_linear:
        param_text += f"\n\nLinear:\nslope = {linear[0]:.2e}\nintercept = {linear[1]:.2e}"
    
    plt.text(0.78, 0.02, param_text, fontsize=fontsize-9,
             bbox=dict(facecolor='white', alpha=0.8),
             transform=ax.transAxes,
             verticalalignment='bottom',
             horizontalalignment='left')

    plt.tight_layout()
    plt.show()

def subset_brain_data(X, X_pca, Y, coords, labels, network_labels, X_corr, X_pca_corr, dist_matrix):
    """
    Subset brain data into full brain, cortical, and subcortical components.
    
    Args:
        X: Gene expression matrix (regions x genes)
        X_pca: PCA-transformed gene expression matrix (regions x PCs)
        Y: Connectivity matrix (regions x regions)
        coords: Region coordinates (regions x 3)
        labels: Region labels (list of strings)
        network_labels: Network assignment labels (regions)
        X_corr: Gene expression correlation matrix (regions x regions)
        X_pca_corr: PCA gene expression correlation matrix (regions x regions)
        dist_matrix: Euclidean distance matrix between regions (regions x regions)
        
    Returns:
        tuple: Contains three feature dictionaries with distances, CGE, FC and PCA_CGE:
            - features: Full brain features
            - features_cortex: Cortical features only
            - features_subcortex: Subcortical features only
    """
    # subset all to valid indices
    valid_indices = ~np.isnan(X).all(axis=1)
    X = X[valid_indices]
    X_pca = X_pca[valid_indices]
    Y = Y[valid_indices][:, valid_indices]
    coords = coords[valid_indices]
    labels = [labels[i] for i in range(len(labels)) if valid_indices[i]]
    network_labels = network_labels[valid_indices]

    # Get upper triangular indices (excluding diagonal)
    triu_indices = np.triu_indices(X_corr.shape[0], k=1)
    X_corr_vec = X_corr[triu_indices]
    X_pca_vec = X_pca_corr[triu_indices]
    dist_vec = dist_matrix[triu_indices] 
    Y_vec = Y[triu_indices]

    # Create features dictionary with distances and fc values
    features = {
        'distances': dist_vec,
        'CGE': X_corr_vec,
        'FC': Y_vec,
        'PCA_CGE': X_pca_vec
    }
    # Separate cortical data
    n = (X.shape[0] // 100) * 100  # Hacky way to retain only cortical regions by rounding down to nearest hundred
    X_cortex = X[:n]
    X_cortex_pca = X_pca[:n] 
    Y_cortex = Y[:n, :n]
    coords_cortex = coords[:n]
    labels_cortex = labels[:n]
    network_labels_cortex = network_labels[:n]

    # Calculate cortical correlations
    X_cortex_corr = np.corrcoef(X_cortex, rowvar=True)
    X_cortex_pca_corr = np.corrcoef(X_cortex_pca, rowvar=True)

    # compute distance matrix of coords
    dist_cortex_matrix = cdist(coords_cortex, coords_cortex)

    # Get upper triangular indices for cortex (excluding diagonal)
    triu_indices_cortex = np.triu_indices(X_cortex_corr.shape[0], k=1)
    X_cortex_corr_vec = X_cortex_corr[triu_indices_cortex]
    X_cortex_pca_corr_vec = X_cortex_pca_corr[triu_indices_cortex]
    dist_vec_cortex = dist_cortex_matrix[triu_indices_cortex]
    Y_cortex_vec = Y_cortex[triu_indices_cortex]

    # Create features dictionary for cortex
    features_cortex = {
        'distances': dist_vec_cortex,
        'CGE': X_cortex_corr_vec, 
        'FC': Y_cortex_vec,
        'PCA_CGE': X_cortex_pca_corr_vec
    }
    # Separate subcortical data
    X_subcortex = X[n:]
    X_subcortex_pca = X_pca[n:]
    Y_subcortex = Y[n:, n:]
    coords_subcortex = coords[n:]
    labels_subcortex = labels[n:]
    network_labels_subcortex = network_labels[n:]

    # Calculate subcortical correlations
    X_subcortex_corr = np.corrcoef(X_subcortex, rowvar=True)
    X_subcortex_pca_corr = np.corrcoef(X_subcortex_pca, rowvar=True)

    # compute distance matrix of coords
    dist_subcortex_matrix = cdist(coords_subcortex, coords_subcortex)

    # Get upper triangular indices for subcortex (excluding diagonal)
    triu_indices_subcortex = np.triu_indices(X_subcortex_corr.shape[0], k=1)
    X_subcortex_corr_vec = X_subcortex_corr[triu_indices_subcortex]
    X_subcortex_pca_corr_vec = X_subcortex_pca_corr[triu_indices_subcortex]
    dist_vec_subcortex = dist_subcortex_matrix[triu_indices_subcortex]
    Y_subcortex_vec = Y_subcortex[triu_indices_subcortex]

    # Create features dictionary for subcortex
    features_subcortex = {
        'distances': dist_vec_subcortex,
        'CGE': X_subcortex_corr_vec, 
        'FC': Y_subcortex_vec,
        'PCA_CGE': X_subcortex_pca_corr_vec
    }
    
    return features, features_cortex, features_subcortex

# Lightweight form SA functions without plotting
def load_features(X, coords, verbose=False):
    if verbose:
        start_time = time.time()

    # Filter valid data
    valid_indices = ~np.isnan(X).all(axis=1)    
    X = X[valid_indices]
    coords = coords[valid_indices]
    
    # Compute correlation matrices
    X_corr = np.corrcoef(X, rowvar=True)
    triu_indices = np.triu_indices(X_corr.shape[0], k=1)  # k=1 excludes diagonal
    X_corr_vec = X_corr[triu_indices]
    
    # Compute distance matrix
    dist_matrix = cdist(coords, coords)
    dist_vec = dist_matrix[triu_indices]
    
    if verbose:
        elapsed_time = time.time() - start_time
        print(f"Feature computation took {elapsed_time:.2f} seconds")

    return X_corr, X_corr_vec, dist_matrix, dist_vec

def compute_exponential_distance_decay(features, targets, bin_size_mm=5):
    # Create distance bins
    bin_edges = np.arange(0, max(features) + bin_size_mm, bin_size_mm)

    # Calculate binned statistics
    bin_means, bin_edges, _ = stats.binned_statistic(features, targets, 
                                             statistic='mean', bins=bin_edges)
    bin_std, _, _ = stats.binned_statistic(features, targets, 
                                   statistic='std', bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit exponential decay
    popt, _ = curve_fit(exp_decay, bin_centers, bin_means,
                      p0=[0, 10], bounds=([-1, 0], [1, 100]), maxfev=5000)

    # Get best fit parameters
    SA_inf, SA_lambda = popt

    return SA_lambda, SA_inf

def compute_distance_decay_poly3(features, targets, bin_size_mm=5):
    
    # Create distance bins
    bin_edges = np.arange(0, max(features) + bin_size_mm, bin_size_mm)

    # Calculate binned statistics
    bin_means, bin_edges, _ = stats.binned_statistic(features, targets, 
                                                     statistic='mean', bins=bin_edges)
    bin_std, _, _ = stats.binned_statistic(features, targets, 
                                           statistic='std', bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Remove NaNs (may occur if a bin is empty)
    valid = ~np.isnan(bin_means)
    bin_centers = bin_centers[valid]
    bin_means = bin_means[valid]
    bin_std = bin_std[valid]

    # Fit a 3rd order polynomial
    poly3 = Polynomial.fit(bin_centers, bin_means, deg=3)
    coefs = poly3.convert().coef  # Convert to standard basis to extract raw coefficients

    return coefs

### NULL SPIN TEST FUNCTIONS ###
def generate_null_spins(n_rotations=100, seed=42, bin_size_mm=5, save_csv=False):
    """Generate null spin models and compute distance decay parameters for both raw gene expression and PCA-transformed data.
    
    This function performs spatial rotations of brain regions to create null models while preserving spatial relationships.
    It handles cortical and subcortical regions separately, computes various distance decay parameters including exponential
    decay and polynomial fits, and standardizes the results relative to the true brain data.
    
    Args:
        n_rotations (int): Number of spin rotations to generate
        seed (int): Random seed for reproducibility 
        bin_size_mm (float): Bin size in mm for distance decay calculations
        save_csv (bool): Whether to save results to CSV
        
    Returns:
        pd.DataFrame: DataFrame containing:
            - Spin indices for cortical and subcortical regions
            - Cost metrics for the spin rotations
            - Standardized distance decay parameters (exponential and polynomial)
            - Error metrics comparing null models to true brain data
    """
    # LOAD IN TRUE GENE EXPRESSION DATA, GENE EXPRESSION PCA, TRUE COORDINATES
    genes_data, valid_genes = load_transcriptome(parcellation='S400', gene_list='0.2', run_PCA=False, omit_subcortical=False, hemisphere='both', impute_strategy='mirror_interpolate', sort_genes='expression', return_valid_genes=True, null_model='none', random_seed=42)
    genes_data_PCA = load_transcriptome(parcellation='S400', gene_list='0.2', run_PCA=True, omit_subcortical=False, hemisphere='both', impute_strategy='mirror_interpolate', sort_genes='expression', return_valid_genes=False, null_model='none', random_seed=42)
    all_region_coords = load_coords(parcellation='S400', omit_subcortical=False)

    # GENERATE SPINS FOR EACH HEMISPHERE OF CORTEX MATCHED
    lh_annot, rh_annot = nndata.fetch_schaefer2018('fsaverage', data_dir='data/UKBB', verbose=1)['400Parcels7Networks']
    coords, hemi = nnsurf.find_parcel_centroids(lhannot=lh_annot, rhannot=rh_annot, version='fsaverage', surf='sphere')
    cortical_spins, cortical_cost = nnstats.gen_spinsamples(coords, hemi, n_rotate=n_rotations, seed=seed, method='vasa',return_cost=True)

    num_cortical_regions = (genes_data.shape[0] // 100) * 100
    cortical_genes_data = genes_data[:num_cortical_regions]
    cortical_genes_data_PCA = genes_data_PCA[:num_cortical_regions]

    # GENERATE SPINS FOR EACH HEMISPHERE OF SUBCORTEX MATCHED
    subcortical_genes_data = genes_data[num_cortical_regions:]
    subcortical_genes_data_PCA = genes_data_PCA[num_cortical_regions:]
    subcortical_coords = all_region_coords[num_cortical_regions:]
    x_coords = subcortical_coords[:, 0]
    hemi_subcort = np.zeros_like(x_coords) # create mask for subcortical hemisphere
    hemi_subcort[x_coords > 0] = 1
    subcortical_spins, subcortical_cost = nnstats.gen_spinsamples(subcortical_coords, hemi_subcort, n_rotate=n_rotations, seed=seed, method='vasa', return_cost=True)

    # SAVE INDICES OF SPINS TO UNIFIED DATAFRAME
    cortical_spin_indices_T = cortical_spins.T
    subcortical_spin_indices_T = subcortical_spins.T
    spins_df = pd.DataFrame({
        'cortical_spins': cortical_spin_indices_T.tolist(),
        'subcortical_spins': subcortical_spin_indices_T.tolist()
    })

    # SAVE COST OF SPINS
    cortical_cost_T = cortical_cost.T
    subcortical_cost_T = subcortical_cost.T
    cortical_cost_sum = cortical_cost_T.sum(axis=1)
    subcortical_cost_sum = subcortical_cost_T.sum(axis=1)
    total_cost = subcortical_cost_sum + cortical_cost_sum
    spins_df['cortical_cost'] = cortical_cost_sum
    spins_df['subcortical_cost'] = subcortical_cost_sum 
    spins_df['total_cost'] = total_cost

    # INITIALIZE ARRAYS TO STORE SA EXPONENTIAL DECAY AND POLYNOMIAL COEFFICIENTS
    sa_poly_params = np.zeros((len(spins_df), 6))
    sa_poly_params_PCA = np.zeros((len(spins_df), 6))

    for i in range(len(spins_df)): # loop takes ~3 mins for 1000 spatial spins
        if i % 25 == 0:
            print("processing spin", i)
        # Process normal genes data
        cortical_spun_genes_data = cortical_genes_data[:][cortical_spin_indices_T[i]]
        subcortical_spun_genes_data = subcortical_genes_data[:][subcortical_spin_indices_T[i]]
        spun_genes_data = np.vstack([cortical_spun_genes_data, subcortical_spun_genes_data])
        X_corr, CGE_vec, dist_matrix, dist_vec = load_features(spun_genes_data, all_region_coords)

        SA_lambda, SA_inf = compute_exponential_distance_decay(dist_vec, CGE_vec, bin_size_mm=bin_size_mm)
        coefs = compute_distance_decay_poly3(dist_vec, CGE_vec, bin_size_mm=bin_size_mm)
        a1, a2, a3, a4 = coefs    
        sa_poly_params[i] = [SA_lambda, SA_inf, a1, a2, a3, a4]

        # Process PCA genes data
        cortical_spun_genes_data_PCA = cortical_genes_data_PCA[:][cortical_spin_indices_T[i]]
        subcortical_spun_genes_data_PCA = subcortical_genes_data_PCA[:][subcortical_spin_indices_T[i]]
        spun_genes_data_PCA = np.vstack([cortical_spun_genes_data_PCA, subcortical_spun_genes_data_PCA])
        X_corr_PCA, CGE_vec_PCA, dist_matrix_PCA, dist_vec_PCA = load_features(spun_genes_data_PCA, all_region_coords)
        
        SA_lambda_PCA, SA_inf_PCA = compute_exponential_distance_decay(dist_vec_PCA, CGE_vec_PCA, bin_size_mm=bin_size_mm)
        coefs_PCA = compute_distance_decay_poly3(dist_vec_PCA, CGE_vec_PCA, bin_size_mm=bin_size_mm)
        a1_PCA, a2_PCA, a3_PCA, a4_PCA = coefs_PCA
        sa_poly_params_PCA[i] = [SA_lambda_PCA, SA_inf_PCA, a1_PCA, a2_PCA, a3_PCA, a4_PCA]

    # ADD PARAMS TO UNIFIED DATAFRAME
    spins_df['SA_lambda'] = sa_poly_params[:,0]
    spins_df['SA_inf'] = sa_poly_params[:,1]
    spins_df['poly_a1'] = sa_poly_params[:,2] 
    spins_df['poly_a2'] = sa_poly_params[:,3]
    spins_df['poly_a3'] = sa_poly_params[:,4]
    spins_df['poly_a4'] = sa_poly_params[:,5]

    spins_df['SA_lambda_PCA'] = sa_poly_params_PCA[:,0]
    spins_df['SA_inf_PCA'] = sa_poly_params_PCA[:,1]
    spins_df['poly_a1_PCA'] = sa_poly_params_PCA[:,2]
    spins_df['poly_a2_PCA'] = sa_poly_params_PCA[:,3]
    spins_df['poly_a3_PCA'] = sa_poly_params_PCA[:,4]
    spins_df['poly_a4_PCA'] = sa_poly_params_PCA[:,5]

    # RELOAD IN TRUE BRAIN DATA AND COMPUTE TRUE PARAMETER VALUES
    X, X_pca, Y, coords, labels, network_labels, X_corr, X_pca_corr, dist_matrix = load_and_plot_data(parcellation='S400', hemisphere='both', omit_subcortical=False, 
                          sort_genes='expression', impute_strategy='mirror_interpolate', null_model='none', fontsize=26)

    features, features_cortex, features_subcortex = subset_brain_data(X, X_pca, Y, coords, labels, network_labels, X_corr, X_pca_corr, dist_matrix)
    dist_vec = features['distances']
    CGE_vec = features['CGE']
    PCA_CGE_vec = features['PCA_CGE']
    true_SA_lambda, true_SA_inf = compute_exponential_distance_decay(dist_vec, CGE_vec, bin_size_mm=5)
    true_coefs = compute_distance_decay_poly3(dist_vec, CGE_vec, bin_size_mm=5)
    true_SA_lambda_PCA, true_SA_inf_PCA = compute_exponential_distance_decay(dist_vec, PCA_CGE_vec, bin_size_mm=5)
    true_coefs_PCA = compute_distance_decay_poly3(dist_vec, PCA_CGE_vec, bin_size_mm=5)

    # STANDARDIZE PARAMETERS AND COMPUTE ERROR TO TRUE VALUES
    parameter_cols = ['SA_lambda', 'SA_inf', 'SA_lambda_PCA', 'SA_inf_PCA', 'poly_a1', 'poly_a2', 'poly_a3', 'poly_a4',
                     'poly_a1_PCA', 'poly_a2_PCA', 'poly_a3_PCA', 'poly_a4_PCA']
    standardized_df = spins_df.copy()

    # Standardize params of spin nulls
    standardized_df[parameter_cols] = (spins_df[parameter_cols] - spins_df[parameter_cols].mean()) / spins_df[parameter_cols].std()

    # Standardize true parameters using same mean/std as spin nulls
    true_SA_lambda_std = (true_SA_lambda - spins_df['SA_lambda'].mean()) / spins_df['SA_lambda'].std()
    true_SA_inf_std = (true_SA_inf - spins_df['SA_inf'].mean()) / spins_df['SA_inf'].std()
    true_SA_lambda_PCA_std = (true_SA_lambda_PCA - spins_df['SA_lambda_PCA'].mean()) / spins_df['SA_lambda_PCA'].std()
    true_SA_inf_PCA_std = (true_SA_inf_PCA - spins_df['SA_inf_PCA'].mean()) / spins_df['SA_inf_PCA'].std()
    true_coefs_std = (true_coefs - spins_df[['poly_a1', 'poly_a2', 'poly_a3', 'poly_a4']].mean()) / spins_df[['poly_a1', 'poly_a2', 'poly_a3', 'poly_a4']].std()
    true_coefs_PCA_std = (true_coefs_PCA - spins_df[['poly_a1_PCA', 'poly_a2_PCA', 'poly_a3_PCA', 'poly_a4_PCA']].mean()) / spins_df[['poly_a1_PCA', 'poly_a2_PCA', 'poly_a3_PCA', 'poly_a4_PCA']].std()

    # Extract parameter columns as numpy array
    null_params = standardized_df[parameter_cols].to_numpy()

    # Create array of true standardized values in same order as numeric columns
    true_std_vals = np.array([
        true_SA_lambda_std,
        true_SA_inf_std, 
        true_SA_lambda_PCA_std,
        true_SA_inf_PCA_std,
        true_coefs_std[0],
        true_coefs_std[1],
        true_coefs_std[2], 
        true_coefs_std[3],
        true_coefs_PCA_std[0],
        true_coefs_PCA_std[1],
        true_coefs_PCA_std[2],
        true_coefs_PCA_std[3]
    ])

    # Compute residuals (differences) between each null spin and true values
    residuals = null_params - true_std_vals[np.newaxis, :]

    # Calculate summed absolute residuals for each parameter group 
    # (L1 error i.e. sum of absolute differences)
    standardized_SA_error = np.abs(residuals[:, 0:2]).sum(axis=1)  # SA_lambda and SA_inf
    standardized_SA_PCA_error = np.abs(residuals[:, 2:4]).sum(axis=1)  # SA_lambda_PCA and SA_inf_PCA
    standardized_poly_error = np.abs(residuals[:, 4:8]).sum(axis=1)  # poly_a1 through poly_a4
    standardized_poly_PCA_error = np.abs(residuals[:, 8:12]).sum(axis=1)  # poly_a1_PCA through poly_a4_PCA

    # ADD ERROR COLUMNS AND COMPUTE MEAN ERROR RANK
    spins_df['standardized_SA_error'] = standardized_SA_error
    spins_df['standardized_poly_error'] = standardized_poly_error 
    spins_df['standardized_SA_PCA_error'] = standardized_SA_PCA_error
    spins_df['standardized_poly_PCA_error'] = standardized_poly_PCA_error
    
    spins_df['total_cost_rank'] = spins_df['total_cost'].rank()
    spins_df['SA_error_rank'] = spins_df['standardized_SA_error'].rank()
    spins_df['poly_error_rank'] = spins_df['standardized_poly_error'].rank()
    spins_df['mean_error_rank'] = (spins_df['total_cost_rank'] + 
                                  spins_df['SA_error_rank'] + 
                                  spins_df['poly_error_rank']).div(3)

    spins_df = spins_df[['cortical_spins', 'subcortical_spins', 'cortical_cost', 'subcortical_cost', 'total_cost', 
              'mean_error_rank', 'total_cost_rank', 'SA_error_rank', 'poly_error_rank', 'standardized_SA_error', 'standardized_poly_error', 'standardized_SA_PCA_error', 'standardized_poly_PCA_error',
              'SA_lambda', 'SA_inf', 'SA_lambda_PCA', 'SA_inf_PCA', 'poly_a1', 'poly_a2', 'poly_a3', 'poly_a4',
              'poly_a1_PCA', 'poly_a2_PCA', 'poly_a3_PCA', 'poly_a4_PCA']]

    if save_csv:
        spins_df.to_csv(f'./data/enigma/{n_rotations}_null_spins.csv', index=False)
        
    return spins_df

def get_iPA_masks(parcellation):
    """
    Get hemisphere and subcortical masks for a given parcellation.
    
    Args:
        parcellation (str): Name of parcellation (e.g. 'iPA_391')
        
    Returns:
        tuple: (hemi_mask_list, subcort_mask_list, n_cortical)
            - hemi_mask_list: List of 0s and 1s indicating right (1) vs left (0) hemisphere
            - subcort_mask_list: List of 0s and 1s indicating subcortical (1) vs cortical (0) regions
            - n_cortical: Number of cortical regions (optional)
    """
    BHA2_path = absolute_data_path + '/BHA2/'
    metadata = pd.read_csv(os.path.join(BHA2_path, parcellation, f'{parcellation}.csv'), index_col=0)

    # Create hemisphere mask based on right-sided regions
    right_cols = [col for col in metadata.columns if '_R' in col]
    hemi_mask = (metadata[right_cols] > 0).any(axis=1).astype(int)
    hemi_mask_list = hemi_mask.tolist()

    # Create subcortical mask based on subcortical regions 
    subcort_cols = [col for col in metadata.columns if 'Subcortical' in col]
    subcort_mask = (metadata[subcort_cols] > 0).any(axis=1).astype(int)
    subcort_mask_list = subcort_mask.tolist()

    # Count number of cortical regions (where subcortical mask is 0)
    n_cortical = sum(x == 0 for x in subcort_mask_list)

    return hemi_mask_list, subcort_mask_list, n_cortical

def generate_null_spins_iPA(n_rotations=100, seed=42, bin_size_mm=5, parcellation='iPA_391', save_csv=False):
    """Generate null spin models and compute distance decay parameters for both raw gene expression and PCA-transformed data.
    
    This function performs spatial rotations of brain regions to create null models while preserving spatial relationships.
    It handles cortical and subcortical regions separately, computes various distance decay parameters including exponential
    decay and polynomial fits, and standardizes the results relative to the true brain data.
    
    Args:
        n_rotations (int): Number of spin rotations to generate
        seed (int): Random seed for reproducibility 
        bin_size_mm (float): Bin size in mm for distance decay calculations
        save_csv (bool): Whether to save results to CSV
        
    Returns:
        pd.DataFrame: DataFrame containing:
            - Spin indices for cortical and subcortical regions
            - Cost metrics for the spin rotations
            - Standardized distance decay parameters (exponential and polynomial)
            - Error metrics comparing null models to true brain data
    """
    # LOAD IN TRUE GENE EXPRESSION DATA, GENE EXPRESSION PCA, TRUE COORDINATES
    genes_data = load_transcriptome(parcellation=parcellation, gene_list='0.2', run_PCA=False, omit_subcortical=False, hemisphere='both', impute_strategy='mirror_interpolate', sort_genes='expression', return_valid_genes=True, null_model='none', random_seed=42)
    genes_data_PCA = load_transcriptome(parcellation=parcellation, gene_list='0.2', run_PCA=True, omit_subcortical=False, hemisphere='both', impute_strategy='mirror_interpolate', sort_genes='expression', return_valid_genes=False, null_model='none', random_seed=42)
    all_region_coords = load_coords(parcellation=parcellation, omit_subcortical=False)

    # GENERATE SPINS FOR EACH HEMISPHERE OF CORTEX MATCHED
    #lh_annot, rh_annot = nndata.fetch_schaefer2018('fsaverage', data_dir='data/UKBB', verbose=1)['400Parcels7Networks']
    #coords, hemi = nnsurf.find_parcel_centroids(lhannot=lh_annot, rhannot=rh_annot, version='fsaverage', surf='sphere')
    hemi_mask_list, subcort_mask_list, n_cortical = get_iPA_masks(parcellation)
    hemi_cortical = [hemi for hemi, subcort in zip(hemi_mask_list, subcort_mask_list) if subcort == 0]
    coords_cortical = all_region_coords[:len(hemi_cortical)]
    
    hemi_subcortical = [hemi for hemi, subcort in zip(hemi_mask_list, subcort_mask_list) if subcort == 1]
    
    cortical_spins, cortical_cost = nnstats.gen_spinsamples(coords_cortical, hemi_cortical, n_rotate=n_rotations, seed=seed, method='vasa',return_cost=True)

    num_cortical_regions = len(hemi_cortical) # (genes_data.shape[0] // 100) * 100
    cortical_genes_data = genes_data[:num_cortical_regions]
    cortical_genes_data_PCA = genes_data_PCA[:num_cortical_regions]

    # GENERATE SPINS FOR EACH HEMISPHERE OF SUBCORTEX MATCHED
    subcortical_genes_data = genes_data[num_cortical_regions:]
    subcortical_genes_data_PCA = genes_data_PCA[num_cortical_regions:]
    subcortical_coords = all_region_coords[num_cortical_regions:]
    x_coords = subcortical_coords[:, 0]
    hemi_subcort = np.zeros_like(x_coords) # create mask for subcortical hemisphere
    hemi_subcort[x_coords > 0] = 1
    print(hemi_subcort)
    subcortical_spins, subcortical_cost = nnstats.gen_spinsamples(subcortical_coords, hemi_subcort, n_rotate=n_rotations, seed=seed, method='vasa', return_cost=True)

    # SAVE INDICES OF SPINS TO UNIFIED DATAFRAME
    cortical_spin_indices_T = cortical_spins.T
    subcortical_spin_indices_T = subcortical_spins.T
    spins_df = pd.DataFrame({
        'cortical_spins': cortical_spin_indices_T.tolist(),
        'subcortical_spins': subcortical_spin_indices_T.tolist()
    })

    # SAVE COST OF SPINS
    cortical_cost_T = cortical_cost.T
    subcortical_cost_T = subcortical_cost.T
    cortical_cost_sum = cortical_cost_T.sum(axis=1)
    subcortical_cost_sum = subcortical_cost_T.sum(axis=1)
    total_cost = subcortical_cost_sum + cortical_cost_sum
    spins_df['cortical_cost'] = cortical_cost_sum
    spins_df['subcortical_cost'] = subcortical_cost_sum 
    spins_df['total_cost'] = total_cost

    # INITIALIZE ARRAYS TO STORE SA EXPONENTIAL DECAY AND POLYNOMIAL COEFFICIENTS
    sa_poly_params = np.zeros((len(spins_df), 6))
    sa_poly_params_PCA = np.zeros((len(spins_df), 6))

    for i in range(len(spins_df)): # loop takes ~3 mins for 1000 spatial spins
        if i % 25 == 0:
            print("processing spin", i)
        # Process normal genes data
        cortical_spun_genes_data = cortical_genes_data[:][cortical_spin_indices_T[i]]
        subcortical_spun_genes_data = subcortical_genes_data[:][subcortical_spin_indices_T[i]]
        spun_genes_data = np.vstack([cortical_spun_genes_data, subcortical_spun_genes_data])
        X_corr, CGE_vec, dist_matrix, dist_vec = load_features(spun_genes_data, all_region_coords)

        SA_lambda, SA_inf = compute_exponential_distance_decay(dist_vec, CGE_vec, bin_size_mm=bin_size_mm)
        coefs = compute_distance_decay_poly3(dist_vec, CGE_vec, bin_size_mm=bin_size_mm)
        a1, a2, a3, a4 = coefs    
        sa_poly_params[i] = [SA_lambda, SA_inf, a1, a2, a3, a4]

        # Process PCA genes data
        cortical_spun_genes_data_PCA = cortical_genes_data_PCA[:][cortical_spin_indices_T[i]]
        subcortical_spun_genes_data_PCA = subcortical_genes_data_PCA[:][subcortical_spin_indices_T[i]]
        spun_genes_data_PCA = np.vstack([cortical_spun_genes_data_PCA, subcortical_spun_genes_data_PCA])
        X_corr_PCA, CGE_vec_PCA, dist_matrix_PCA, dist_vec_PCA = load_features(spun_genes_data_PCA, all_region_coords)
        
        SA_lambda_PCA, SA_inf_PCA = compute_exponential_distance_decay(dist_vec_PCA, CGE_vec_PCA, bin_size_mm=bin_size_mm)
        coefs_PCA = compute_distance_decay_poly3(dist_vec_PCA, CGE_vec_PCA, bin_size_mm=bin_size_mm)
        a1_PCA, a2_PCA, a3_PCA, a4_PCA = coefs_PCA
        sa_poly_params_PCA[i] = [SA_lambda_PCA, SA_inf_PCA, a1_PCA, a2_PCA, a3_PCA, a4_PCA]

    # ADD PARAMS TO UNIFIED DATAFRAME
    spins_df['SA_lambda'] = sa_poly_params[:,0]
    spins_df['SA_inf'] = sa_poly_params[:,1]
    spins_df['poly_a1'] = sa_poly_params[:,2] 
    spins_df['poly_a2'] = sa_poly_params[:,3]
    spins_df['poly_a3'] = sa_poly_params[:,4]
    spins_df['poly_a4'] = sa_poly_params[:,5]

    spins_df['SA_lambda_PCA'] = sa_poly_params_PCA[:,0]
    spins_df['SA_inf_PCA'] = sa_poly_params_PCA[:,1]
    spins_df['poly_a1_PCA'] = sa_poly_params_PCA[:,2]
    spins_df['poly_a2_PCA'] = sa_poly_params_PCA[:,3]
    spins_df['poly_a3_PCA'] = sa_poly_params_PCA[:,4]
    spins_df['poly_a4_PCA'] = sa_poly_params_PCA[:,5]

    # RELOAD IN TRUE BRAIN DATA AND COMPUTE TRUE PARAMETER VALUES
    X, X_pca, Y, coords, labels, network_labels, X_corr, X_pca_corr, dist_matrix = load_and_plot_data(parcellation=parcellation, hemisphere='both', omit_subcortical=False, 
                          sort_genes='expression', impute_strategy='mirror_interpolate', null_model='none', fontsize=26)

    features, features_cortex, features_subcortex = subset_brain_data(X, X_pca, Y, coords, labels, network_labels, X_corr, X_pca_corr, dist_matrix)
    dist_vec = features['distances']
    CGE_vec = features['CGE']
    PCA_CGE_vec = features['PCA_CGE']
    true_SA_lambda, true_SA_inf = compute_exponential_distance_decay(dist_vec, CGE_vec, bin_size_mm=bin_size_mm)
    true_coefs = compute_distance_decay_poly3(dist_vec, CGE_vec, bin_size_mm=bin_size_mm)
    true_SA_lambda_PCA, true_SA_inf_PCA = compute_exponential_distance_decay(dist_vec, PCA_CGE_vec, bin_size_mm=bin_size_mm)
    true_coefs_PCA = compute_distance_decay_poly3(dist_vec, PCA_CGE_vec, bin_size_mm=bin_size_mm)

    # STANDARDIZE PARAMETERS AND COMPUTE ERROR TO TRUE VALUES
    parameter_cols = ['SA_lambda', 'SA_inf', 'SA_lambda_PCA', 'SA_inf_PCA', 'poly_a1', 'poly_a2', 'poly_a3', 'poly_a4',
                     'poly_a1_PCA', 'poly_a2_PCA', 'poly_a3_PCA', 'poly_a4_PCA']
    standardized_df = spins_df.copy()

    # Standardize params of spin nulls
    standardized_df[parameter_cols] = (spins_df[parameter_cols] - spins_df[parameter_cols].mean()) / spins_df[parameter_cols].std()

    # Standardize true parameters using same mean/std as spin nulls
    true_SA_lambda_std = (true_SA_lambda - spins_df['SA_lambda'].mean()) / spins_df['SA_lambda'].std()
    true_SA_inf_std = (true_SA_inf - spins_df['SA_inf'].mean()) / spins_df['SA_inf'].std()
    true_SA_lambda_PCA_std = (true_SA_lambda_PCA - spins_df['SA_lambda_PCA'].mean()) / spins_df['SA_lambda_PCA'].std()
    true_SA_inf_PCA_std = (true_SA_inf_PCA - spins_df['SA_inf_PCA'].mean()) / spins_df['SA_inf_PCA'].std()
    true_coefs_std = (true_coefs - spins_df[['poly_a1', 'poly_a2', 'poly_a3', 'poly_a4']].mean()) / spins_df[['poly_a1', 'poly_a2', 'poly_a3', 'poly_a4']].std()
    true_coefs_PCA_std = (true_coefs_PCA - spins_df[['poly_a1_PCA', 'poly_a2_PCA', 'poly_a3_PCA', 'poly_a4_PCA']].mean()) / spins_df[['poly_a1_PCA', 'poly_a2_PCA', 'poly_a3_PCA', 'poly_a4_PCA']].std()

    # Extract parameter columns as numpy array
    null_params = standardized_df[parameter_cols].to_numpy()

    # Create array of true standardized values in same order as numeric columns
    true_std_vals = np.array([
        true_SA_lambda_std,
        true_SA_inf_std, 
        true_SA_lambda_PCA_std,
        true_SA_inf_PCA_std,
        true_coefs_std[0],
        true_coefs_std[1],
        true_coefs_std[2], 
        true_coefs_std[3],
        true_coefs_PCA_std[0],
        true_coefs_PCA_std[1],
        true_coefs_PCA_std[2],
        true_coefs_PCA_std[3]
    ])

    # Compute residuals (differences) between each null spin and true values
    residuals = null_params - true_std_vals[np.newaxis, :]

    # Calculate summed absolute residuals for each parameter group 
    # (L1 error i.e. sum of absolute differences)
    standardized_SA_error = np.abs(residuals[:, 0:2]).sum(axis=1)  # SA_lambda and SA_inf
    standardized_SA_PCA_error = np.abs(residuals[:, 2:4]).sum(axis=1)  # SA_lambda_PCA and SA_inf_PCA
    standardized_poly_error = np.abs(residuals[:, 4:8]).sum(axis=1)  # poly_a1 through poly_a4
    standardized_poly_PCA_error = np.abs(residuals[:, 8:12]).sum(axis=1)  # poly_a1_PCA through poly_a4_PCA

    # ADD ERROR COLUMNS AND COMPUTE MEAN ERROR RANK
    spins_df['standardized_SA_error'] = standardized_SA_error
    spins_df['standardized_poly_error'] = standardized_poly_error 
    spins_df['standardized_SA_PCA_error'] = standardized_SA_PCA_error
    spins_df['standardized_poly_PCA_error'] = standardized_poly_PCA_error
    
    spins_df['total_cost_rank'] = spins_df['total_cost'].rank()
    spins_df['SA_error_rank'] = spins_df['standardized_SA_error'].rank()
    spins_df['poly_error_rank'] = spins_df['standardized_poly_error'].rank()
    spins_df['mean_error_rank'] = (spins_df['total_cost_rank'] + 
                                  spins_df['SA_error_rank'] + 
                                  spins_df['poly_error_rank']).div(3)

    spins_df = spins_df[['cortical_spins', 'subcortical_spins', 'cortical_cost', 'subcortical_cost', 'total_cost', 
              'mean_error_rank', 'total_cost_rank', 'SA_error_rank', 'poly_error_rank', 'standardized_SA_error', 'standardized_poly_error', 'standardized_SA_PCA_error', 'standardized_poly_PCA_error',
              'SA_lambda', 'SA_inf', 'SA_lambda_PCA', 'SA_inf_PCA', 'poly_a1', 'poly_a2', 'poly_a3', 'poly_a4',
              'poly_a1_PCA', 'poly_a2_PCA', 'poly_a3_PCA', 'poly_a4_PCA']]

    if save_csv:
        spins_df.to_csv(f'./data/enigma/{n_rotations}_{parcellation}_null_spins.csv', index=False)
        
    return spins_df

def run_spin_test(X, Y_true, valid_indices, spins_df, model_type='CM', n_perms=1000, sort_spins='mean_error_rank', num_components=10, pre_fit=False):
    """
    Run spin test using precomputed spins to generate null distribution
    
    Parameters:
    -----------
    X : array-like
        Gene expression data matrix (n_regions x n_genes)
    Y_true : array-like 
        Connectivity matrix (n_regions x n_regions)
    spins_df : pandas DataFrame
        DataFrame containing precomputed spin indices
    model_type : str
        'CM' or 'PLS' - which model to use for fitting
    n_perms : int
        Number of null permutations to run
    shuffle_target : bool
        If True, shuffle connectivity matrix, if False shuffle gene expression
        
    Returns:
    --------
    empirical_corr : float
        Correlation between true and predicted values
    p_value : float
        Spin test p-value 
    null_corrs : array
        Distribution of null correlations
    """
    
    # Sort spins_df by standardized_SA_error in ascending order
    if sort_spins is not None:
        spins_df = spins_df.sort_values(sort_spins, ascending=True)

    # Get spin indices
    cortical_spins_list = spins_df['cortical_spins'].tolist()[:n_perms]
    cortical_spins_list = [eval(x) for x in cortical_spins_list]
    cortical_spin_indices = np.array(cortical_spins_list)
    
    subcortical_spins_list = spins_df['subcortical_spins'].tolist()[:n_perms]
    subcortical_spins_list = [eval(x) for x in subcortical_spins_list]
    subcortical_spin_indices = np.array(subcortical_spins_list)

    # Fit model to true data
    if model_type == 'CM':
        O, Y_pred = fit_cm_closed(X, Y_true)
        Y_pred_empirical = Y_pred
    else:  # PLS
        best_pls_model = PLSRegression(n_components=num_components)
        best_pls_model.fit(X, Y_true)
        Y_pred_empirical = best_pls_model.predict(X)
    
    # Calculate empirical correlation
    empirical_corr = pearsonr(Y_true.flatten(), Y_pred_empirical.flatten())[0]

    # Initialize array for null correlations
    if pre_fit:
        null_corrs = spins_df[f'pearsonr_{model_type}'].to_numpy()[:n_perms]
    else:
        null_corrs = np.zeros(n_perms)
    
        # Generate null distribution
        for i in range(n_perms):
            if i % 50 == 0:
                print(f"permutation: {i}")
            
            # Get spin indices for this permutation
            cortical_spin_idx = cortical_spin_indices[i] # min index here is 0, max is 399
            subcortical_spin_idx = subcortical_spin_indices[i]+400 # min index here is 400, max is 456
            
            # Drop index 455 from subcortical spin indices (always missing)
            subcortical_spin_idx = np.delete(subcortical_spin_idx, np.where(subcortical_spin_idx == 455))
            subcortical_spin_idx[subcortical_spin_idx == 456] = 455
            
            # Shuffle gene expression
            Y_rotated = Y_true
            X_cortical_rotated = X[cortical_spin_idx]
            X_subcortical_rotated = X[subcortical_spin_idx]
            X_rotated = np.vstack([X_cortical_rotated, X_subcortical_rotated])

            # Fit model on rotated data and get predictions
            if model_type == 'CM':
                O_null, Y_pred_null = fit_cm_closed(X_rotated, Y_rotated)
            else:  # PLS
                best_pls_model = PLSRegression(n_components=num_components)
                best_pls_model.fit(X_rotated, Y_rotated)
                Y_pred_null = best_pls_model.predict(X_rotated)
                
            null_corrs[i] = pearsonr(Y_rotated.flatten(), Y_pred_null.flatten())[0]

    # Calculate p-value
    p_value = max(1/(n_perms + 1), np.mean(null_corrs >= empirical_corr))

    # Plot null distribution
    plt.figure(figsize=(10, 6))
    plt.hist(null_corrs, bins=50, alpha=0.6, color='gray', label='Null distribution')
    plt.axvline(empirical_corr, color='red', linestyle='--', 
                label=f'Empirical (r={empirical_corr:.3f}, p={p_value:.3f})')
    plt.xlabel('Pearson correlation')
    plt.ylabel('Count')
    plt.title(f'{model_type} Spin Test Null Distribution')
    plt.legend()
    plt.show()
    
    return empirical_corr, p_value, null_corrs


def run_spin_test_random(X, Y_true, valid_indices, spins_df, model_type='CM', n_perms=1000, num_components=10):
    """
    Run spin test using precomputed spins to generate null distribution
    
    Parameters:
    -----------
    X : array-like
        Gene expression data matrix (n_regions x n_genes)
    Y_true : array-like 
        Connectivity matrix (n_regions x n_regions)
    spins_df : pandas DataFrame
        DataFrame containing precomputed spin indices
    model_type : str
        'CM' or 'PLS' - which model to use for fitting
    n_perms : int
        Number of null permutations to run
    shuffle_target : bool
        If True, shuffle connectivity matrix, if False shuffle gene expression
        
    Returns:
    --------
    empirical_corr : float
        Correlation between true and predicted values
    p_value : float
        Spin test p-value 
    null_corrs : array
        Distribution of null correlations
    """
    # Get spin indices from dataframe
    spin_indices_list = spins_df['true_random_spins'].tolist()[:n_perms]
    spin_indices_list = [eval(x) for x in spin_indices_list]
    spin_indices = np.array(spin_indices_list)
    
    # Adjust indices to be 0-based and within bounds
    # spin_indices = spin_indices - 1  # Convert from 1-based to 0-based indexing
    # spin_indices = np.clip(spin_indices, 0, X.shape[0]-1)  # Clip to valid range

    # Fit model to true data
    if model_type == 'CM':
        O, Y_pred = fit_cm_closed(X, Y_true)
        Y_pred_empirical = Y_pred
    else:  # PLS
        best_pls_model = PLSRegression(n_components=num_components)
        best_pls_model.fit(X, Y_true)
        Y_pred_empirical = best_pls_model.predict(X)
    
    # Calculate empirical correlation
    empirical_corr = pearsonr(Y_true.flatten(), Y_pred_empirical.flatten())[0]

    # Initialize array for null correlations
    null_corrs = np.zeros(n_perms)
    
    # Generate null distribution
    for i in range(n_perms):
        if i % 50 == 0:
            print(f"permutation: {i}")
        
        # Get spin indices for this permutation
        spin_idx = spin_indices[i] # min index here is 0, max is 456

        # Drop index 455 from spin indices (always missing)
        spin_idx = np.delete(spin_idx, np.where(spin_idx == 455))
        spin_idx[spin_idx == 456] = 455
        
        # Shuffle gene expression
        Y_rotated = Y_true
        X_rotated = X[spin_idx]

        # Fit model on rotated data and get predictions
        if model_type == 'CM':
            O_null, Y_pred_null = fit_cm_closed(X_rotated, Y_rotated)
        else:  # PLS
            best_pls_model = PLSRegression(n_components=num_components)
            best_pls_model.fit(X_rotated, Y_rotated)
            Y_pred_null = best_pls_model.predict(X_rotated)
            
        null_corrs[i] = pearsonr(Y_rotated.flatten(), Y_pred_null.flatten())[0]

    # Calculate p-value
    p_value = max(1/(n_perms + 1), np.mean(null_corrs >= empirical_corr))

    # Plot null distribution
    plt.figure(figsize=(10, 6))
    plt.hist(null_corrs, bins=50, alpha=0.6, color='gray', label='Null distribution')
    plt.axvline(empirical_corr, color='red', linestyle='--', 
                label=f'Empirical (r={empirical_corr:.3f}, p={p_value:.3f})')
    plt.xlabel('Pearson correlation')
    plt.ylabel('Count')
    plt.title(f'{model_type} Spin Test Null Distribution')
    plt.legend()
    plt.show()
    
    return empirical_corr, p_value, null_corrs


def run_spin_test_precomputed_colored(X, Y_true, valid_indices, spins_df, model_type='CM', num_components=10, n_perms=1000, sort_spins='mean_error_rank', bins=25, fontsize=24, pre_fit=False):
    """
    Run spin test using precomputed spins to generate null distribution
    
    Parameters:
    -----------
    X : array-like
        Gene expression data matrix (n_regions x n_genes)
    Y_true : array-like 
        Connectivity matrix (n_regions x n_regions)
    spins_df : pandas DataFrame
        DataFrame containing precomputed spin indices
    model_type : str
        'CM' or 'PLS' - which model to use for fitting
    num_components : int
        Number of PLS components to use (only used if model_type='PLS')
    n_perms : int
        Number of null permutations to run
    sort_spins : str
        Metric to sort spins by
    fontsize : int
        Font size for plot text elements
    prefit : bool
        If True, use pre fit null distribution
    """
    
    # Sort spins_df by standardized_SA_error in ascending order
    spins_df = spins_df.sort_values(sort_spins, ascending=True)

    # Get spin indices
    cortical_spins_list = spins_df['cortical_spins'].tolist()[:n_perms]
    cortical_spins_list = [eval(x) for x in cortical_spins_list]
    cortical_spin_indices = np.array(cortical_spins_list)
    
    subcortical_spins_list = spins_df['subcortical_spins'].tolist()[:n_perms]
    subcortical_spins_list = [eval(x) for x in subcortical_spins_list]
    subcortical_spin_indices = np.array(subcortical_spins_list)

    # Fit model to true data
    if model_type == 'CM':
        O, Y_pred = fit_cm_closed(X, Y_true)
        Y_pred_empirical = Y_pred
    else:  # PLS
        best_pls_model = PLSRegression(n_components=num_components)
        best_pls_model.fit(X, Y_true)
        Y_pred_empirical = best_pls_model.predict(X)
    
    # Calculate empirical correlation
    empirical_corr = pearsonr(Y_true.flatten(), Y_pred_empirical.flatten())[0]

    # Initialize arrays for null correlations and error metrics
    # Initialize array for null correlations
    if pre_fit:
        null_corrs = spins_df[f'pearsonr_{model_type}'].to_numpy()[:n_perms]
        error_metrics = {
            'mean_error_rank': spins_df['mean_error_rank'].to_numpy()[:n_perms],
            'total_cost': spins_df['total_cost'].to_numpy()[:n_perms],
            'standardized_SA_error': spins_df['standardized_SA_error'].to_numpy()[:n_perms],
            'standardized_poly_error': spins_df['standardized_poly_error'].to_numpy()[:n_perms],
            'standardized_SA_PCA_error': spins_df['standardized_SA_PCA_error'].to_numpy()[:n_perms],
            'standardized_poly_PCA_error': spins_df['standardized_poly_PCA_error'].to_numpy()[:n_perms]
        }
    else:
        null_corrs = np.zeros(n_perms)
        error_metrics = {
            'mean_error_rank': np.zeros(n_perms),
            'total_cost': np.zeros(n_perms),
            'standardized_SA_error': np.zeros(n_perms),
            'standardized_poly_error': np.zeros(n_perms),
            'standardized_SA_PCA_error': np.zeros(n_perms),
            'standardized_poly_PCA_error': np.zeros(n_perms)
        }
        
        # Generate null distribution
        # Generate null distribution
        for i in range(n_perms):
            if i % 50 == 0:
                print(f"permutation: {i}")
            
            # Get spin indices for this permutation
            cortical_spin_idx = cortical_spin_indices[i] # min index here is 0, max is 399
            subcortical_spin_idx = subcortical_spin_indices[i]+400 # min index here is 400, max is 456
            
            # Drop index 455 from subcortical spin indices (always missing)
            subcortical_spin_idx = np.delete(subcortical_spin_idx, np.where(subcortical_spin_idx == 455))
            subcortical_spin_idx[subcortical_spin_idx == 456] = 455
            
            # Shuffle gene expression
            Y_rotated = Y_true
            X_cortical_rotated = X[cortical_spin_idx]
            X_subcortical_rotated = X[subcortical_spin_idx]
            X_rotated = np.vstack([X_cortical_rotated, X_subcortical_rotated])
            
            # Fit model on rotated data and get predictions
            if model_type == 'CM':
                O_null, Y_pred_null = fit_cm_closed(X_rotated, Y_rotated)
            else:  # PLS
                best_pls_model = PLSRegression(n_components=num_components)
                best_pls_model.fit(X_rotated, Y_rotated)
                Y_pred_null = best_pls_model.predict(X_rotated)
                
            # Store error metrics for this permutation
            for metric in error_metrics.keys():
                error_metrics[metric][i] = spins_df[metric].iloc[i]
            
            null_corrs[i] = pearsonr(Y_rotated.flatten(), Y_pred_null.flatten())[0]

    # Calculate p-value
    p_value = max(1/(n_perms + 1), np.mean(null_corrs >= empirical_corr))

    # Create figure with 3x2 subplots
    fig, axes = plt.subplots(4, 2, figsize=(20, 18))
    axes = axes.flatten()  # Flatten for easier indexing
    
    # Plot 1: Standard uncolored histogram
    axes[0].hist(null_corrs, bins=bins, alpha=0.6, color='gray', label='Null distribution')
    axes[0].axvline(empirical_corr, color='red', linestyle='--', 
                    label=f'Empirical (r={empirical_corr:.2f})')
    
    # Add line for mean of top 10 lowest error rank spins
    top_10_idx = np.argsort(error_metrics['mean_error_rank'])[:10]
    top_10_mean = np.mean(null_corrs[top_10_idx])
    axes[0].axvline(top_10_mean, color='blue', linestyle='--',
                    label=f'Top 10 mean (r={top_10_mean:.2f})')
    
    axes[0].set_xlabel('Pearson correlation', fontsize=fontsize)
    axes[0].set_ylabel('Count', fontsize=fontsize)
    axes[0].set_title(f'Standard {model_type} Spin Test\nNull Distribution', fontsize=fontsize, pad=20)
    axes[0].legend(fontsize=fontsize-8)
    axes[0].tick_params(labelsize=fontsize-2)
    
    # Set x-axis ticks at 0.1 intervals
    axes[0].set_xticks(np.arange(0.2, 0.6, 0.1))
    # Set y-axis ticks at 500 intervals
    axes[0].set_yticks(np.arange(0, 1001, 500))
    
    # Calculate bin edges and centers once
    counts, bin_edges = np.histogram(null_corrs, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot colored histograms for each error metric
    for idx, (metric, values) in enumerate(error_metrics.items(), 1):
        # Calculate mean error metric for each bin
        bin_errors = np.zeros(len(bin_centers))
        for i in range(len(bin_centers)):
            mask = (null_corrs >= bin_edges[i]) & (null_corrs < bin_edges[i+1])
            bin_errors[i] = np.mean(values[mask]) if np.any(mask) else 0
        
        # Create colored histogram with darker colors for lower values
        norm = plt.Normalize(bin_errors.min(), bin_errors.max())
        colors = plt.cm.viridis(norm(bin_errors))  # Using reversed colormap
        
        # Plot bars
        bars = axes[idx].bar(bin_centers, counts, width=np.diff(bin_edges), 
                            color=colors, alpha=0.6)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        plt.colorbar(sm, ax=axes[idx], label=metric)
        
        # Add empirical line and top 10 mean line
        axes[idx].axvline(empirical_corr, color='red', linestyle='--',
                         label=f'Empirical (r={empirical_corr:.2f})')
        axes[idx].axvline(top_10_mean, color='blue', linestyle='--',
                         label=f'Top 10 mean (r={top_10_mean:.2f})')
        
        axes[idx].set_xlabel('Pearson correlation', fontsize=fontsize)
        axes[idx].set_ylabel('Count', fontsize=fontsize)
        axes[idx].set_title(f'{model_type} Spin Test Distribution\nColored by {metric}', fontsize=fontsize, pad=20)
        axes[idx].legend(fontsize=fontsize-8)
        axes[idx].tick_params(labelsize=fontsize-2)
        
        # Set x-axis ticks at 0.1 intervals
        axes[idx].set_xticks(np.arange(0.5, 0.8, 0.1))
        # Set y-axis ticks at 500 intervals
        axes[idx].set_yticks(np.arange(0, 1501, 500))
    
    plt.tight_layout()
    plt.show()
    
    return empirical_corr, p_value, null_corrs, error_metrics

### CM AND PLS FIT FUNCTIONS ###
# CM
def fit_cm(X, Y, alpha=0.0, verbose=True):
    """
    Fit Closed Form Connectome Model (CM) to predict connectivity from gene expression.
    Adapted from https://github.com/kpisti/SCM/tree/v1.0. Below implementation is aligned with Ridge Regression model from full paper.

    Args:
        X: Gene expression PCA matrix (regions x gene_PCA_dim)
        Y: Connectivity matrix (regions x regions) 
        alpha: Regularization parameter for ridge regression
        
    Returns:
        O: Estimated rule matrix (gene_PCA_dim x gene_PCA_dim)
        Y_pred: Predicted connectivity matrix (regions x regions)
        objective: Objective function value
    """
    # Check memory requirements before proceeding
    if verbose:
        print(f"X shape: {X.shape}")
        # Calculate memory needed for Kronecker product
        kron_size = (X.shape[0]**2) * (X.shape[1]**2) * 8  # 8 bytes per float64
        available_mem = psutil.virtual_memory().available
        if kron_size > available_mem:
            raise MemoryError(f"Kronecker product would require {kron_size/1e9:.2f}GB but only {available_mem/1e9:.2f}GB available")
    
    # Compute Kronecker product of X with itself to obtain K matrix
    K = np.kron(X, X)  
    if verbose: print(f"K shape after Kronecker: {K.shape}")
    
    # Flatten connectivity matrix Y
    if verbose: print(f"Y shape before flatten: {Y.shape}")
    Y_flat = Y.flatten() 
    if verbose: print(f"Y_flat shape: {Y_flat.shape}")
    
    # Compute K transpose multiplied by K
    K_transpose_K = np.dot(K.T, K) 
    if verbose: print(f"K_transpose_K shape: {K_transpose_K.shape}")
    
    # Add regularization term
    # K_transpose_K /= np.linalg.norm(K_transpose_K)
    K_reg = K_transpose_K + alpha * np.identity(K_transpose_K.shape[0]) 
    if verbose: print(f"K_reg shape: {K_reg.shape}")

    # Compute the pseudo-inverse solution
    K_pseudo_inv = np.linalg.pinv(K_reg).dot(K.T)
    if verbose: print(f"K_pseudo_inv shape: {K_pseudo_inv.shape}")
    
    # Estimate the rule matrix in vectorized form
    O_flat = K_pseudo_inv.dot(Y_flat)
    if verbose: print(f"O_flat shape: {O_flat.shape}")
    
    # Compute residuals and objective
    residuals = Y_flat - np.dot(K, O_flat)
    if verbose: print(f"Residuals shape: {residuals.shape}")
    residual_norm = np.linalg.norm(residuals)

    K_proj = np.dot(K, K_pseudo_inv)
    if verbose: print(f"K_proj shape: {K_proj.shape}")
    
    tau = np.trace(np.identity(K_proj.shape[0]) - K_proj)
    if verbose: print(f"tau: {tau}")
    
    objective = residual_norm*residual_norm/tau/tau # ** 2)
    if verbose: print(f"objective: {objective}")
    
    # Reshape O to original dimensions
    O = O_flat.reshape(X.shape[1], X.shape[1])
    
    # Predict connectivity matrix
    Y_pred = np.dot(X, np.dot(O, X.T))
    
    # Min-max scale Y_pred to [-1,1] range
    return O, Y_pred, objective

def fit_cm_closed(X, Y, alpha=0.0, plot=False):
    """
    Fit Closed Form Connectome Model (CM) using closed form solution.
    
    Args:
        X: Gene expression PCA matrix (regions x gene_PCA_dim)
        Y: Connectivity matrix (regions x regions)
        alpha: Regularization parameter
        
    Returns:
        O: Estimated rule matrix (gene_PCA_dim x gene_PCA_dim) 
        Y_pred: Predicted connectivity matrix (regions x regions)
    """
    A = X.T @ X + alpha * np.eye(X.shape[1])
    A_inv = np.linalg.inv(A)
    O = A_inv @ X.T @ Y @ X @ A_inv
    Y_pred = X @ O @ X.T

    if plot:
        # Flatten matrices for calculations
        y_true = Y.flatten()
        y_pred = Y_pred.flatten()

        # Metrics
        pearson_r, _ = stats.pearsonr(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        print("\nCM model metrics:")
        print(f"Pearson r: {pearson_r:.3f}")
        print(f"R-squared: {r2:.5f}") 
        print(f"MSE: {mse:.5f}")

        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot ground truth connectome
        im1 = ax1.imshow(Y, cmap='RdBu_r', vmin=-0.75, vmax=1)
        ax1.set_title('Ground Truth Connectome')
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Plot predicted connectome  
        im2 = ax2.imshow(Y_pred, cmap='RdBu_r', vmin=-0.75, vmax=1)
        ax2.set_title('Predicted Connectome')
        ax2.set_xticks([])
        ax2.set_yticks([])

        plt.tight_layout()
        plt.show()


        plt.figure(figsize=(8, 6))
        plt.imshow(O)
        #plt.colorbar(label='Beta Weight')
        plt.title('Bilinear Beta Weight Matrix')
        plt.xlabel('Gene Expression PCs')
        plt.ylabel('Gene Expression PCs') 
        plt.show()

    return O, Y_pred

def fit_cm_closed_with_scalar_bias(X, Y, alpha=0.0, plot=False):
    """
    Fit Closed Form Connectome Model (CM) with a learned scalar bias term.
    
    Args:
        X: Gene expression PCA matrix (regions x gene_PCA_dim)
        Y: Connectivity matrix (regions x regions)
        alpha: Regularization parameter for ridge regression
        
    Returns:
        O: Estimated bilinear operator (d x d)
        b: Learned scalar bias
        Y_pred: Predicted connectivity matrix (XOXᵀ + b)
    """
    # Compute inverse of regularized Gram matrix
    A = X.T @ X + alpha * np.eye(X.shape[1])
    A_inv = np.linalg.inv(A)

    # Estimate O from closed-form solution
    O = A_inv @ X.T @ Y @ X @ A_inv

    # Compute bilinear prediction
    Y_lin = X @ O @ X.T

    # Learn scalar bias to minimize mean residual
    b = np.mean(Y - Y_lin)

    # Final prediction with scalar bias
    Y_pred = Y_lin + b

    if plot:
        # Flatten matrices for calculations
        y_true = Y.flatten()
        y_pred = Y_pred.flatten()

        # Metrics
        pearson_r, _ = stats.pearsonr(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        print("\nCM model metrics:")
        print(f"Pearson r: {pearson_r:.3f}")
        print(f"R-squared: {r2:.5f}") 
        print(f"MSE: {mse:.5f}")

        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot ground truth connectome
        im1 = ax1.imshow(Y, cmap='RdBu_r', vmin=-0.75, vmax=1)
        ax1.set_title('Ground Truth Connectome')
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Plot predicted connectome  
        im2 = ax2.imshow(Y_pred, cmap='RdBu_r', vmin=-0.75, vmax=1)
        ax2.set_title('Predicted Connectome')
        ax2.set_xticks([])
        ax2.set_yticks([])

        plt.tight_layout()
        plt.show()


        plt.figure(figsize=(8, 6))
        plt.imshow(O)
        #plt.colorbar(label='Beta Weight')
        plt.title('Bilinear Beta Weight Matrix')
        plt.xlabel('Gene Expression PCs')
        plt.ylabel('Gene Expression PCs') 
        plt.show()
    
    return O, b, Y_pred

def fit_cm_closed_with_gcv(X, Y, alpha=0.0, return_all=False):
    """
    Closed-form CM fit with optional GCV scoring.

    Args:
        X (n x d): Input feature matrix (e.g. PCA of gene expression)
        Y (n x n): Symmetric target matrix (e.g. SC or FC)
        alpha: Ridge penalty
        return_all: If True, returns intermediate terms including GCV

    Returns:
        O: Learned operator
        Y_pred: Predicted target matrix
        GCV: Generalized cross-validation objective
    """
    n, d = X.shape
    A = X.T @ X + alpha * np.eye(d)
    A_inv = np.linalg.inv(A)

    O = A_inv @ X.T @ Y @ X @ A_inv
    Y_pred = X @ O @ X.T

    # Residual Frobenius norm
    residual_norm = np.linalg.norm(Y - Y_pred, ord='fro')

    # Compute hat matrix trace: Tr(H) = (Tr(P))^2
    P = X @ A_inv @ X.T
    tr_H = np.trace(P) ** 2
    tau = n**2 - tr_H

    gcv = (residual_norm ** 2) / (tau ** 2)

    if return_all:
        return O, Y_pred, gcv, residual_norm, tau, tr_H
    else:
        return O, Y_pred, gcv

def grid_search_alpha_with_gcv(X, Y, alpha_values):
    gcv_scores = []
    residuals = []
    taus = []
    traces = []

    for alpha in alpha_values:
        _, _, gcv, res_norm, tau, tr_H = fit_cm_closed_with_gcv(X, Y, alpha=alpha, return_all=True)
        print(f"alpha={alpha:<6} | GCV={gcv:.4e} | Residual={res_norm:.4e} | τ={tau:.2f} | Tr(H)={tr_H:.2f}")
        gcv_scores.append(gcv)
        residuals.append(res_norm)
        taus.append(tau)
        traces.append(tr_H)

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    axs[0].plot(alpha_values, gcv_scores, 'o-', lw=2)
    axs[0].set_xscale('symlog', linthresh=0.01)
    axs[0].set_title('GCV Objective vs Alpha')
    axs[0].set_ylabel('GCV Score')
    axs[0].grid(True)

    axs[1].plot(alpha_values, residuals, 's--', lw=2)
    axs[1].set_xscale('symlog', linthresh=0.01)
    axs[1].set_title('Residual Norm vs Alpha')
    axs[1].set_ylabel('Frobenius Norm of Residual')
    axs[1].grid(True)

    axs[2].plot(alpha_values, taus, 'd-.', lw=2)
    axs[2].set_xscale('symlog', linthresh=0.01)
    axs[2].set_title('Effective DoF (τ) vs Alpha')
    axs[2].set_ylabel('τ = n² − Tr(H)')
    axs[2].grid(True)

    axs[3].plot(alpha_values, traces, '^:', lw=2)
    axs[3].set_xscale('symlog', linthresh=0.01)
    axs[3].set_title('Trace of Hat Matrix vs Alpha')
    axs[3].set_ylabel('Tr(H) = (Tr(P))²')
    axs[3].grid(True)

    for ax in axs:
        ax.set_xlabel('Alpha')
        ax.tick_params(labelsize=12)
        ax.set_title(ax.get_title(), fontsize=14)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)

    plt.suptitle('CM Model Diagnostics vs Alpha', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return {
        'alpha': alpha_values,
        'gcv': gcv_scores,
        'residual': residuals,
        'tau': taus,
        'tr_H': traces
    }


# PLS
def get_best_pls_model(X, Y, max_components=25, use_vertical_elbow=False):
    """
    Evaluate PLS models and return the best one based on correlation score and elbow point.
    
    Args:
        X: Input features array
        Y: Target array
        max_components: Maximum number of components to try
        use_vertical_elbow: Whether to use vertical elbow point lines in plots
        
    Returns:
        PLSRegression: Best performing PLS model
        dict: Dictionary containing performance metrics
    """
    # Initialize lists to store metrics
    mse_scores = []
    r2_scores = []
    corr_scores = []
    covariance_explained = []
    models = []

    # Scale data for covariance calculation
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)

    # Compute total covariance between X and Y
    total_covariance = np.sum(np.cov(X_scaled.T, Y_scaled.T, bias=True)[0:len(X_scaled.T), len(X_scaled.T):])

    # Try different numbers of components
    for n_comp in range(1, max_components + 1):
        if n_comp % 5 == 0: 
            print(n_comp)
        
        # Calculate covariance explained
        pls_scaled = PLSRegression(n_components=n_comp)
        pls_scaled.fit(X_scaled, Y_scaled)
        X_scores = pls_scaled.x_scores_
        Y_scores = pls_scaled.y_scores_
        cov_sum = np.sum([np.cov(X_scores[:, i], Y_scores[:, i])[0, 1] for i in range(X_scores.shape[1])])
        covariance_explained.append(cov_sum / total_covariance)

        # refit on unscaled data for predictions in original data space        
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X, Y)
        models.append(pls)
        #models.append(pls_scaled)

        X = X # X_scaled
        Y = Y # Y_scaled
        # Make predictions
        Y_pred = pls.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(Y, Y_pred)
        r2 = r2_score(Y, Y_pred)
        
        # Calculate Pearson correlation for each connection
        pearson_corr = pearsonr(Y.flatten(), Y_pred.flatten())[0]
        
        # Store metrics
        mse_scores.append(mse)
        r2_scores.append(r2)
        corr_scores.append(pearson_corr)

    # Find elbow point using kneedle algorithm
    x = np.array(range(1, max_components + 1))
    y = np.array(corr_scores)
    
    # Normalize data for knee detection
    x_normalized = (x - min(x)) / (max(x) - min(x))
    y_normalized = (y - min(y)) / (max(y) - min(y))
    
    # Find knee point
    kn = KneeLocator(x_normalized, y_normalized, curve='concave', direction='increasing')
    elbow_idx = int(kn.knee * max_components) - 1  # Convert back to original scale
    
    # Use elbow point as best model
    best_model = models[elbow_idx]
    
    # Store metrics in dictionary
    metrics = {
        'n_components': elbow_idx + 1,
        'mse': mse_scores[elbow_idx],
        'r2': r2_scores[elbow_idx],
        'correlation': corr_scores[elbow_idx]
    }
    
    # Plot metrics
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].plot(range(1, max_components + 1), mse_scores)
    if use_vertical_elbow:
        axes[0].axvline(x=elbow_idx + 1, color='r', linestyle='--', label='Elbow point')
    axes[0].set_xlabel('Number of Components')
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].set_title('MSE vs Components')
    if use_vertical_elbow:
        axes[0].legend()

    axes[1].plot(range(1, max_components + 1), r2_scores)
    if use_vertical_elbow:
        axes[1].axvline(x=elbow_idx + 1, color='r', linestyle='--', label='Elbow point')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('R-squared Score')
    axes[1].set_title('R2 vs Components')
    if use_vertical_elbow:
        axes[1].legend()

    axes[2].plot(range(1, max_components + 1), corr_scores)
    if use_vertical_elbow:
        axes[2].axvline(x=elbow_idx + 1, color='r', linestyle='--', label='Elbow point')
    axes[2].set_xlabel('Number of Components')
    axes[2].set_ylabel('Pearson Correlation')
    axes[2].set_title('Correlation vs Components')
    if use_vertical_elbow:
        axes[2].legend()

    axes[3].plot(range(1, max_components + 1), covariance_explained, marker='o')
    axes[3].set_xlabel('Number of PLS Components')
    axes[3].set_ylabel('Cumulative Covariance Explained')
    axes[3].set_title('Cumulative Covariance Explained')
    
    # Find elbow point in covariance explained curve
    x = np.array(range(1, max_components + 1))
    y = np.array(covariance_explained)
    x_normalized = (x - min(x)) / (max(x) - min(x))
    y_normalized = (y - min(y)) / (max(y) - min(y))
    kn = KneeLocator(x_normalized, y_normalized, curve='concave', direction='increasing')
    covar_elbow_idx = int(kn.knee * max_components)
    if use_vertical_elbow:
        axes[3].axvline(x=covar_elbow_idx, color='r', linestyle='--', 
                        label=f'Elbow point at {covar_elbow_idx} components')
    
    # Find where 95% threshold is crossed
    threshold_idx = next((i for i, x in enumerate(covariance_explained) if x >= 0.95), None)
    if threshold_idx is not None and use_vertical_elbow:
        axes[3].axvline(x=threshold_idx + 1, color='g', linestyle='--', 
                       label=f'95% reached at {threshold_idx + 1} components')
    
    axes[3].grid(True)
    if use_vertical_elbow:
        axes[3].legend()

    plt.tight_layout()
    plt.show()

    print(f"Best model performance (at elbow point):")
    print(f"Number of components: {metrics['n_components']}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"R2: {metrics['r2']:.4f}")
    print(f"Mean correlation: {metrics['correlation']:.4f}")
    
    return best_model, metrics
