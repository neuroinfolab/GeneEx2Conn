from env.imports import *

import importlib
import data
import data.data_utils
from data.data_load import load_transcriptome, load_connectome, load_coords
from models import *
from data import * 
from sim import *
import models
import models.metrics
from models.metrics import *

# SCM functions
def fit_scm(X, Y, alpha=0.0, verbose=True):
    """
    Fit Structural Covariance Model (SCM) to predict connectivity from gene expression.
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

def fit_scm_closed(X, Y, alpha=0.0, plot=False):
    """
    Fit Structural Covariance Model (SCM) using closed form solution.
    
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
        print("\nSCM model metrics:")
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

    return O, Y_pred

def fit_scm_closed_with_scalar_bias(X, Y, alpha=0.0, plot=False):
    """
    Fit Structural Covariance Model (SCM) with a learned scalar bias term.
    
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
        print("\nSCM model metrics:")
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

def fit_scm_closed_with_gcv(X, Y, alpha=0.0, return_all=False):
    """
    Closed-form SCM fit with optional GCV scoring.

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
        _, _, gcv, res_norm, tau, tr_H = fit_scm_closed_with_gcv(X, Y, alpha=alpha, return_all=True)
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

    plt.suptitle('SCM Model Diagnostics vs Alpha', fontsize=16)
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



