# Gene2Conn/data/data_viz.py

from imports import *


def plot_connectome(Y):
    """
    Function to plot the connectome without network labels.
    
    Parameters:
    Y (ndarray): Connectivity matrix.
    """
    # Visualize the matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(Y, cmap='Reds', vmin=0, vmax=1)
    
    # Add colorbar
    plt.colorbar(cax, shrink=0.8)
    
    #plt.title('Connectivity Matrix', fontsize=16)
    plt.xlabel('Regions', fontsize=14)
    plt.ylabel('Regions', fontsize=14)
    plt.show()


def plot_transcriptome(X):
    """
    Function to plot the transcriptome data.
    
    Parameters:
    X (ndarray): Transcriptome data matrix.
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    cax = ax.imshow(X, aspect=40, cmap='viridis')
    
    # Add colorbar
    plt.colorbar(cax, shrink=0.5)
    
    #plt.title('Transcriptome Data', fontsize=16)
    plt.xlabel('Genes', fontsize=14)
    plt.ylabel('Regions', fontsize=14)
    plt.show()

def plot_connectome_with_labels(Y, labels):
    """
    Function to plot the connectome with network labels.
    
    Parameters:
    Y (ndarray): Connectivity matrix.
    labels (list): List of region labels with network information.
    """
    networks = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'Subcortical']
    
    # Create a new ordering of nodes based on network
    new_order = []
    network_indices = {network: [] for network in networks}
    
    for idx, label in enumerate(labels):
        for network in networks:
            if network in label:
                network_indices[network].append(idx)
                break
        else:  # Handle Subcortical case
            if label.startswith(('L', 'R')):
                network_indices['Subcortical'].append(idx)
    
    for network in networks:
        new_order.extend(network_indices[network])
    
    # Map the original indices to the new ordering
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(new_order)}
    
    # Reorder the matrix
    Y_reordered = Y[np.ix_(new_order, new_order)]
    
    # Visualize the reordered matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(Y_reordered, cmap='viridis', vmin=0, vmax=np.max(Y))

    # Add red boxes around networks and place labels to the right
    start = 0
    for network in networks:
        size = len(network_indices[network])
        if size > 0:
            rect = patches.Rectangle((start, start), size, size, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(start + size + 2, start + size / 2, network, ha='left', va='center', fontsize=16, color='yellow')
            start += size
    
    #plt.title('Reordered Connectivity Matrix by Network Structure', fontsize=16)
    plt.xlabel('Regions', fontsize=14)
    plt.ylabel('Regions', fontsize=14)
    plt.show()





