#!/usr/bin/env python

import os
import sys
import numpy as np
import pandas as pd
import argparse
import csv
import torch
import pickle
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader, Subset

# Define the absolute path to the BHA2 directory (using the same path from data_load.py)
absolute_data_path = '/scratch/asr655/neuroinformatics/GeneEx2Conn_data'
BHA2_path = absolute_data_path + '/BHA2/'

def load_ipa_coordinates(parcellation):
    """
    Load MNI coordinates for a specified iPA parcellation.
    
    Args:
        parcellation (str): The iPA parcellation (iPA_183, iPA_391, iPA_729)
        
    Returns:
        tuple: (coordinates array, parcellation dataframe)
    """
    atlas_info = pd.read_csv(os.path.join(BHA2_path, parcellation, f'{parcellation}.csv'))
    coordinates = atlas_info[['X_MNI', 'Y_MNI', 'Z_MNI']].values
    print(f"Loaded {parcellation} parcellation with {len(coordinates)} regions")
    return coordinates, atlas_info

def create_mapping_between_parcellations(source_parcellation, target_parcellation, distance_threshold=15.0):
    """
    Create a mapping from regions in a lower-resolution parcellation to 
    regions in a higher-resolution parcellation based on spatial proximity.
    Uses the "Closest Region Wins" approach to ensure each target region
    is assigned to exactly one source region (its closest source).
    
    Args:
        source_parcellation (str): Lower-resolution parcellation (e.g., iPA_183)
        target_parcellation (str): Higher-resolution parcellation (e.g., iPA_391)
        
    Returns:
        dict: Dictionary mapping source indices to lists of target indices
    """
    print(f"Creating mapping from {source_parcellation} to {target_parcellation}...")
    
    # Load coordinates for both parcellations
    source_coords, source_info = load_ipa_coordinates(source_parcellation)
    target_coords, target_info = load_ipa_coordinates(target_parcellation)
    
    # Calculate distance matrix between all pairs of regions
    distances = cdist(source_coords, target_coords, metric='euclidean')
    
    # Step 1: For each target region, determine which source region is closest
    closest_source_for_target = np.argmin(distances, axis=0)
    
    # Initialize the mapping dictionary with empty lists for each source region
    mapping = {i: [] for i in range(len(source_coords))}
    
    # Step 2: Assign each target to its closest source
    for target_idx, source_idx in enumerate(closest_source_for_target):
        mapping[source_idx].append(target_idx)
    
    # Step 3: Handle any source regions that didn't get assigned any targets
    empty_sources = [i for i, targets in mapping.items() if len(targets) == 0]
    if empty_sources:
        print(f"Warning: {len(empty_sources)} source regions have no assigned targets.")
        for source_idx in empty_sources:
            # Find the closest target for this source
            closest_target = np.argmin(distances[source_idx, :])
            # Find current assignment of this target
            current_source = closest_source_for_target[closest_target]
            # Only reassign if this source is actually closer
            if distances[source_idx, closest_target] < distances[current_source, closest_target]:
                # Reassign the target
                mapping[current_source].remove(closest_target)
                mapping[source_idx].append(closest_target)
                closest_source_for_target[closest_target] = source_idx
                print(f"  Reassigned target {closest_target} from source {current_source} to source {source_idx}")
            else:
                print(f"  Source {source_idx} remains without targets (closest target {closest_target} is better matched to source {current_source})")
    
    # For sources that still have no targets, assign the closest one regardless
    empty_sources = [i for i, targets in mapping.items() if len(targets) == 0]
    for source_idx in empty_sources:
        closest_target = np.argmin(distances[source_idx, :])
        current_source = closest_source_for_target[closest_target]
        # Remove from current assignment and assign to this source
        mapping[current_source].remove(closest_target)
        mapping[source_idx].append(closest_target)
        closest_source_for_target[closest_target] = source_idx
        print(f"  Forced assignment of target {closest_target} to source {source_idx} (from {current_source})")
    
    # Check coverage - each source region should map to at least one target region
    # and ideally target regions should be assigned to at least one source
    source_coverage = len(mapping)
    target_assigned = set(idx for sublist in mapping.values() for idx in sublist)
    target_coverage = len(target_assigned)
    
    print(f"Mapping coverage:")
    print(f"  Source regions mapped: {source_coverage}/{len(source_coords)} ({source_coverage/len(source_coords)*100:.1f}%)")
    print(f"  Target regions assigned: {target_coverage}/{len(target_coords)} ({target_coverage/len(target_coords)*100:.1f}%)")
    
    # Calculate distribution statistics for targets per source region
    targets_per_source = [len(targets) for targets in mapping.values()]
    avg_targets = sum(targets_per_source) / len(mapping)
    min_targets = min(targets_per_source)
    max_targets = max(targets_per_source)
    
    print(f"  1-to-many mapping statistics:")
    print(f"    Minimum targets per source region: {min_targets}")
    print(f"    Average targets per source region: {avg_targets:.2f}")
    print(f"    Maximum targets per source region: {max_targets}")
    
    # Count regions with various numbers of mappings
    single_mappings = sum(1 for count in targets_per_source if count == 1)
    multi_mappings = sum(1 for count in targets_per_source if count > 1)
    print(f"    Source regions with only 1 target: {single_mappings} ({single_mappings/len(mapping)*100:.1f}%)")
    print(f"    Source regions with multiple targets: {multi_mappings} ({multi_mappings/len(mapping)*100:.1f}%)")
    
    # Return the mapping from source to target indices
    return mapping, distances, distance_threshold

def print_mapping_stats(mapping, source_name, target_name):
    """
    Print statistics about the mapping between parcellations.
    
    Args:
        mapping (dict): Mapping from source to target
        source_name (str): Name of the source parcellation
        target_name (str): Name of the target parcellation
    """
    # Calculate the number of target regions per source region
    mapping_counts = [len(targets) for targets in mapping.values()]
    
    # Verify exclusivity - each target appears in exactly one mapping
    all_targets = []
    for targets in mapping.values():
        all_targets.extend(targets)
    unique_targets = set(all_targets)
    
    # Print statistics
    print(f"\nMapping statistics from {source_name} to {target_name}:")
    print(f"  Total source regions: {len(mapping)}")
    print(f"  Total target regions: {len(unique_targets)}")
    print(f"  Total mappings: {sum(mapping_counts)}")
    print(f"  Min targets per source: {min(mapping_counts)}")
    print(f"  Max targets per source: {max(mapping_counts)}")
    print(f"  Avg targets per source: {sum(mapping_counts)/len(mapping):.2f}")
    
    # Verify exclusivity
    if len(all_targets) == len(unique_targets):
        print(f"  ✓ Exclusive mapping confirmed: each target region is assigned to exactly one source region")
    else:
        print(f"  ✗ WARNING: Non-exclusive mapping detected! {len(all_targets) - len(unique_targets)} duplicate assignments")
    
    # Print distribution
    counts = {}
    for count in mapping_counts:
        counts[count] = counts.get(count, 0) + 1
    
    print("\n  Distribution of mappings:")
    for count in sorted(counts.keys()):
        print(f"    {count} targets: {counts[count]} source regions ({counts[count]/len(mapping)*100:.1f}%)")
    
    # List source regions with no targets, if any
    empty_sources = [src for src, targets in mapping.items() if len(targets) == 0]
    if empty_sources:
        print(f"\n  WARNING: {len(empty_sources)} source regions have no assigned targets: {empty_sources[:10]}{'...' if len(empty_sources) > 10 else ''}")
    
    # Print a few examples sorted by mapping count (prioritize showing diverse examples)
    print("\n  Sample mappings (from different count groups):")
    # Group by count
    count_groups = {}
    for src, targets in mapping.items():
        count = len(targets)
        if count not in count_groups:
            count_groups[count] = []
        count_groups[count].append(src)
    
    # Show at least one example from each count group, prioritizing diverse examples
    examples_shown = 0
    for count in sorted(count_groups.keys()):
        if examples_shown >= 5: break
        src = np.random.choice(count_groups[count])
        targets = mapping[src]
        print(f"    Source {src} → {len(targets)} targets: {targets[:10]}{'...' if len(targets) > 10 else ''}")
        examples_shown += 1
    
    # Add additional random examples if needed
    if examples_shown < 5:
        remaining_sources = set(mapping.keys()) - set([src for count in count_groups.values() for src in count_groups[count][:1]])
        if remaining_sources:
            samples = np.random.choice(list(remaining_sources), min(5-examples_shown, len(remaining_sources)), replace=False)
            for src in samples:
                targets = mapping[src]
                print(f"    Source {src} → {len(targets)} targets: {targets[:10]}{'...' if len(targets) > 10 else ''}")

def save_mapping(mapping, source_name, target_name):
    """
    Save the mapping to a CSV file.
    
    Args:
        mapping (dict): Mapping dictionary
        source_name (str): Name of the source parcellation
        target_name (str): Name of the target parcellation
    """
    # Convert the mapping to a dataframe
    mapping_rows = []
    for source_idx, target_indices in mapping.items():
        for target_idx in target_indices:
            mapping_rows.append({
                'source_index': source_idx,
                'target_index': target_idx
            })
    
    mapping_df = pd.DataFrame(mapping_rows)
    
    # Save to CSV
    output_file = f'mapping_{source_name}_to_{target_name}.csv'
    mapping_df.to_csv(output_file, index=False)
    print(f"Mapping saved to {output_file}")

def main():
    # Define the parcellations
    parcellations = ['iPA_183', 'iPA_391', 'iPA_729']
    
    # Create mappings between consecutive parcellation levels
    mappings = {}
    parcellation_info = {}
    
    for i in range(len(parcellations) - 1):
        source = parcellations[i]
        target = parcellations[i + 1]
        
        mapping, source_info, target_info = create_mapping_between_parcellations(source, target)
        mappings[(source, target)] = mapping
        
        if source not in parcellation_info:
            parcellation_info[source] = source_info
        if target not in parcellation_info:
            parcellation_info[target] = target_info
        
        print_mapping_stats(mapping, source, target)
        save_mapping(mapping, source, target)
    
    # Now create a direct mapping from iPA_183 to iPA_729 by composition
    if len(parcellations) >= 3:
        print("\nCreating composite mapping from iPA_183 to iPA_729...")
        
        mapping_183_to_391 = mappings[('iPA_183', 'iPA_391')]
        mapping_391_to_729 = mappings[('iPA_391', 'iPA_729')]
        
        composite_mapping = {}
        for source_idx, intermediate_indices in mapping_183_to_391.items():
            target_indices = []
            for intermediate_idx in intermediate_indices:
                if intermediate_idx in mapping_391_to_729:
                    target_indices.extend(mapping_391_to_729[intermediate_idx])
            
            composite_mapping[source_idx] = list(set(target_indices))  # Remove duplicates
        
        mappings[('iPA_183', 'iPA_729')] = composite_mapping
        print_mapping_stats(composite_mapping, 'iPA_183', 'iPA_729')
        save_mapping(composite_mapping, 'iPA_183', 'iPA_729')
    
    # Create a unified JSON representation of all mappings
    # Convert NumPy int64 values to Python native integers to make them JSON serializable
    all_mappings = {}
    for (source, target), mapping in mappings.items():
        converted_mapping = {}
        for k, v in mapping.items():
            # Convert keys to strings and values (lists of indices) to python native integers
            converted_mapping[str(k)] = [int(idx) for idx in v]
        all_mappings[f"{source}_to_{target}"] = converted_mapping
    
    import json
    with open('ipa_parcellation_mappings.json', 'w') as f:
        json.dump(all_mappings, f, indent=2)
    
    print("\nAll mappings created and saved!")

def map_cv_indices_between_parcellations(cv_indices_file, source_parcellation, target_parcellation, 
                                 X_target=None, Y_target=None, distance_threshold=15.0):
    """
    Map cross-validation indices from a source parcellation to a target parcellation and
    optionally create a dataset for the target parcellation using these mapped indices.
    
    Parameters:
    -----------
    cv_indices_file : str
        Path to the CSV file containing CV indices from the source parcellation
    source_parcellation : str
        Name of the source parcellation (e.g., 'iPA_183')
    target_parcellation : str
        Name of the target parcellation to map to (e.g., 'iPA_391')
    X_target : numpy.ndarray, optional
        Feature matrix for the target parcellation
    Y_target : numpy.ndarray, optional
        Connectivity matrix for the target parcellation
    distance_threshold : float, optional
        Maximum distance (in mm) for mapping between regions
        
    Returns:
    --------
    dict
        A dictionary containing:
        - 'mapped_indices': Dictionary with 'train_indices' and 'test_indices' for the target parcellation
        - 'X_train', 'X_test', 'Y_train', 'Y_test': If X_target and Y_target are provided
        - 'mapping': The mapping dictionary used for the conversion
    """
    # Load the CV indices from the source parcellation
    cv_df = pd.read_csv(cv_indices_file)
    
    # Extract train and test indices
    if 'train_index' in cv_df.columns:
        source_train_indices = cv_df['train_index'].values
        source_test_indices = None
    elif 'test_index' in cv_df.columns:
        source_train_indices = None
        source_test_indices = cv_df['test_index'].values
    else:
        raise ValueError(f"CSV file must contain either 'train_index' or 'test_index' column")
    
    # Create mapping between parcellations
    mapping, _, _ = create_mapping_between_parcellations(
        source_parcellation, target_parcellation, distance_threshold)
    
    result = {'mapping': mapping}
    
    # Map the indices
    mapped_indices = {}
    
    if source_train_indices is not None:
        target_train_indices = []
        for idx in source_train_indices:
            if idx in mapping:
                target_train_indices.extend(mapping[idx])
        mapped_indices['train_indices'] = sorted(list(set(target_train_indices)))
    
    if source_test_indices is not None:
        target_test_indices = []
        for idx in source_test_indices:
            if idx in mapping:
                target_test_indices.extend(mapping[idx])
        mapped_indices['test_indices'] = sorted(list(set(target_test_indices)))
    
    result['mapped_indices'] = mapped_indices
    
    # If feature and target matrices are provided, create train/test splits
    if X_target is not None and Y_target is not None:
        if 'train_indices' in mapped_indices and 'test_indices' in mapped_indices:
            result['X_train'] = X_target[mapped_indices['train_indices']]
            result['X_test'] = X_target[mapped_indices['test_indices']]
            result['Y_train'] = Y_target[mapped_indices['train_indices']]
            result['Y_test'] = Y_target[mapped_indices['test_indices']]
        elif 'train_indices' in mapped_indices:
            # If only train indices are available, assume the rest are test
            all_indices = set(range(len(X_target)))
            test_indices = sorted(list(all_indices - set(mapped_indices['train_indices'])))
            mapped_indices['test_indices'] = test_indices
            result['X_train'] = X_target[mapped_indices['train_indices']]
            result['X_test'] = X_target[test_indices]
            result['Y_train'] = Y_target[mapped_indices['train_indices']]
            result['Y_test'] = Y_target[test_indices]
        elif 'test_indices' in mapped_indices:
            # If only test indices are available, assume the rest are train
            all_indices = set(range(len(X_target)))
            train_indices = sorted(list(all_indices - set(mapped_indices['test_indices'])))
            mapped_indices['train_indices'] = train_indices
            result['X_train'] = X_target[train_indices]
            result['X_test'] = X_target[mapped_indices['test_indices']]
            result['Y_train'] = Y_target[train_indices]
            result['Y_test'] = Y_target[mapped_indices['test_indices']]
    
    return result


def create_multi_resolution_dataset(cv_indices_dir, source_parcellation, target_parcellation,
                                   X_source, Y_source, X_target, Y_target,
                                   coords_source=None, coords_target=None, fold=0, distance_threshold=15.0):
    """
    Create a multi-resolution dataset by mapping train/test splits from a source parcellation
    to a target parcellation. Returns RegionPairDataset objects for both resolutions.
    
    Parameters:
    -----------
    cv_indices_dir : str
        Directory containing CV indices CSV files
    source_parcellation : str
        Name of the source parcellation (e.g., 'iPA_183')
    target_parcellation : str
        Name of the target parcellation to map to (e.g., 'iPA_391')
    X_source : numpy.ndarray
        Feature matrix for the source parcellation
    Y_source : numpy.ndarray
        Connectivity matrix for the source parcellation
    X_target : numpy.ndarray
        Feature matrix for the target parcellation
    Y_target : numpy.ndarray
        Connectivity matrix for the target parcellation
    coords_source : numpy.ndarray, optional
        Coordinates for the source parcellation
    coords_target : numpy.ndarray, optional
        Coordinates for the target parcellation
    fold : int, optional
        Fold index to use
    distance_threshold : float, optional
        Maximum distance (in mm) for mapping between regions
        
    Returns:
    --------
    dict
        A dictionary containing:
        'source_dataset': RegionPairDataset for the source parcellation
        'target_dataset': RegionPairDataset for the target parcellation
        'mapping': The mapping between source and target parcellations
    """
    from data.data_utils import RegionPairDataset
    
    # Load coordinates if not provided
    if coords_source is None:
        coords_source, _ = load_ipa_coordinates(source_parcellation)
    if coords_target is None:
        coords_target, _ = load_ipa_coordinates(target_parcellation)
    
    # Handle NaN values in gene expression data for both source and target parcellations
    # For source parcellation
    source_valid_indices = ~np.isnan(X_source).all(axis=1)
    source_valid_indices_values = np.where(source_valid_indices)[0]
    source_valid2true_mapping = dict(enumerate(source_valid_indices_values))
    
    # Subset source data using valid indices
    X_source_valid = X_source[source_valid_indices]
    Y_source_valid = Y_source[source_valid_indices][:, source_valid_indices]
    coords_source_valid = coords_source[source_valid_indices]
    
    # For target parcellation
    target_valid_indices = ~np.isnan(X_target).all(axis=1)
    target_valid_indices_values = np.where(target_valid_indices)[0]
    target_valid2true_mapping = dict(enumerate(target_valid_indices_values))
    
    # Subset target data using valid indices
    X_target_valid = X_target[target_valid_indices]
    Y_target_valid = Y_target[target_valid_indices][:, target_valid_indices]
    coords_target_valid = coords_target[target_valid_indices]
    
    print(f"Source parcellation: {len(source_valid_indices_values)}/{len(X_source)} valid regions")
    print(f"Target parcellation: {len(target_valid_indices_values)}/{len(X_target)} valid regions")
    
    # Locate the train and test index files for the specified fold
    train_file = os.path.join(cv_indices_dir, f"{source_parcellation}_fold{fold}_train_indices.csv")
    test_file = os.path.join(cv_indices_dir, f"{source_parcellation}_fold{fold}_test_indices.csv")
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(f"CV indices files not found for fold {fold}. Please run RandomCVSplit first.")
    
    # Map the train indices - use valid data only
    train_result = map_cv_indices_between_parcellations(
        train_file, source_parcellation, target_parcellation, X_target_valid, Y_target_valid, distance_threshold)
    
    # Map the test indices - use valid data only
    test_result = map_cv_indices_between_parcellations(
        test_file, source_parcellation, target_parcellation, X_target_valid, Y_target_valid, distance_threshold)
    
    # Combine mapped indices
    target_train_indices = train_result['mapped_indices'].get('train_indices', [])
    target_test_indices = test_result['mapped_indices'].get('test_indices', [])

    # Create source dataset from original indices
    # Read train indices
    try:
        source_train_df = pd.read_csv(train_file)
        if 'train_index' in source_train_df.columns:
            source_train_indices = source_train_df['train_index'].values.tolist()
        else:
            # Try to read as a single column CSV
            source_train_indices = source_train_df.iloc[:, 0].values.tolist()
    except Exception as e:
        print(f"Error reading train indices CSV: {e}")
        # Fallback to reading as plain CSV
        source_train_indices = []
        with open(train_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                source_train_indices.extend([int(idx) for idx in row if idx.strip()])
    
    # Read test indices
    try:
        source_test_df = pd.read_csv(test_file)
        if 'test_index' in source_test_df.columns:
            source_test_indices = source_test_df['test_index'].values.tolist()
        else:
            # Try to read as a single column CSV
            source_test_indices = source_test_df.iloc[:, 0].values.tolist()
    except Exception as e:
        print(f"Error reading test indices CSV: {e}")
        # Fallback to reading as plain CSV
        source_test_indices = []
        with open(test_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                source_test_indices.extend([int(idx) for idx in row if idx.strip()])
    
    # Map source indices through valid2true mapping if needed
    # This ensures we're referencing the correct indices in the original (non-NaN-filtered) dataset
    if len(source_valid_indices_values) < len(X_source):
        print("Adjusting source indices for NaN-filtered data...")
        # We need to adjust the train/test indices to account for NaN values
        source_train_indices = [i for i in source_train_indices if i < len(source_valid_indices_values)]
        source_test_indices = [i for i in source_test_indices if i < len(source_valid_indices_values)]
        
    # Similarly for target indices
    if len(target_valid_indices_values) < len(X_target):
        print("Adjusting target indices for NaN-filtered data...")
        target_train_indices = [i for i in target_train_indices if i < len(target_valid_indices_values)]
        target_test_indices = [i for i in target_test_indices if i < len(target_valid_indices_values)]

    # Create mappings from subset indices to full dataset indices
    source_valid2true_mapping = {}
    for i, idx in enumerate(np.concatenate([source_train_indices, source_test_indices])):
        source_valid2true_mapping[i] = int(idx)
    
    target_valid2true_mapping = {}
    for i, idx in enumerate(np.concatenate([target_train_indices, target_test_indices])):
        target_valid2true_mapping[i] = int(idx)
    
    # Create subsets of data for the valid indices
    valid_source_indices = np.concatenate([source_train_indices, source_test_indices])
    valid_target_indices = np.concatenate([target_train_indices, target_test_indices])
    
    X_source_subset = X_source[valid_source_indices]
    Y_source_subset = Y_source[valid_source_indices]
    coords_source_subset = coords_source[valid_source_indices]
    
    X_target_subset = X_target[valid_target_indices]
    Y_target_subset = Y_target[valid_target_indices]
    coords_target_subset = coords_target[valid_target_indices]
    
    # Create mappings from original indices to their positions in the subset
    source_true2subset = {int(idx): i for i, idx in enumerate(valid_source_indices)}
    target_true2subset = {int(idx): i for i, idx in enumerate(valid_target_indices)}
    
    # Map the original train/test indices to their positions in the subset
    source_subset_train_indices = np.array([source_true2subset[int(idx)] for idx in source_train_indices])
    source_subset_test_indices = np.array([source_true2subset[int(idx)] for idx in source_test_indices])
    target_subset_train_indices = np.array([target_true2subset[int(idx)] for idx in target_train_indices])
    target_subset_test_indices = np.array([target_true2subset[int(idx)] for idx in target_test_indices])

    # Create RegionPairDataset objects
    source_dataset = RegionPairDataset(
        X=X_source_subset,
        Y=Y_source_subset,
        coords=coords_source_subset,
        valid2true_mapping=source_valid2true_mapping
    )
    
    target_dataset = RegionPairDataset(
        X=X_target_subset,
        Y=Y_target_subset,
        coords=coords_target_subset,
        valid2true_mapping=target_valid2true_mapping
    )
    # Create dataset pair object
    return {
        'source_dataset': source_dataset,
        'target_dataset': target_dataset,
        'source_train_indices': source_subset_train_indices,
        'source_test_indices': source_subset_test_indices,
        'target_train_indices': target_subset_train_indices,
        'target_test_indices': target_subset_test_indices,
        'original_source_train_indices': source_train_indices,
        'original_source_test_indices': source_test_indices,
        'original_target_train_indices': target_train_indices,
        'original_target_test_indices': target_test_indices,
        'mapping': train_result['mapping']  # The mapping used for conversion
    }


def evaluate_model_on_multi_resolution(model_checkpoint_path, source_parcellation, target_parcellation,
                                    cv_indices_dir, X_source, Y_source, X_target, Y_target,
                                    coords_source=None, coords_target=None, fold=0, distance_threshold=15.0,
                                    batch_size=512):
    """
    Evaluate a saved transformer model checkpoint on a multi-resolution dataset.
    
    Parameters:
    -----------
    model_checkpoint_path : str
        Path to the saved PyTorch transformer model checkpoint file (.pt)
    source_parcellation : str
        Name of the source parcellation used to train the model (e.g., 'iPA_183')
    target_parcellation : str
        Name of the target parcellation to evaluate on (e.g., 'iPA_391')
    cv_indices_dir : str
        Directory containing CV indices CSV files
    X_source : numpy.ndarray
        Feature matrix for the source parcellation
    Y_source : numpy.ndarray
        Connectivity matrix for the source parcellation
    X_target : numpy.ndarray
        Feature matrix for the target parcellation
    Y_target : numpy.ndarray
        Connectivity matrix for the target parcellation
    coords_source : numpy.ndarray, optional
        Coordinates for the source parcellation
    coords_target : numpy.ndarray, optional
        Coordinates for the target parcellation
    fold : int, optional
        Fold index to use
    distance_threshold : float, optional
        Maximum distance (in mm) for mapping between regions
    batch_size : int, optional
        Batch size for DataLoader (default: 512)
        
    Returns:
    --------
    dict
        A dictionary containing metrics for both source and target datasets
    """
    # Load the PyTorch transformer model checkpoint
    # Map to CPU first to avoid device mismatch issues
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    
    # Check if the loaded object is a state_dict (OrderedDict) or a full model
    if isinstance(checkpoint, dict):
        # We need to create a model instance first and then load the state dict
        from models.transformer_models import SharedSelfAttentionCLSModel
        import json
        import os
        
        # Try to find and load the config file that matches the checkpoint
        config_path = None
        if model_checkpoint_path.endswith('_best.pt'):
            # Replace _best.pt with _config.json as specified by the user
            possible_config_path = model_checkpoint_path.replace('_best.pt', '_config.json')
            if os.path.exists(possible_config_path):
                config_path = possible_config_path
        elif model_checkpoint_path.endswith('.pt'):
            # Try other patterns if no match with _best.pt
            # Try with .json extension
            possible_config_path = model_checkpoint_path.replace('.pt', '.json')
            if os.path.exists(possible_config_path):
                config_path = possible_config_path
            else:
                # Try with _config.json suffix
                possible_config_path = model_checkpoint_path.replace('.pt', '_config.json')
                if os.path.exists(possible_config_path):
                    config_path = possible_config_path
        
        if config_path:
            print(f"Loading model config from {config_path}")
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
                
            # Create a model with the exact same configuration as the saved model
            model = SharedSelfAttentionCLSModel(
                input_dim=saved_config.get('input_dim', X_source.shape[1] * 2),  # The saved model might have used a different input dimension
                binarize=saved_config.get('binarize', False),
                token_encoder_dim=saved_config.get('token_encoder_dim', 30),
                d_model=saved_config.get('d_model', 128),
                encoder_output_dim=saved_config.get('encoder_output_dim', 10),
                nhead=saved_config.get('nhead', 4),
                num_layers=saved_config.get('num_layers', 4),
                deep_hidden_dims=saved_config.get('deep_hidden_dims', [256, 128]),
                cls_init=saved_config.get('cls_init', 'spatial_learned'),
                use_alibi=saved_config.get('use_alibi', False),
                transformer_dropout=saved_config.get('transformer_dropout', 0.2),
                dropout_rate=saved_config.get('dropout_rate', 0.2),
                learning_rate=saved_config.get('learning_rate', 0.00009),
                weight_decay=saved_config.get('weight_decay', 0.0001),
                batch_size=saved_config.get('batch_size', 512),
                aug_prob=saved_config.get('aug_prob', 0.0)
            )
        else:
            # If no config file is found, use the sweep config parameters but try to guess the input dimension from the checkpoint
            print("No config file found. Attempting to infer model architecture from checkpoint...")
            
            # Get the first layer's weight shape to calculate the correct input dimension
            if 'deep_layers.0.weight' in checkpoint:
                first_layer_shape = checkpoint['deep_layers.0.weight'].shape
                # First dimension is output dim (256), second is input dim (what we need)
                total_input_dim = first_layer_shape[1]
                
                # Calculate the correct input_dim based on the formula in SharedSelfAttentionCLSModel.__init__
                # prev_dim = (self.input_dim // self.token_encoder_dim * self.encoder_output_dim) * 2 + 2 * self.encoder_output_dim
                token_encoder_dim = 30
                encoder_output_dim = 10
                
                # Solving for input_dim
                # total_input_dim = (input_dim // token_encoder_dim * encoder_output_dim) * 2 + 2 * encoder_output_dim
                # (total_input_dim - 2 * encoder_output_dim) / 2 = input_dim // token_encoder_dim * encoder_output_dim
                # input_dim = token_encoder_dim * ((total_input_dim - 2 * encoder_output_dim) / 2) / encoder_output_dim
                
                input_dim = token_encoder_dim * ((total_input_dim - 2 * encoder_output_dim) / 2) / encoder_output_dim
                input_dim = int(input_dim) * 2  # input_dim in the model is divided by 2 during forward pass
                print(f"Inferred input_dim: {input_dim} from first layer shape: {first_layer_shape}")
            else:
                # Default to using the provided input dimension if we can't infer it
                input_dim = X_source.shape[1] * 2
                print(f"Using provided input_dim: {input_dim}")
            
            # Create the model with our best estimate of the architecture
            model = SharedSelfAttentionCLSModel(
                input_dim=input_dim,
                binarize=False,
                token_encoder_dim=30,
                d_model=128,
                encoder_output_dim=10,
                nhead=4,
                num_layers=4,
                deep_hidden_dims=[256, 128],
                cls_init='spatial_learned',
                use_alibi=False,
                transformer_dropout=0.2,
                dropout_rate=0.2,
                learning_rate=0.00009,
                weight_decay=0.0001,
                batch_size=512,
                aug_prob=0.0
            )
        
        # Load the state dict
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Assume the checkpoint itself is the state dict
            model.load_state_dict(checkpoint)
    else:
        # The checkpoint is the full model
        model = checkpoint
    
    model.eval()  # Set to evaluation mode
    
    # Create the multi-resolution dataset
    multi_res_data = create_multi_resolution_dataset(
        cv_indices_dir=cv_indices_dir,
        source_parcellation=source_parcellation,
        target_parcellation=target_parcellation,
        X_source=X_source,
        Y_source=Y_source,
        X_target=X_target,
        Y_target=Y_target,
        coords_source=coords_source,
        coords_target=coords_target,
        fold=fold,
        distance_threshold=distance_threshold
    )
    
    # Get datasets and indices
    source_dataset = multi_res_data['source_dataset']
    target_dataset = multi_res_data['target_dataset']
    source_train_indices = multi_res_data.get('source_train_indices', [])
    source_test_indices = multi_res_data.get('source_test_indices', [])
    target_train_indices = multi_res_data.get('target_train_indices', [])
    target_test_indices = multi_res_data.get('target_test_indices', [])
    
    # Function to compute evaluation metrics
    def compute_metrics(y_true, y_pred):
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'pearson_corr': pearsonr(y_true, y_pred)[0]
        }
        return metrics
    
    # Determine the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move model to the right device
    print(f"Using device: {device}")
    
    # Create source test dataloader
    source_test_dataset = Subset(source_dataset, source_test_indices)
    source_test_loader = DataLoader(source_test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # Create target test dataloader
    target_test_dataset = Subset(target_dataset, target_test_indices)
    target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # Evaluate on source dataset using predict method
    model.eval()
    # Make sure model's device knowledge is updated before prediction
    model.device = device  # Update model's device attribute
    Y_pred_source, Y_test_source = model.predict(source_test_loader)
    source_metrics = compute_metrics(Y_test_source, Y_pred_source)
    
    # Evaluate on target dataset using predict method
    model.eval()
    Y_pred_target, Y_test_target = model.predict(target_test_loader)
    target_metrics = compute_metrics(Y_test_target, Y_pred_target)
    
    # Return the metrics for both datasets
    return {
        'source_parcellation': source_parcellation,
        'target_parcellation': target_parcellation,
        'source_metrics': source_metrics,
        'target_metrics': target_metrics,
        'source_predictions': Y_pred_source,
        'target_predictions': Y_pred_target,
        'source_ground_truth': Y_test_source,
        'target_ground_truth': Y_test_target,
        'model_path': model_checkpoint_path
    }


if __name__ == "__main__":
    # Define the command line argument parser
    parser = argparse.ArgumentParser(description='Map between iPA parcellation resolutions')
    parser.add_argument('--source', type=str, required=True, help='Source parcellation (e.g., iPA_183)')
    parser.add_argument('--target', type=str, required=True, help='Target parcellation (e.g., iPA_391)')
    parser.add_argument('--distance_threshold', type=float, default=15.0, help='Distance threshold in mm (default: 15.0)')
    parser.add_argument('--output_dir', type=str, default='mappings', help='Output directory for mapping files')
    parser.add_argument('--cv_indices', type=str, help='Path to CV indices file for mapping')
    parser.add_argument('--model_checkpoint', type=str, help='Path to a model checkpoint to evaluate on multiple resolutions')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create the mapping between the specified parcellations
    mapping, distances, threshold = create_mapping_between_parcellations(
        args.source, args.target, args.distance_threshold)
    
    # Save the mapping to a CSV file
    output_file = os.path.join(args.output_dir, f"{args.source}_to_{args.target}_mapping.csv")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_index', 'target_indices'])
        for source_idx, target_indices in mapping.items():
            writer.writerow([source_idx, ','.join(map(str, target_indices))])
    
    print(f"Mapping saved to {output_file}")
    
    # Print mapping statistics
    targets_per_source = [len(v) for v in mapping.values()]
    print(f"Mapping statistics:")
    print(f"  Min targets per source: {min(targets_per_source)}")
    print(f"  Max targets per source: {max(targets_per_source)}")
    print(f"  Avg targets per source: {np.mean(targets_per_source):.2f}")
    print(f"  Total source regions: {len(mapping)}")
    print(f"  Total target regions: {sum(targets_per_source)}")
    
    # If CV indices are provided, map them to the target parcellation
    if args.cv_indices:
        print(f"\nMapping CV indices from {args.cv_indices}...")
        result = map_cv_indices_between_parcellations(
            args.cv_indices, args.source, args.target)
        
        # Save the mapped indices
        cv_type = 'train' if 'train_indices' in result['mapped_indices'] else 'test'
        mapped_indices = result['mapped_indices'].get(f'{cv_type}_indices', [])
        
        output_file = os.path.join(
            args.output_dir, 
            f"{os.path.basename(args.cv_indices).replace('.csv', '')}_mapped.csv")
        
        pd.DataFrame({
            f"{cv_type}_index": mapped_indices, 
            'region_id': mapped_indices
        }).to_csv(output_file, index=False)
        
        print(f"Mapped indices saved to {output_file}")
        print(f"  Original {cv_type} indices: {len(result['mapped_indices'].get(f'{cv_type}_indices', []))}")
        print(f"  Mapped {cv_type} indices: {len(mapped_indices)}")
