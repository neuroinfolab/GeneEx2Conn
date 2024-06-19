# Gene2Conn/test_main.py

def main():
    X, Y = load_transcriptome(), load_connectome()

    print(X.shape)
    print(Y.shape)
    
    shuffled_cv_obj = RandomCVSplit(X, Y, num_splits=4, shuffled=True, use_random_state=True)
    shuffled_cv_obj.display_splits()

    subnetwork_cv_obj = SchaeferCVsplit()
    subnetwork_cv_obj.print_folds_with_networks()

# Example driver code to get a model and parameter grid
if __name__ == "__main__":
    model_type = 'xgboost'  # Replace with the desired model type
    model = ModelBuild.get_model(model_type)
    print("Model:", model.get_model())
    print("Parameter Grid:", model.get_param_grid())

'''
def run_innercv(X_train, Y_train, X, Y, train_indices, test_indices, train_network_dict, split_params):
    # unpack split params
    use_shared_regions, include_conn_feats, test_shared_regions = split_params

    # Create inner CV object (just indices) for X_train and Y_train
    inner_cv_obj = SubnetworkCVSplit(train_indices, train_network_dict)
    print(inner_cv_obj.get_n_splits()) 
    
    # Process cv splits with the same procedure as the original train-test split, return all expanded folds
    inner_fold_splits = process_cv_splits(X, Y, inner_cv_obj, use_shared_regions, include_conn_feats, test_shared_regions)
    
    # Create combined split object that can be indexed for cross-validation
    X_combined, Y_combined, train_test_indices = expanded_inner_folds_combined_plus_indices(inner_fold_splits)
    
    # Initialize model
    model = ModelBuild.init_model('xgboost')
    print("Model:", model.get_model())
    param_grid = model.get_param_grid()
    print("Parameter Grid:", param_grid)

    # convert to cp array when being gpu accelerated
    X_combined = cp.array(X_combined)
    Y_combined = cp.array(Y_combined)
    
    # Setup GridSearchCV with custom scorer
    scoring = 'neg_mean_squared_error'
    cupy_scorer = make_scorer(pearson_cupy, greater_is_better=True) # make sure to change greater_is better param appropriately 
    grid_search = GridSearchCV(model.get_model(), param_grid, cv=train_test_indices, scoring=cupy_scorer, verbose=2)
    
    # Fit GridSearchCV on the combined data
    grid_search.fit(X_combined, Y_combined) # should refit param be modified here?
    
    # Get the best estimator
    best_estimator = grid_search.best_estimator_

    # Display comprehensive results
    print("\nGrid Search CV Results:")
    print("=======================")
    print("Best Parameters: ", grid_search.best_params_)
    print("Best Cross-Validation Score: ", grid_search.best_score_)

    # Calculate performance on each fold, skip for now due to memory constraints
    # or consider switching model and data back to CPU

    print("\nPerformance on each fold for best model:")
    for fold_idx, (train_idx, test_idx) in enumerate(train_test_indices):
        X_train_fold, X_test_fold = X_combined[train_idx], X_combined[test_idx]
        Y_train_fold, Y_test_fold = Y_combined[train_idx], Y_combined[test_idx]

        print(f"\nFold {fold_idx + 1}:")
        evaluator = ModelEvaluator(best_estimator, X_train_fold, Y_train_fold, X_test_fold, Y_test_fold, split_params)
        train_metrics = evaluator.get_train_metrics()
        test_metrics = evaluator.get_test_metrics()
        print("Train Fold Metrics:", train_metrics)
        print("Test Fold Metrics:", test_metrics)
    
    return best_estimator
    


def drop_dict_entry_by_value(d, value):
    """Drop an entry from the dictionary based on the given value and return a new dictionary."""
    # Create a copy of the original dictionary
    new_dict = d.copy()
    
    # Identify the keys to remove
    keys_to_remove = [key for key, val in new_dict.items() if val == value]
    
    # Remove the identified keys from the new dictionary
    for key in keys_to_remove:
        del new_dict[key]

    return new_dict

def evaluate_folds(X, Y, fold_splits, cv_obj, split_params):
    for i in range(len(fold_splits)):
        print('Test fold num: ', i)
        X_train, X_test, Y_train, Y_test = fold_splits[i][0], fold_splits[i][1], fold_splits[i][2], fold_splits[i][3]
        print(X_train.shape)
        print(Y_train.shape)
        print(X_test.shape)
        print(Y_test.shape)
        
        train_indices = cv_obj.folds[i][0]
        test_indices = cv_obj.folds[i][1]
        
        network_dict = cv_obj.networks
        train_network_dict = drop_dict_entry_by_value(network_dict, test_indices)
    
        best_model = run_innercv(X_train, Y_train, X, Y, train_indices, test_indices, train_network_dict, split_params)

        X_train = cp.array(X_train)
        Y_train = cp.array(Y_train)
        X_test = cp.array(X_test)
        Y_test = cp.array(Y_test)
        
        # Train the model on the training data
        best_model.fit(X_train, Y_train)
        
        evaluator = ModelEvaluator(best_model, X_train, Y_train, X_test, Y_test, split_params)
        train_metrics = evaluator.get_train_metrics()
        test_metrics = evaluator.get_test_metrics()
        print("\n Train Metrics:", train_metrics)
        print("Test Metrics:", test_metrics)

        print_system_usage()
        
def run_sim():
    # load data
    X, Y = load_transcriptome(), load_connectome()

    # random split
    shuffled_cv_obj = RandomCVSplit(X, Y, num_splits=4, shuffled=True, use_random_state=True)    
    print(shuffled_cv_obj.folds[0])
    print(shuffled_cv_obj.networks)

    # community based split 
    community_cv_obj = CommunityCVSplit(X, Y, resolution=1.0)
    print(community_cv_obj.folds[0])
    print(community_cv_obj.networks)
    community_cv_obj.display_communities

    # network based split
    subnetwork_cv_obj = SchaeferCVSplit()
    print(subnetwork_cv_obj.folds[0]) 
    print(subnetwork_cv_obj.networks) 

    # select cv object 
    cv_obj = subnetwork_cv_obj
    
    # train/test parameters
    use_shared_regions = False # train on non-shared and shared connections between train and test set
    include_conn_feats = False # including connectivity profiles as regional input features
    test_shared_regions = False # add shared regions to the test set
    split_params = [use_shared_regions, include_conn_feats, test_shared_regions]

    # expand data
    fold_splits = process_cv_splits(X, Y, cv_obj, use_shared_regions, include_conn_feats, test_shared_regions) # process_cv_splits_conn_only_model()

    # outer CV on folds
    evaluate_folds(X, Y, fold_splits, cv_obj, split_params)
    
run_sim()
'''