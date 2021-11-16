#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of functions for ML-based classification models

Created on Thu May 13 15:43:56 2021

@author: Dr. Freddy Bernal
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_validate

from sklearn.model_selection import RandomizedSearchCV

from preprocessing import preprocessing

#%% Function to statistically evaluate the performance of models using cross-validation

def model_eval(model, X_train, X_test, y_train, y_test):
    """
    Assess the performance of a generated ML model using an stratified 5-fold 
    CV with 10 repetitions for training and direct evaluation on test set. 
    Four different metrics are considered: ROC AUC, balanced accuracy, f1 score,
    and Matthews Correlation Coefficient. Based on sklearn functions.

    Parameters
    ----------
    model : sklearn obj
        Instantiated ML model.
    X_train : Pandas DataFrame
        Filtered and preprocessed descriptor data for training set.
    X_test : Pandas DataFrame
        Filtered and preprocessed descriptor data for test set. 
    y_train : Pandas Series
        Activity data for training set.
    y_test : Pandas Series
        Activity data for test set.

    Returns
    -------
    stats : list
        statistical parameters from evaluation of the model in the following 
        order: ROC AUC, balanced accuracy, f1 score, and MCC for training 
        (mean values from the 10-repeated stratified 5-fold CV), and ROC AUC, 
        balanced accuracy, f1 score, and MCC for test set predictions.
    """    
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Statistics on TEST SET (external evaluation)
    
    # Predict activity class on test set
    y_test_pred = model.predict(X_test)    
    
    # Calculate probabilities or decisions depending on the ML algorithm
    try:
        pp = model.predict_proba(X_test)[:,1]
    except AttributeError:
        pp = model.decision_function(X_test)
    
    # Generate ROC curve and calculate AUC
    fpr, tpr, _ = roc_curve(y_test, pp)
    roc_auc_t = auc(fpr, tpr)
    # Calculate balanced accuracy
    ba_t = balanced_accuracy_score(y_test.to_numpy(), y_test_pred)
    # Calculate f1
    f1_t = f1_score(y_test.to_numpy(), y_test_pred)
    # Calculate MCC
    mcc_t = matthews_corrcoef(y_test.to_numpy(), y_test_pred)

    # Statistics on TRAINING SET (internal evaluation=CV)

    # Perform an stratified 5-fold CV with 10 repetitions
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=21)
    roc_auc_cv = []
    ba_cv = []
    f1_cv = []
    mcc_cv = []
    for train_ix, test_ix in cv.split(X_train, y_train):
        # split training data
        Xtr, Xt = X_train.iloc[train_ix, :], X_train.iloc[test_ix, :]
        ytr, yt = y_train.iloc[train_ix], y_train.iloc[test_ix]
        # Fit model 
        cv_m = model.fit(Xtr, ytr)
        # Evaluate model on the hold out dataset
        yhat = cv_m.predict(Xt)
        # Calculate probabilities or decisions
        try:
            pp_cv = model.predict_proba(Xt)[:,1]
        except AttributeError:
            pp_cv = model.decision_function(Xt)
        # Calculate AUC from ROC curve
        fprcv, tprcv, _ = roc_curve(yt, pp_cv)
        roc_auc = auc(fprcv, tprcv)
        roc_auc_cv.append(roc_auc)
        # Calculate balanced accuracy
        acc = balanced_accuracy_score(yt, yhat)
        ba_cv.append(acc)
        # Calculate f1
        f1 = f1_score(yt, yhat)
        f1_cv.append(f1)
        # Calculate mcc
        mcc = matthews_corrcoef(yt, yhat)
        mcc_cv.append(mcc)

        # To calculate mcc without getting warnings:  
        # import warnings
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore')
        #     mcc = matthews_corrcoef(yt, yhat)
        #     mcc_cv.append(mcc)

    # Combine measured statistics into list
    stats = [np.mean(roc_auc_cv), np.mean(ba_cv), np.mean(f1_cv), np.mean(mcc_cv), 
             roc_auc_t, ba_t, f1_t, mcc_t]
    
    return stats


# %% Function to create a model from optimized hyperparameters

def create_model(name, hyperparam, seed):
    """
    Creates a ML model from a set of hyperparameters stored during optimization

    Parameters
    ----------
    name : str
        Type of model as rf, svm, knn, or ada, for random forest, support vector 
        machines, k-nearest neighbors, and adaboost, respectively.
    hyperparam : Pandas DataFrame
        Optimized hyperparameters obtained after optimization by Randomized 
        search with sklearn.
    seed : int
        Random seed/numeric id used to identify the model during hyperparameter
        optimization.

    Returns
    -------
    model : sklearn obj
        Instantiated model with best hyperparameters set.
    """
    
    # Instantiate model using hyperparameters speficied in hyperparam dict
    if name == 'rf':
        model = RandomForestClassifier(n_estimators=hyperparam.n_estimators[seed],
                                       min_samples_split=hyperparam.min_samples_split[seed],
                                       min_samples_leaf=hyperparam.min_samples_leaf[seed],
                                       max_features=hyperparam.max_features[seed],
                                       random_state=21)
        
    elif name == 'svm':
        model = SVC(kernel=hyperparam.kernel[seed],
                    gamma=hyperparam.gamma[seed],
                    C=hyperparam.C[seed])
        
    elif name == 'knn':
        model = KNeighborsClassifier(n_neighbors=hyperparam.n_neighbors[seed],
                                     leaf_size=hyperparam.leaf_size[seed],
                                     algorithm=hyperparam.algorithm[seed],
                                     weights=hyperparam.weights[seed])
        
    elif name == 'ada':
        # retrieve base estimator (not always Decision Tree as happend with desc)
        m = hyperparam.base_estimator[seed].split('(')
        if m[0] == 'LogisticRegression':
            estimator = LogisticRegression(random_state=21)
        elif m[0] == 'DecisionTreeClassifier':
            estimator = DecisionTreeClassifier(max_depth=int(m[1].split(',')[0][-1]), random_state=21)
        # Recreate the model
        model = AdaBoostClassifier(n_estimators=hyperparam.n_estimators[seed],
                                   learning_rate=hyperparam.learning_rate[seed],
                                   base_estimator=estimator,
                                   random_state=21)
        
    
    else:
        print('Model name not recognized')
    
    return model


#%% Function for loading optimized hyperparameters, build models and evaluate them.

def model_stats(filenamescsv, seed, X, y, preproc):
    """
    Loads hyperparameters from optimization, build models, and assess their 
    performance. Used for analysis of several models at once.

    Parameters
    ----------
    filenamescsv : list
        File names of csv files with best hyperparameters after randomized search.
    seed : list
        Random seed/numeric id (int) used to identify models during optimization.
    X : Pandas DataFrame
        Data matrix of compounds x features.
    y : Pandas DataFrame or Series
        Activity values.
    preproc : str
        Preprocessing method. For fingerprints set select 'split'. 
        For descriptors set select 'filter'. In this case, the function
        feature_filt is used.        

    Returns
    -------
    stats : Pandas DataFrame
        Statistical parameters from evaluation of the model including algortihm
        used for bulding the model and random search/ID.
    """
    
    # Define dataframe to store statistics
    stats = pd.DataFrame(columns=['AUC_cv', 'BA_cv', 'f1_cv', 'MCC_cv',
                                  'AUC_t', 'BA_t', 'f1_t', 'MCC_t',
                                  'algorithm', 'seed'])
    
    # Loop over list of seeds
    for i in seed:
        
        # Split data into training and test according to selected
        # preprocessing method
        X_train, X_test, y_train, y_test = preprocessing(X, y, preproc, i)

        # Create an empty dictionary for storing hyperparameter information 
        # for all the algorithms analyzed
        hyperparam = {}
        # Iterate over csv files to get best hyperparameters for each model
        for filename in filenamescsv:
            d = pd.read_csv(filename, index_col=0)
            lab = filename.split('_')[2].split('.')[0]
            hyperparam[lab] = d.loc[[i]]
        
        # Iterate over sets of hyperparameters and run statistical evaluation 
        # of models 
        for alg in hyperparam.keys():
            # Create model
            model = create_model(alg, hyperparam[alg], i)
            # Statistical assessment
            s = model_eval(model, X_train, X_test, y_train, y_test)
            # Compilation of results
            stats.loc[len(stats)] = s + [alg, i]
    
    
    return stats


#%% Functions to perform hyperparameter optimization search 

def optimization_rands(model, X_train, X_test, y_train, y_test, params_dict, 
                      scoring, iterations=2000):
    """
    Performs randomized search (sklearn) and evaluates predictive power of the 
    best model using the test set. A 3-times repeated stratified 5-fold CV is 
    used during the search.

    Parameters
    ----------
    model : sklearn obj
        Instantiated model.
    X_train : Pandas DataFrame
        Filtered and preprocessed descriptor data for training set.
    X_test : Pandas DataFrame
        Filtered and preprocessed descriptor data for test set. 
    y_train : Pandas Series
        Activity data for training set.
    y_test : Pandas Series
        Activity data for test set.
    params_dict : dict
        Hyperparameters and their values to be used during optimization search.
    scoring : str
        Scoring for selection of best hyperparameters set.
    iterations : int
        Number of iterations to perform during randomized search. 
        The default is 2000.

    Returns
    -------
    best_params : dict
        Set of best hyperparameters.
    val : list
        Statistical performance of the model using the hyperparameters found. 
        It contains the sklearn score during randomized search, accuracy and f1
        score from predictions on the test set, and the mean value of accuracy
        and f1 score from sklearn function crossvalidate
    """
    
    # Configure cross-validation procedure
    cv_inner = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=21)
    
    # Instantiate the randomized search
    rand_search = RandomizedSearchCV(model, param_distributions=params_dict, 
                                     n_iter=iterations, scoring=scoring, 
                                     cv=cv_inner, verbose=1, n_jobs=-1,
                                     random_state=21)
    
    # Run the search
    rand_search.fit(X_train.to_numpy(), y_train.to_numpy())
    # Collect data
    score = rand_search.best_score_
    best_params = rand_search.best_params_
    # Uses best model to predict values for test set
    y_test_pred = rand_search.predict(X_test.to_numpy())
    
    # Statistics for TEST SET
    # Calculate accuracy
    acc = accuracy_score(y_test.to_numpy(), y_test_pred)
    # Calculate f1
    f1 = f1_score(y_test.to_numpy(), y_test_pred)

    # Perform 5-fold CV with cross_val_score (TRAINING SET)
    scoring_cv = ['accuracy', 'f1']
    cv_scores = cross_validate(model, X_train, y_train, scoring=scoring_cv,
                               cv=cv_inner)

    
    
    # store data in a list
    val = [score, acc, f1, cv_scores['test_accuracy'].mean(), 
             cv_scores['test_f1'].mean()]
    
    
    return best_params, val



# Function to iterate over different training/test set compositions and search for 
# optimal set of hyperparameters

def iter_rand_search(model, X, y, preproc, seed, iterations, param_dict):
    """
    Performs iterative randomized search to obtain the best set of hyperparameters
    of a group of models. It uses preprocessing and optimization_rands.

    Parameters
    ----------
    model : sklearn obj
        Instantiated model.
    X : Pandas DataFrame
        Data matrix of compounds x features.
    y : Pandas DataFrame or Series
        Activity values.
    preproc : str
        Preprocessing method. For fingerprints set select 'split'. 
        For descriptors set select 'filter'. In this case, the function
        feature_filt is used.        
    seed : list
        Random seed/numeric id (int) used to identify models during optimization.
    iterations : int
        Number of iterations to be considered for randomized search. 
    params_dict : dict
        Hyperparameters and their values to be used during optimization search.

    Returns
    -------
    result_df : Pandas DataFrame
        Set of best hyperparameters and statistical results of evaluation of 
        the model.
    """
    
    # Create empty dataframe and list to store results
    stats = pd.DataFrame(columns=['score', 'Accuracy', 'f1_score',
                                  'Acc_CV', 'f1_CV'])
    bestpar = []

    # Loop over different training/test set compositions
    for i in seed:
        # Split data into training/test set 
        X_train, X_test, y_train, y_test = preprocessing(X, y, preproc, i)
        # Run hyperparameter search
        bp, val = optimization_rands(model, X_train, X_test, y_train, y_test, 
                                             param_dict, scoring='f1', 
                                             iterations=iterations)
        # Collect results
        bestpar.append(bp)
        stats.loc[len(stats)] = val
        
    # Compile information
    bestpar = pd.DataFrame(bestpar, index=[i for i in seed])
    stats.index = [i for i in seed]

    result_df = pd.concat((bestpar, stats), axis=1, sort=False)
    
    return result_df


#%% Function for statistical assessment of external set (scope of model)

def scoring_new_class(y, ypred, pp):
    """
    Calculate statistical parameters for predictions of new reported compounds 
    (validation purposes).

    Parameters
    ----------
    y : Numpy Array
        Measured (reported) activity class.
    ypred : Numpy Array
        Predicted activity class.
    pp : Numpy Array
        Predicted probability.

    Returns
    -------
    stats : list
        Statistical parameters in the following order: ROC AUC, Brier score, 
        BA, f1 score, and MCC.
    """
    
    # Generate ROC curve and calculate AUC
    fprcv, tprcv, _ = roc_curve(y, pp)
    rocauc = auc(fprcv, tprcv)

    # Calculate Brier score
    brier = brier_score_loss(y, pp)

    # Calculate other metrics
    ba = balanced_accuracy_score(y, ypred)
    f1 = f1_score(y, ypred)
    mcc = matthews_corrcoef(y, ypred)
    
    # Compile results in a list
    stats = [rocauc, brier, ba, f1, mcc]
    
    return stats