#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of functions for evaluation of applicability domain 
of ML models

Created on Thu May 13 15:43:56 2021

@author: Dr. Freddy Bernal
"""

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cdist


#%% FUNCTIONS BASED ON COSINE DISTANCES

# Function to calculate AD by novelty measurement
# adapted from Klingspohn et al. J Cheminform (2017) 9:44

def cos_calc(X, cmp_id, k):
    """
    Defines cosine values between a selected compound from training set 
    and its k neighbors.

    Parameters
    ----------
    X : Numpy Array
        Training set X data.
    cmp_id : int
        Index (row) of selected compound to compare with the rest.
    k : int
        Number of neighbors to consider during cosine averaging.

    Returns
    -------
    res : Numpy Array
        Cosine distances of selected compound to its k neighbors.
    """
    
    # Calculate cosine distances
    cos = cosine_distances(X)
    # re-organize output
    cosl = list()
    for idx, distv in enumerate(cos[cmp_id]):
        cosl.append((idx, distv))
    # Sort by distance
    cosl.sort(key=lambda tup: tup[1]) # this lambda key ensures the sorting is done by distance (second item)

    cosl = np.array(cosl)
    # Retrieve the cosine values from the specified neighbors
    res = cosl[1:k + 1, 1] # take out first value (itself) until k neighbors and without indexes
    
    return res



# Create a function to calculate the AD by novelty measurement
def cos_calc_test(X, k):
    """
    Defines cosine values between a selected compound from test set and 
    its k neighbors.

    Parameters
    ----------
    X : Numpy Array
        Test set X data.
    k : int
        Number of neighbors to consider during cosine averaging.

    Returns
    -------
    res : Numpy Array
        Cosine distances of selected compound to its k neighbors.
    """
    
    # Calculate cosine distance
    cos = cosine_distances(X)
    # re-organize output
    cosl = list()
    for idx, distv in enumerate(cos[-1]):
        cosl.append((idx, distv))
    # Sort by distance
    cosl.sort(key=lambda tup: tup[1]) # this lambda key ensures the sorting is done by distance (second item)

    cosl = np.array(cosl)
    # # Retrieve the cosine values from the specified neighbors
    res = cosl[1:k + 1, 1] # take out first value (itself) until k neighbors and without indexes
    
    return res


def cos_ad_novelty(X_train, X_test, k=3, pct=0):
    """
    Defines whether a compound is within the AD of a model based on the average 
    of cosine distances to k neighbors 

    Parameters
    ----------
    X_train : Numpy Array
        Training set X data.
    X_test : Numpy Array
        Test set X data.
    k : int
        Number of neighbors to consider during cosine averaging.
        The default is 3.
    pct : int
        Percetile to be used to define the threshold. The default is 0.
        The threshold is then calculated from the 25th and 75th percentiles.

    Returns
    -------
    inad : Numpy Array
        categorical definition of test samples as inside (1) or outside (0) 
        the AD defined by means of cosine distances from k neighbors.
    cthr : float
        Cosine threshold.
    Ctest : Numpy Array
        Cosine distances of test samples from their k neighbors.
    """
    
    # Calculate average cosine distance of each sample in the training set
    # with its k nearest neighbors
    C = np.zeros((len(X_train), 1))
    for i in range(len(X_train)):
        d = cos_calc(X_train, i, k)
        C[i] = np.mean(d)

    # Calculate average cosine distance of each sample in the test set
    # with its k nearest neighbors from the training set
    Ctest = np.zeros((len(X_test), 1))
    for i, x in enumerate(X_test):
        tmp = X_train.copy()
        tmp = np.vstack((tmp, x))    
        d = cos_calc_test(tmp, k)
        Ctest[i] = np.mean(d)
    
    # Define threshold distance (for AD inclusion/exclusion)
    UL, Qperc = calc_thr_prctile(C, pct)
    if pct == 0: # If no specific percentile given
        cthr = UL
    else: # If desired percentile given
        cthr = Qperc
    
    # Compile results in Array
    inad = np.ones((len(X_test), 1))
    inad[Ctest > cthr] = 0

    return inad, cthr, Ctest
    
   
    
def calc_thr_prctile(vtrain, perc):
    """
    Calculates threshold for AD definition as percentile 75th + 1.5 (percentile
    75th - percentile 25th).

    Parameters
    ----------
    vtrain : Numpy Array
        Data matrix to get percentiles from.
    perc : int
        Percetile to be used to define the threshold.

    Returns
    -------
    UL : float
        Threshold obtained from the 25th and 75th percentiles.
    prc : float
        Threshold obtained from selected percentile 'perc'.
    """
    
    Q = np.percentile(vtrain, [25, 75, perc])
    UL = Q[1] + 1.5 * (Q[1] - Q[0])
    prc = Q[2]
    return UL, prc


#%% FUNCTIONS BASED ON LEVERAGE 


# Adapted version from Milano group (MATLAB scripts)

def leverage_ad(X_train, X_test, thr=2.5):
    """
    Defines whether a compound is within the AD of a model based on the leverage 
    method

    Parameters
    ----------
    X_train : Pandas DataFrame
        Training set X data.
    X_test : Pandas DataFrame
        Test set X data.
    thr : float
        factor to be multiplied with the leverage average in order to define 
        the leverage threshold. The default is 2.5.

    Returns
    -------
    inad : Numpy Array
        categorical definition of test samples as inside (1) or outside (0) 
        the AD.
    h_star : float
        Leverage threshold.
    Hcore : Numpy Array
        Leverage core matrix.
    Htrain : Numpy Array
        Leverages of training samples.
    Htest : Numpy Array
        Leverages of test samples.
    """
    
    # Calculate leverages for training compounds
    Hcore = np.linalg.inv(X_train.T.dot(X_train))
    Htrain = X_train.dot(Hcore.dot(X_train.T))
    
    # Calculate leverages for test compounds referred to the training set
    Htest = X_test.dot(Hcore.dot(X_test.T))
    
    # Extract diagonal (hat)
    Htrain = np.diag(Htrain)
    Htest = np.diag(Htest)
    
    # Define the threshold h*
    have = sum(Htrain)
    h_star = thr * (have + 1) / len(X_train)
    
    # Establish presence within AD and compile in array
    inad = np.ones((len(X_test), 1))
    inad[Htest > h_star] = 0
    
    return inad, h_star, Hcore, Htrain, Htest    



#%% FUNCTIONS BASED ON k-neighbors 


# Adapted version from Milano group (MATLAB scripts)
# Function adapted without considering mahalanobis distances, thus "if" removed
def knn_ad(X_train, X_test, k, dist_pct=95):
    """
    Defines whether a compound is within the AD of a model based on the distance 
    to the k neighbors

    Parameters
    ----------
    X_train : Pandas DataFrame
        Training set X data.
    X_test : Pandas DataFrame
        Test set X data.
    k : int
        Number of neighbors to be considered.
    dist_pct : int
        Percetile to be used for threshold definition. 
        The default is 95 (suggested).

    Returns
    -------
    inad : Numpy Array
        Categorical definition of test samples as inside (1) or outside (0) 
        the AD.
    thr : float
        Distance threshold.
    Dtest : Numpy Array
        Distances of test samples to their k neighbors.
    """
    
    # Calculate average Euclidean distance for each sample in training set 
    # with its k nearest neighbors
    Dtrain = cdist(X_train, X_train, metric='euclidean')
    Dtrain.sort()
    Dtrain_mean = Dtrain[:, 1:k].mean(axis=1)

    # Calculate average Euclidean distance for each sample in test set 
    # with its k nearest neighbors from the training set
    Dtest = cdist(X_train, X_test, metric='euclidean').T
    Dtest.sort()
    Dtest_mean = Dtest[:, :k].mean(axis=1)
    
    # Define threshold distance (for AD inclusion/exclusion)
    UL, Qperc = calc_thr_prctile(Dtrain_mean, dist_pct)
    if dist_pct == 0:
        thr = UL
    else:
        thr = Qperc

    # Establish presence within AD and compile in array    
    inad = np.ones((len(X_test), 1))
    inad[Dtest_mean > thr] = 0

    return inad, thr, Dtest
    
    
#%% GLOBAL FUNCTIONS 

# AD definition by the three methods

def AD_definition(X_train, X_test, ad_params):
    """
    Runs AD calculations with three methods: cosine, leverage and k-neighbors. 
    Return a unique array.

    Parameters
    ----------
    X_train : Pandas DataFrame
        Training set X data.
    X_test : Pandas DataFrame
        Test set X data.
    ad_params : dict
        Specified number of neighbors and percentile to be used for threshold 
        calculation.

    Returns
    -------
    inad_all : Numpy Array
        Categorical definition of test samples as inside (1) or outside (0) 
        the AD by each method.
    """
    
    # AD by cosine distances
    inad_cos, _ , _ = cos_ad_novelty(X_train.to_numpy(), 
                                     X_test.to_numpy(), 
                                     k=ad_params['neighbors'], 
                                     pct=ad_params['percentile'])

    # AD by leverages
    inad_lev, _ , _ , _ , _ = leverage_ad(X_train, X_test, 
                                          thr=ad_params['lev_thr'])

    # fixed kNN (Euclidean distances)
    inad_fk, _ , _ = knn_ad(X_train, X_test, 
                            k=ad_params['neighbors'], 
                            dist_pct=ad_params['percentile'])


    # Compile all the information in an array
    inad_all = np.hstack((inad_cos, inad_lev, inad_fk))
    
  
    return inad_all   

    

# Function for majority vote of ADs
def calculate_vote(votes, weights=None, threshold=0.5):
    """
    Runs a majority voting on AD results

    Parameters
    ----------
    votes : Numpy Array
        Matrix with 1 (inside) and 0 (outside) AD by different methods obtained 
        from the function AD_definition.
    weights : list, optional
        Wieghts to be considered during voting. The default is None. In this
        case all votes are weighted equally.
    threshold : float
        Maximum threshold as to be considered within AD. The default is 0.5.

    Returns
    -------
    finalVote : int
        Final categorization from all different methods considered. A test sample
        is defined as inside (1) AD, if the weighted average from the methods 
        is higher than the threshold. Otherwise, the sample is marked as being 
        outside (0)
    votingSum : float
        Weighted average of indenpendent votes from each AD method.
    """
    
    # Weights can be given if certain AD method is preferred. Otherwise, all
    # AD methods are given the same weight.
    if weights is None: 
        weights = np.full((1, votes.shape[0]), 1 / votes.shape[0])
    votingSum = sum(map(lambda x: x[0] * x[1], zip(votes, weights[0])))
    if votingSum > threshold: # A different threshold can be given
        finalVote = 1
    else:
        finalVote = 0
        
    return finalVote, votingSum



# Function to define whether a compound is or not within AD of the model
def full_ad_determination(X_train, Xnew_scaled, ad_params, ynew=None):
    """
    Determine whether a new compound is within the applicability domain of a 
    training set using the Cosine, the Leverage and the k-nearest neighbors 
    methods. It uses AD_definition and calculate_vote.

    Parameters
    ----------
    X_train : Pandas DataFrame
        Training set X data.
    Xnew_scaled : Pandas DataFrame
        X data for the set of new compounds.
    ad_params : dict
        Specified number of neighbors and percentile to be used for threshold 
        calculation.
    ynew : Pandas Series or Numpy Array, optional
        Activity data for the set of new compounds. The default is None.

    Returns
    -------
    Xnew_red : Pandas DataFrame
        Filtered and scaled data set for new compounds within AD. Preprocessing 
        as done for training set.
    ynew_red : Pandas Series or Numpy Array
        Activity class for new compounds within AD (if ynew provided)
    inad_voting : list
        Final AD compliance decision considering the three methods
    inad : Numpy Array
        Categorical definition of test samples as inside (1) or outside (0) 
        the AD by each method. Obtained from function AD_definition
    mask : Numpy Array
        Indices of samples within AD as indicated by majority voting (in inad_voting)
    """

    # Run AD determination by the three methods
    inad = AD_definition(X_train, Xnew_scaled, ad_params)
    
    # Calculate final AD decision looping over each inad
    inad_voting = []
    sum_voting = []
    for i in inad:
        fv, vs = calculate_vote(i)
        inad_voting.append(fv)
        sum_voting.append(vs)
        
    # Check compliance with AD and remove compounds laying outside AD
    mask = np.where(np.array(inad_voting) == 1)[0]
    Xnew_red = Xnew_scaled.iloc[mask, :]

    # If activity class known (validation purposes), remove data of 
    # compounds outside AD
    if ynew is not None:
        if isinstance(ynew, pd.core.series.Series):
            ynew.reset_index(drop=True, inplace=True)
            ynew_red = ynew.iloc[mask]
        elif isinstance(ynew, np.ndarray):
            ynew_red = ynew[mask]
    
        return Xnew_red, ynew_red, inad_voting, inad, mask
    
    else:
        return Xnew_red, inad_voting, inad, mask


