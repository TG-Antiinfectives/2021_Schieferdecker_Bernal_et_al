#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of functions for data preprocessing 

Created on Fri Dec 11 10:30:11 2020

@author: Dr. Freddy Bernal
"""


# Load dependencies
import numpy as np
import pandas as pd

from rdkit.Chem import RDKFingerprint
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import ConvertToNumpyArray

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from rdkit import Chem

#%% Function to generate fingerprints from rdkit


def calculate_fp(mol, method='maccs', n_bits=2048):
    """
    Generates molecular fingerprints for an specified molecule
    from the corresponding rdkit mol object.

    Parameters
    ----------
    mol : rdkit mol
        Molecule's structure.
    method : str
        Fingerprint type. Options: 'maccs', 'ecfp4', 'ecfp6', and 'rdk5'. 
        The default is 'maccs'.
    n_bits : int
        Number of fingerprint bits used for calculation. 
        The default is 2048.

    Returns
    -------
    rdkit obj
        Fingerprint bits for the molecule by chosen method.
    """
    
    # Use RDKit to generate fp according to selected method
    if method == 'maccs':
        return MACCSkeys.GenMACCSKeys(mol)
    if method == 'ecfp4':
        return GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits, useFeatures=False)
    if method == 'ecfp6':
        return GetMorganFingerprintAsBitVect(mol, 3, nBits=n_bits, useFeatures=False)
    if method == 'rdk5':
        return RDKFingerprint(mol, maxPath=5, fpSize=n_bits, nBitsPerHash=2)
    
    
    
def create_mol(df_l, method, n_bits):
    """
    Generates mol object and fingerprint of a molecule from SMILES strings.
    Uses calculate_fp.

    Parameters
    ----------
    df_l : Pandas dataframe
        Dataframe containing a column for SMILES strings called 'Smiles'.
    method : str
        Fingerprint type. Options: 'maccs', 'ecfp4', 'ecfp6', and 'rdk5'.
    n_bits : int
        Number of fingerprint bits used for calculation. 

    Returns
    -------
    None. Add new columns to the dataframe for 'mol', fingerprint object ('bv'), 
    and fingerprint bits as numpy array ('np_bv')
    """
    
    # Generate mol column and add Mol object.
    df_l['mol'] = df_l.Smiles.apply(Chem.MolFromSmiles)
    # Create a column for storing the molecular fingerprint as fingerprint object
    df_l['bv'] = df_l.mol.apply(
        # Apply the lambda function "calculate_fp" for each molecule
        lambda x: calculate_fp(x, method, n_bits)
    )
    # Allocate np.array to contain fp as bit-vector 
    df_l['np_bv'] = np.zeros((len(df_l), df_l['bv'][1].GetNumBits())).tolist()
    df_l.np_bv = df_l.np_bv.apply(np.array)
    # Convert the object fingerprint to NumpyArray and store in np_bv
    df_l.apply(lambda x: ConvertToNumpyArray(x.bv, x.np_bv), axis=1)
    

#%% Functions for feature filtering 

def feature_filt(X, y, test_size, rnd_st):
    """
    Performs data preprocessing: 
        1) splitting data into training and test set (sklearn)
        1) dropping out variables with std = 0 within training set
        2) removal of features with correlation > 0.8 within training set 
        3) standard scaling of training set
        4) transforming test set using same parameters as for training.

    Parameters
    ----------
    X : Pandas DataFrame
        Data matrix of compounds x features.
    y : Pandas DataFrame or Series
        Activity values.
    test_size : float
        Test set ratio.
    rnd_st : int
        random seed used for training/test set splitting in sklearn.

    Returns
    -------
    X_train_sc : Pandas DataFrame
        Filtered and preprocessed descriptor data for training set.
    X_test_sc : Pandas DataFrame
        Filtered and preprocessed descriptor data for test set. 
    y_train : Pandas Series
        Activity data for training set.
    y_test : Pandas Series
        Activity data for test set.
    """
    
    # Training/test set split (sklearn)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=rnd_st)

    # Removing features with std = 0
    Xed = X_train.copy()
    Xed.drop(Xed.std()[Xed.std() == 0].index.values, axis=1, inplace=True)
    
    # Create correlation matrix
    corr = Xed.corr()
    # Check for correlations >= 0.8
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if abs(corr.iloc[i,j]) >= 0.8:
                if columns[j]:
                    columns[j] = False
    # Create a mask                
    selected_columns = Xed.columns[columns]
    # Create the reduced data frame    
    X_filt = Xed[selected_columns]
    
    # Scale variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filt)
    X_train_sc = pd.DataFrame(X_scaled, columns=selected_columns)
    
    # Finally, apply to test set data
    Xt = X_test.copy()
    Xt_filt = Xt[selected_columns]
    Xt_scaled = scaler.transform(Xt_filt)
    X_test_sc = pd.DataFrame(Xt_scaled, columns=selected_columns)
    
    return X_train_sc, X_test_sc, y_train, y_test
    


def feature_filt_adv(X, y, test_size, rnd_st):
    """
    Performs data preprocessing: 
        1) splitting data into training and test set (sklearn)
        1) dropping out variables with std = 0 within training set
        2) removal of features with correlation > 0.8 within training set 
        3) standard scaling of training set
        4) transforming test set using same parameters as for training.

    Parameters
    ----------
    X : Pandas DataFrame
        Data matrix of compounds x features.
    y : Pandas DataFrame or Series
        Activity values.
    test_size : float
        Test set ratio.
    rnd_st : int
        random seed used for training/test set splitting in sklearn.

    Returns
    -------
    X_train_sc : Pandas DataFrame
        Filtered and preprocessed descriptor data for training set.
    X_test_sc : Pandas DataFrame
        Filtered and preprocessed descriptor data for test set. 
    y_train : Pandas Series
        Activity data for training set.
    y_test : Pandas Series
        Activity data for test set.
    scaler : sklearn object
        instantiated scaler fitted on training set.
    """
        
    # Training/test set split (sklearn)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=rnd_st)

    # Removing features with std = 0
    Xed = X_train.copy()
    Xed.drop(Xed.std()[Xed.std() == 0].index.values, axis=1, inplace=True)
    
    # Create correlation matrix
    corr = Xed.corr()
    # Check for correlations >= 0.8
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if abs(corr.iloc[i,j]) >= 0.8:
                if columns[j]:
                    columns[j] = False
    # Create a mask                
    selected_columns = Xed.columns[columns]
    # Create the reduced data frame    
    X_filt = Xed[selected_columns]
    
    # Scale variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filt)
    X_train_sc = pd.DataFrame(X_scaled, columns=selected_columns)
    
    # Finally, apply to test set data
    Xt = X_test.copy()
    Xt_filt = Xt[selected_columns]
    Xt_scaled = scaler.transform(Xt_filt)
    X_test_sc = pd.DataFrame(Xt_scaled, columns=selected_columns)
    
    return X_train_sc, X_test_sc, y_train, y_test, scaler  

    
#%% Function to preprocess input data 

def preprocessing(X, y, preproc, rnd_st, test_size=0.25):
    """
    Generates training and test sets. If X is descriptor-based, splitting 
    with sklearn function, filtering (elimination of low variance and highly 
    correlating features) and scaling are performed using feature_filt. 
    If X is fingerprint-based, only splitting is carried out.

    Parameters
    ----------
    X : Pandas DataFrame
        Data matrix of compounds x features.
    y : Pandas DataFrame or Series
        Activity values.
    preproc : str
        Preprocessing method. For fingerprints set select 'split'. 
        For descriptors set select 'filter'. In this case, the function
        feature_filt is used.        
    rnd_st : int
        random seed used for training/test set splitting in sklearn.
    test_size : float
        Test set ratio. The default is 0.25.

    Returns
    -------
    X_train : Pandas DataFrame
        Filtered and preprocessed descriptor data for training set.
    X_test : Pandas DataFrame
        Filtered and preprocessed descriptor data for test set. 
    y_train : Pandas Series
        Activity data for training set.
    y_test : Pandas Series
        Activity data for test set.
    """
    
    # Create training and test sets according to the selected preprocessing method
    if preproc == 'split':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rnd_st)
    
    elif preproc == 'filter':
        X_train, X_test, y_train, y_test = feature_filt(X, y, test_size, rnd_st)
        
    else:
        print('Preprocessing mode not recognized: please indicate "split" for training/test set splitting')
        print('or "filter" for feature filtering')
    
    return X_train, X_test, y_train, y_test
        



