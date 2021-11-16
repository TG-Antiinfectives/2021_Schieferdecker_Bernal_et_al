#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of miscellaneous tools  

Created on Fri Dec 11 10:30:11 2020

@author: Dr. Freddy Bernal
"""


# Loading dependencies
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from preprocessing import feature_filt_adv
from applicability_domain import full_ad_determination


#%% FUNCTIONS


def prepare_new_data(Xnew, X, y, testsize, rnd_st):
    """
    Prepares data of new compounds for compliance with training data.
    It uses feature_filt_adv.

    Parameters
    ----------
    Xnew : Pandas DataFrame
        X data for a new set of compounds.
    X : Pandas DataFrame
        Data matrix of compounds x features.
    y : Pandas Series
        Activity data.
    test_size : float
        Test set ratio.
    rnd_st : int
        random seed used for training/test set splitting in sklearn.

    Returns
    -------
    X_train : Pandas DataFrame
        Filtered and preprocessed descriptor data for training set.
    y_train : Pandas Series
        Activity data for training.
    Xnew_sc : Pandas DataFrame
        Filtered and preprocessed descriptor data for new set of compounds.
    """
    
    # Apply transformation retrieving the scaler used for training samples
    X_train , _ , y_train , _ , scaler = feature_filt_adv(X, y, testsize, rnd_st)
    
    # Use the scaler for the pretreatment of the new data set
    Xnew_filt = Xnew[X_train.columns]
    Xnew_sc = scaler.transform(Xnew_filt)
    Xnew_sc = pd.DataFrame(Xnew_sc, columns=X_train.columns)
    
    return X_train, y_train, Xnew_sc




def act_conv_calib(base):
    """
    Creates a "calibration curve" for activity conversion from Mtb-H37rv to 
    activity against M. vaccae, based on linear regression.

    Parameters
    ----------
    base : Pandas DataFrame
        Activity data of compounds against both Mtb-H37rv and M. vaccae, both
        expressed as pMIC.

    Returns
    -------
    model : sklearn obj
        Fitted ordinary least squares regression model.
    calib_corr : Numpy Array
        Predicted actvity against M. vaccae.
    stats : Pandas DataFrame
        Statistical parameters for Linear regression model, including R2,
        RMSE, and MAE.
    list
        Coefficient and intercept from calibration.
    """
   
    # Create linear regression model (OLS) and calculate slope and intercept
    model = LinearRegression()
    model.fit(base.iloc[:, 1].to_numpy().reshape(-1, 1), base.iloc[:, 0].to_numpy())
    a = model.coef_
    b = model.intercept_
    
    # Calculate activity for calibration set 
    calib_corr = model.predict(base.iloc[:, 1].to_numpy().reshape(-1, 1))
    
    # Evaluate the performance of the linear model
    r2 = r2_score(base.iloc[:, 0].to_numpy(), calib_corr)
    mse = mean_squared_error(base.iloc[:, 0].to_numpy(), calib_corr)
    mae = mean_absolute_error(base.iloc[:, 0].to_numpy(), calib_corr)
    
    # Report statistics in a DataFrame
    stats = pd.DataFrame([r2, np.sqrt(mse), mae]).T
    stats.columns = ['R2', 'RMSE', 'MAE']

    return model, calib_corr, stats, [a[0], b]



def activity_conversion(base, new_data): 
    """
    Converts activity against Mtb-H37rv to activity against M. vaccae.

    Parameters
    ----------
    base : Pandas DataFrame
        Activity data of compounds against both Mtb-H37rv and M. vaccae, both
        expressed as pMIC.
    new_data : Pandas DataFrame
        Data set containing compounds information and activity against Mtb-H37rv
        (MIC in Molarity).

    Returns
    -------
    activity_corr : Numpy Array
        Actvity against M. vaccae (from prediction using an OLS model)
    newClass : Numpy Array
        Activity class for the set of compounds using 0.1 M as threshold
        (< 0.1 is active, 1).
    """
    
    # Convert activity to pMIC
    pMIC = -np.log10(new_data.MIC_M)
    
    # Calibrate a linear model
    model, _ , _ , _ = act_conv_calib(base) 
    # Calculate pMIC against to M. vaccae using the model
    conv_pMIC = model.predict(pMIC.to_numpy().reshape(-1, 1))
    # Transform pMIC to MIC in uM
    # Consider some compounds were originally tested against M. vaccae
    # (therefore no conversion was actually required)
    new_MIC_uM = []
    for i in range(len(conv_pMIC)):
        if 'vaccae' in new_data.M_species.values[i]:
            d = 1e6 * new_data.MIC_M.values[i] 
            new_MIC_uM.append(d)
        else:
            d = 1e6 * 10 ** (-conv_pMIC[i])
            new_MIC_uM.append(d)

    # Define activity class after conversion
    newClass = []
    for i in new_MIC_uM:
        if i < 0.1:
            newClass.append(1)
        else:
            newClass.append(0)
    
    # Compile results in array
    activity_corr = np.array(new_MIC_uM)
    newClass = np.array(newClass)
    
    return activity_corr, newClass



def rep_pred(model, X_train, y_train, Xnew, rep=10):
    """
    Performs repeated activity class predictions using an specified model.

    Parameters
    ----------
    model : sklearn obj
        Instantiated model.
    X_train : Pandas DataFrame
        Training set X data.
    y_train : Pandas Series
        Training set activity data.
    Xnew : Pandas DataFrame
        X data for the set of new compounds.
    rep : int
        Number of repetitions. The default is 10.

    Returns
    -------
    ypred_f : Numpy Array
        Average predicted activity class.
    pp.mean : Numpy Array
        Average predicted probability.
    """
    
    # Loop over the number of repetitions
    for i in range(rep):
        # Get activity and probability 
        x1, x2 = predict(model, X_train, y_train, Xnew)
        if i == 0:
            ypred = x1
            pp = x2
        else:
            ypred = np.vstack((ypred, x1))
            pp = np.vstack((pp, x2))
            
    # Calculate activity class considering the mean predicted probability 
    # from the specified number of repetitions
    ypred_f = []
    for i in pp.T.mean(axis=1):
        if i > 0.5:
            ypred_f.append(1)
        else:
            ypred_f.append(0)
            
    # Compile results in array
    ypred_f = np.array(ypred_f)      
    pp_f = pp.T.mean(axis=1)
    
    return ypred_f, pp_f




def predict(model, X_train, y_train, Xnew):
    """
    Obtains predicted class and probability for a set of compounds using 
    certain model.

    Parameters
    ----------
    model : sklearn obj
        Instantiated model.
    X_train : Pandas DataFrame
        Training set X data.
    y_train : Pandas Series
        Training set activity data.
    Xnew : Pandas DataFrame
        X data for the set of new compounds.

    Returns
    -------
    y_pred : Numpy Array
        Predicted activity class.
    pp : Numpy Array
        Predicted probability.
    """
    
    # Fit the model
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(Xnew)

    # Calculate probabilities or decisions
    try:
        pp = model.predict_proba(Xnew)[:,1]
    except AttributeError:
        pp = model.decision_function(Xnew)
        
    return y_pred, pp




def report(inad_voting, ypred, pp, names):
    """
    Creates a DataFrame reporting Name, AD, predicted Class and predicted
    Probability calculated by the model.

    Parameters
    ----------
    inad_voting : Numpy Array
        AD determination by different methods.
    ypred : Numpy Array
        Predicted activity class.
    pp : Numpy Array
        Predicted probability.
    names : list or Numpy Array
        Names or IDs for samples.

    Returns
    -------
    report : Pandas DataFrame
        Compilation report with ID, AD, predicted activity class, and predicted 
        probability expressed as %.
    """
        
    # Define empty lists for each element to report and a counter
    ad = []
    pred = []
    prob = []
    c = 0
    # Loop over the AD results and assign each sample to a category (in or out)
    # with the respective predicted activity class and probability
    for i, vote in enumerate(inad_voting):
        if vote == 0:
            ad.append(vote)
            pred.append(np.nan)
            prob.append(np.nan)
            c += 1
        else:
            ad.append(vote)
            pred.append(ypred[i - c])
            if ypred[i - c] == 1:
                p = np.round(pp[i - c] * 100, 2)
            else:
                p = np.round((1 - pp[i - c]) * 100, 2)
            prob.append(p)
    
    # Compile results in DataFrame
    report = np.vstack((names, ad, pred, prob)).T
    report = pd.DataFrame(report)
    report.columns = ['ID', 'AD', 'Activity Class', 'Probability']
    
    return report



def compile_summary(model, X_train, y_train, Xnew_sc, ad_params, newClass, 
                    MIC_cor, names=None, gen_rep=False):
    """
    Creates a summary report to be used for further analysis of predictions
    and sharing.

    Parameters
    ----------
    model : sklearn obj
        Instantiated ML model.
    X_train : Pandas DataFrame
        Training set X data.
    y_train : Pandas Series
        Training set activity data.
    Xnew_sc : Pandas DataFrame
        X data for the set of new compounds.
    ad_params : dict
        Specified number of neighbors and percentile to be used for threshold 
        calculation.
    newClass : Numpy Array
        Activity class for the set of compounds using 0.1 M as threshold
        (< 0.1 is active, 1).
    MIC_cor : Numpy Array
        Activity against M. vaccae (from prediction using an OLS model)
    names : Numpy Array or list
        Names or IDs for samples.
    gen_rep : bool
        Whether generates an additional report (dataframe) indicating predicted
        class, probability (expressed as %), and compliance with AD. Useful for
        new unseen compounds in further lead optimization efforts.
        The default is False.

    Returns
    -------
    summary: Numpy Array
        Actual activity class, predicted activity and probability. Proximity to 
        activity class cutoff (0 indicates proximity and probable problems with
        predictions) and accepted probability range (0 indicates the sample is 
        too close to the decision boundary of the model)
    """
    
    # Evaluate compliance of each sample with AD of the model and remove
    # samples outside AD
    Xnew_red, ynew_red, inad_voting, _ , mask = full_ad_determination(X_train, 
                                                                      Xnew_sc, 
                                                                      ad_params, 
                                                                      ynew=newClass)
    
    # Calculate activity class repeated times
    ypred, pp = rep_pred(model, X_train, y_train, Xnew_red, rep=10)
    # Reduce activity according to compounds within AD
    act = MIC_cor[mask]

    # Select samples close to the boundary for class definition
    close_limit = []
    for i, j in enumerate(act):
        if (j > 0.01) & (j < 0.2):
            close_limit.append(i)        
    
    # Create a list of indices for compounds within an acceptable probability
    within_prob = [i for i, val in enumerate(pp) if val > 0.55 or val < 0.45]
    
    # Create a summary array including information about boundary
    # and predicted class and probability
    a = np.ones(len(ypred))

    summary = np.vstack((ynew_red, ypred, pp, a, a)).T

    for i, j in enumerate(summary):
        if i in close_limit:
            j[3] = 0
        if i not in within_prob:
            j[4] = 0

        
    if gen_rep:
        # Generate report using the custom function for it
        report_df = report(inad_voting, ypred, pp, names)
        
        return summary, report_df
    
    else:
        
        return summary

