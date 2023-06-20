# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..base import GBMWrapper
from ..utils import check_is_gbm_regressor

# TODO: can be written faster?
def LCM(vector1, vector2):
    '''
    utility function to create leaf coincidence matrix L from leaf membership vectors vector1 (train) and vector2 (test)
    element l_s,f in L takes value of 1 IFF observations v1 and v2 are allocated to the same leaf (for v1 in vector1 and v2 in vector2)
    that is, vector1[v1] == vector2[v2], otherwise 0

    Input arguments vector1 and vector2 are list-like, that is, support indexing and len()
    Output L is a python matrix of dimension (len(vector1), len(vector2))
    '''
    # tree contains a list of predicted values
    N1 = len(vector1)
    N2 = len(vector2)

    L = np.full((N1, N2), np.nan)
    for v1 in range(0, N1):
        for v2 in range(0, N2):
            same = (vector1[v1] == vector2[v2])
            L[v1][v2] = same

    return L

class AXIL(BaseEstimator, TransformerMixin):
    """
    Instance importance for regression.
    
    
    Parameters
    ----------
    estimator: object
        A gradient boosting model from scikit-learn, XGBoost, LightGBM or CatBoost library   
    
    References
    ----------
    .. [1] P. Geertsema and  H. Lu, `"Instance-based Explanations for Gradient Boosting Machine Predictions with AXIL Weights"
           <https://arxiv.org/abs/2301.01864>`_, 
           2009.
    .. [2] `AXIL_paper GitHub repository <https://github.com/pgeertsema/AXIL_paper>`_
    
    Examples
    --------
    >>> from skgbm.xai import AXIL
    
    
    """    
    
    def __init__(self, estimator: object):
        # TODO: check is_fitted
        # TODO: we don't have to store the data?
        check_is_gbm_regressor(estimator)
        self.wrapped_ = GBMWrapper(estimator)
        super().__init__()
    
    @property
    def estimator(self):
        return self.wrapped_.estimator
    
    def fit(self, X, y, **kwargs):
        # TODO: is the first estimator always mean? (for any loss function?)
        # check with e.g. quantile loss
        # https://github.com/pgeertsema/AXIL_paper/blob/main/axil.py
        
        # number of observations
        N = len(X)
        num_trees = self.model.num_trees() # n_estimators?
        learning_rate = self.wrapped_.learning_rate
        
        # useful matrices
        ones = np.ones((N,N))
        I = np.identity(N)
        
        # Creating matrix of target-leaf membership
        instance_leaf_membership = self.wrapped_.apply(X)
        self.lm = np.concatenate((
            np.ones((1, N)), # First prediction: "one leaf" 
            instance_leaf_membership.T
        ), axis = 0) + 1
        
        # Clear list of P matrices (to be used for calculating AXIL weights)
        self.P_list = []
        
        # iterations 0 model predictions (simply average of training data)
        # corresponds to "tree 0"
        # Contribution to mean
        P_0 = (1/N) * ones
        self.P_list.append(P_0)
        G_prev = P_0
        
        # do iterations for trees 1 to num_trees (inclusive)
        # note, LGB trees ingnores the first (training data mean) predictor, so offset by 1
        for i in range(1, num_trees+1):
            D = LCM(self.lm[i], self.lm[i])
            P = self.learning_rate * ( (D / (ones @ D)) @ (I-G_prev) )
            self. P_list.append(P)
            G_prev = G_prev + P
        
        self.trained = True
        return self
    
    def transform(X, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
        return X


if __name__ == '__main__':
    pass

    