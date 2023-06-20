# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..base import GBMWrapper

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
        # TOOD: ensure is regressor
        # TODO: we don't have to store the data?
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
        
        # Creating matrix of
        instance_leaf_membership = self.wrapped_.apply(X)
        lm = np.concatenate((
            np.ones((1, N)), 
            instance_leaf_membership.T
        ), axis = 0) + 1
    