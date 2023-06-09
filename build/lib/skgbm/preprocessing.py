#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 17:46:28 2023

@author: krzysztof
"""

import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from .base import GBMWrapper

import pdb

class GBMFeaturizer(BaseEstimator, TransformerMixin, GBMWrapper):
    """
    Feature generator for any GBM models
    
    Parameters
    ----------
    estimator: object
        A gradient boosting model from scikit-learn, XGBoost, LightGBM or CatBoost library    
    one_hot: bool
        Transform the ouput categorical features using one-hot encoding
    append: bool 
        Append the newly created features to the original ones
    
    Examples
    --------
    >>> from skgbm.preprocessing import GBMFeaturizer
    >>> 
    
    
    """
    
    def __init__(self, estimator, one_hot: bool = True, append: bool = True):
        self.one_hot = one_hot
        if one_hot:
            self.ohe = OneHotEncoder()
        self.append = append
        super().__init__(estimator)
    
    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        if hasattr(self, 'ohe'):
            X_ = self.apply(X)
            self.ohe.fit(X_)
        return self
    
    def transform(self, X, y=None, **kwargs):
        output = self.apply(X)
        if hasattr(self, 'ohe'):
            output = self.ohe.transform(output)
        if self.append:
            # pdb.set_trace()
            output = scipy.sparse.hstack([X, output])
        return output
    


