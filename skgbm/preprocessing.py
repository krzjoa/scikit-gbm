#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 17:46:28 2023

@author: krzysztof
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
# rename as preprocessing.py

from .base import GBMWrapper


class GBMFeaturizer(BaseEstimator, TransformerMixin):
    """
    Feature generator for any GBM models
    
    Parameters
    ----------
    estimator: object
        A gradient boosting model from scikit-learn, XGBoost, LightGBM or CatBoost library    
    
    
    Examples
    --------
    >>>
    >>>
    
    
    """
    
    def __init__(self, estimator):
        #self.estimator = estimator
        self.wrapped_gbm = GBMWrapper(estimator)
        self.ohe = OneHotEncoder()
    
    @property
    def estimator(self):
        return self.wrapped_gbm.estimator
    
    def fit(self, X, y, **kwargs):
        self.wrapped_gbm.fit(X, y, **kwargs)
        X_ = self.wrapped_gbm.apply(X)
        self.ohe.fit(X_)
        return self
    
    def transform(self, X, y=None, **kwargs):
        leaves = self.wrapped_gbm.apply(X)
        return self.ohe.transform(leaves)
    


