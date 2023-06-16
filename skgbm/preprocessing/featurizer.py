#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from ..base import GBMWrapper


class GBMFeaturizer(BaseEstimator, TransformerMixin, GBMWrapper):
    """
    Feature generator for any GBDT model.
    
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
    >>> from sklearn.datasets import load_diabetes
    >>> from skgbm.preprocessing import GBMFeaturizer
    >>> from lightgbm import LGBMRegressor
    >>>
    >>> X, y = load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> gbm_featurizer = GBMFeaturizer(LGBMRegressor())
    >>> gbm_featurizer.fit(X_train, y_train)
    >>> gbm_featurizer.transform(X_test)
    
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
            output = scipy.sparse.hstack([X, output])
        return output
    


