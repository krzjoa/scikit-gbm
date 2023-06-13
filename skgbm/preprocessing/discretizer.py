#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import clone

from .base import GBMWrapper
from ..tools import trees_to_dataframe
from ..trees_extraction import _catboost_raw_trees, _catbost_get_splits

try:
    import catboost
    CATBOOST_CLASSES = [
        catboost.CatBoostRegressor,
        catboost.CatBoostClassifier,
        catboost.CatBoostRanker
    ]
except:
    # If there is no CatBoost, any CatBoost model can't be passed anyway
    CATBOOST_CLASSES = []




class GBMDiscretizer(BaseEstimator, TransformerMixin, GBMWrapper):
    """
    Continuous feature discretizer based on gradinet boosted decision tree ensembles
    
    Parameters
    ----------
    estimator: object
        A gradient boosting model from scikit-learn, XGBoost, LightGBM or CatBoost library    
    one_hot: bool
        Transform the ouput categorical features using one-hot encoding
    columns: list of str
        List of column names to be transformed
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
    
    def __init__(self, estimator, 
                 columns: list, 
                 one_hot: bool = True, 
                 append: bool = False):
        self.one_hot = one_hot
        if one_hot:
            self.ohe = OneHotEncoder()
        self.append = append
        self.columns = columns
        super().__init__(estimator)
    
    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        
        # Fitting estimators (one per tranformed column)
        self.estimators_ = {}
        
        for col in self.columns:
            self.estimators_[col] = est_ =  \
                clone(self.estimator).fit(X, y)
            
            # Getting the data frame for CatBoost is redundant
            if type(est_) not in CATBOOST_CLASSES:
                trees = trees_to_dataframe(est_)
                splits = trees.sort_values('threshold')['threshold'].drop_duplicates()
            else:
                trees = _catboost_raw_trees(est_)
        
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
    
    def fit_transform(self, X, y, **kwargs):
        return self.fit(X, y, **kwargs).transform(X, y, **kwargs)
    


