#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 16:40:06 2023

@author: krzysztof
"""

from typing import Final

from sklearn.tree._tree import Tree
import json

import numpy as np
try:
    import xgboost as xgb
except:
    pass

from .trees_extraction import sklearn_trees_to_dataframe, \
    xgboost_trees_to_dataframe, lightgbm_trees_to_dataframe, \
    catboost_trees_to_dataframe
from .utils import check_estimator, is_catboost

# read: https://arxiv.org/pdf/2101.07077.pdf
name_map: Final = {
    
    # scikit-learn
    'GradientBoostingClassifier' : 'sklearn',
    'GradientBoostingRegressor'  : 'sklearn',
    
    # XGBoost
    'XGBClassifier' : 'xgboost',
    'XGBRegressor'  : 'xgboost',
    'XGBRanker'     : 'xgboost', # ?
    
    # LightGBM
    'LGBMClassifier' : 'lightgbm',
    'LGBMRegressor'  : 'lightgbm',
    'LGBMRanker'     : 'lightgbm',
    
    # CatBoost
    'CatBoostClassifier' : 'catboost',
    'CatBoostRegressor'  : 'catboost',
    'CatBoostRanker'     : 'catboost'
}

def match_fun(obj, fun_map: dict):
    return fun_map[name_map[type(obj).__name__]]

# =============================================================================
#                                   APPLY
# =============================================================================

# TODO: consider this https://stackoverflow.com/questions/12846054/calling-a-function-from-string-inside-the-same-module-in-python

def sklearn_apply(obj, X):
    return np.squeeze(obj.apply(X))

def xgboost_apply(obj, X):
    return obj._Booster.predict(xgb.DMatrix(X), pred_leaf=True)
    
def lightgbm_apply(obj, X):
    return obj.predict(X, pred_leaf=True)

def catboost_apply(obj, X):
    return obj.calc_leaf_indexes(X)


apply_fun = {
    'sklearn': sklearn_apply,
    'xgboost' : xgboost_apply,
    'lightgbm' : lightgbm_apply,
    'catboost' : catboost_apply,
}


# =============================================================================
#                               LEARNING RATE
# =============================================================================

def sklearn_learning_rate(obj):
    return obj.learning_rate

def xgboost_learning_rate(obj):
    if obj.learning_rate is not None:
        return obj.learning_rate is not None
    else:
        config = json.loads(obj.get_booster().save_config())
        # Duplicates: eta and learning_rate
        return config['train_param']['eta']

def lightgbm_learning_rate(obj):
    return lgb.learning_rate

def catboost_learning_rate(obj):
    return cab.learning_rate_

learning_rate_fun = {
    'sklearn': sklearn_learning_rate,
    'xgboost' : xgboost_learning_rate,
    'lightgbm' : lightgbm_learning_rate,
    'catboost' : catboost_learning_rate
}

# =============================================================================
#                               GET FEATURES
# =============================================================================

# gradinet boosting 
#     feature_names_in_ : ndarray of shape (`n_features_in_`,)
#        Names of features seen during :term:`fit`. Defined only when `X`
#        has feature names that are all strings.


# =============================================================================
#                             GET TREES
# =============================================================================

trees_to_dataframe_fun = {
    'sklearn': sklearn_trees_to_dataframe,
    'xgboost' : xgboost_trees_to_dataframe,
    'lightgbm' : lightgbm_trees_to_dataframe,
    'catboost' : catboost_trees_to_dataframe,
}


class GBMWrapper:
    """A general wrapper object for all the acceptable models"""
    
    def __init__(self, estimator):
        check_estimator(estimator)
        # TODO: try to simplify it
        self.estimator = estimator
        self._apply = match_fun(self.estimator, apply_fun) 
        self._trees_to_dataframe = match_fun(self.estimator, trees_to_dataframe_fun) 
        self._learning_rate = match_fun(self.estimator, learning_rate_fun)
        
    def fit(self, X, y, **kwargs):
        self.estimator.fit(X, y, **kwargs)
        return self
    
    def apply(self, X):
        return self._apply(self.estimator, X)
    
    def trees_to_dataframe(self):
        return self._trees_to_dataframe(self.estimator)
    
    @property
    def learning_rate(self):
        return self.learning_rate(self.estimator)
    
    @property
    def n_estimators(self):
        if not is_catboost(self.estimator):
            return self.estimator.n_estimators
        else:
            return cab.get_all_params()['iterations']
            
        

if __name__ == '__main__':
    from sklearn.datasets import make_classification
    
    from sklearn.ensemble import GradientBoostingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    
    from sklearn.tree import plot_tree  
    
    X, y = make_classification()
    
    
    sklearn_gbm = GradientBoostingClassifier().fit(X, y)
    xgb = XGBClassifier().fit(X, y)
    cab = CatBoostClassifier().fit(X, y)
    lgb = LGBMClassifier().fit(X, y)
    
    dt = sklearn_gbm.estimators_[0]
    
    
    gbm_wrapped = GBMWrapper(GradientBoostingClassifier())
    gbm_wrapped.fit(X, y)    
    gbm_wrapped.apply(X)
    
    
    