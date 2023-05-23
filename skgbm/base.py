#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 16:40:06 2023

@author: krzysztof
"""

from typing import Final

from sklearn.tree._tree import Tree

import numpy as np
try:
    import xgboost as xgb
except:
    pass

# from sklearn.ensemble import GradientBoostingRegressor
# from xgboost import XGBRanker
# from lightgbm import LGBMRanker
# from catboost import CatBoostRanker

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
#                             GET FEATURES
# =============================================================================

# gradinet boosting 
#     feature_names_in_ : ndarray of shape (`n_features_in_`,)
#        Names of features seen during :term:`fit`. Defined only when `X`
#        has feature names that are all strings.




# =============================================================================
#                              GET TREES
# =============================================================================

def sklearn_get_trees(obj):
    # TODO: use common representation for trees
    return obj.estimators_

def xgboost_get_trees(obj):
    return obj.get_booster().get_dump()

def lightgbm_get_trees(obj):
    return obj.booster_.dump_model()

def catboost_get_trees(obj):
    n_trees = cab._object._get_tree_count()
    tree_splits = [
        cab._object._get_tree_splits(i, None) for i in range(n_trees)
    ]
    leaf_values = [
        cab._get_tree_leaf_values(i) for i in range(n_trees)
    ]

    if not cab._object._is_oblivious():
        step_nodes = [
            cab._get_tree_step_nodes(i) for i in range(n_trees) 
        ]
        node_to_leaf = [
            cab._get_tree_node_to_leaf(i) for i in range(n_trees) 
        ] 
    return 

# trees_to_dataframe() LightGBM

get_trees_fun = {
    'sklearn': sklearn_get_trees,
    'xgboost' : xgboost_get_trees,
    'lightgbm' : lightgbm_get_trees,
    'catboost' : catboost_get_trees,
}

class GBMWrapper:
    """A general wrapper object for all the acceptable models"""
    
    def __init__(self, estimator):
        self.estimator = estimator
        self._apply = match_fun(self.estimator, apply_fun) 
        self._get_trees = match_fun(self.estimator, get_trees_fun) 
        
    def fit(self, X, y, **kwargs):
        self.estimator.fit(X, y, **kwargs)
        return self
    
    def apply(self, X):
        return self._apply(self.estimator, X)
    
    def get_trees(self):
        return self._get_trees()


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
    
    dt = sklearn_gbm.estimators_[0]
    
    
    gbm_wrapped = GBMWrapper(GradientBoostingClassifier())
    gbm_wrapped.fit(X, y)    
    gbm_wrapped.apply(X)
    
    
    