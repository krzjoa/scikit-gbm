#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 16:40:06 2023

@author: krzysztof
"""

import numpy as np
try:
    import xgboost as xgb
except:
    pass

# from sklearn.ensemble import GradientBoostingRegressor
# from xgboost import XGBRanker
# from lightgbm import LGBMRanker
# from catboost import CatBoostRanker

# =============================================================================
#                                   APPLY
# =============================================================================


def sklearn_apply(obj, X):
    return np.squeeze(obj.apply(X))

def xgboost_apply(obj, X):
    return obj._Booster.predict(xgb.DMatrix(X), pred_leaf=True)
    
def lightgbm_apply(obj, X):
    return obj.predict(X, pred_leaf=True)

def catboost_apply(obj, X):
    return obj.calc_leaf_indexes(X)


apply_fun = {
    
    # scikit-learn
    'GradientBoostingClassifier' : sklearn_apply,
    'GradientBoostingRegressor'  : sklearn_apply,
    
    # XGBoost
    'XGBClassifier' : xgboost_apply,
    'XGBRegressor'  : xgboost_apply,
    'XGBRanker'     : xgboost_apply, # ?
    
    # LightGBM
    'LGBMClassifier' : lightgbm_apply,
    'LGBMRegressor'  : lightgbm_apply,
    'LGBMRanker'     : lightgbm_apply,
    
    # CatBoost
    'CatBoostClassifier' : catboost_apply,
    'CatBoostRegressor'  : catboost_apply,
    'CatBoostRanker'     : catboost_apply
    
}



class GBMWrapper:
    
    def __init__(self, estimator):
        self.estimator = estimator
        self._apply = apply_fun[type(self.estimator).__name__]
        
    def fit(self, X, y, **kwargs):
        self.estimator.fit(X, y, **kwargs)
        return self
    
    def apply(self, X):
        return self._apply(self.estimator, X)


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    
    from sklearn.ensemble import GradientBoostingClassifier
    from xgboost import XGBClassifier
    
    X, y = make_classification()
    
    gbm_wrapped = GBMWrapper(GradientBoostingClassifier())
    gbm_wrapped.fit(X, y)    
    gbm_wrapped.apply(X)
    
    
    