#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 16:40:06 2023

@author: krzysztof
"""


from .wrappers import wrap_estimator
from .utils import check_estimator

# read: https://arxiv.org/pdf/2101.07077.pdf


class GBM:
    """A general wrapper object for all the acceptable models"""
    
    def __init__(self, estimator):
        check_estimator(estimator)
        # TODO: try to simplify it
        self.estimator = estimator
        self.wrapped_estimator_ = wrap_estimator(self.estimator)
        
        # Maybe inject somehow inheriting class??
        # Idea:
        # Turn GBMWrapper into an interface, abstract class with only .from_estimator method
        # https://docs.python.org/3/library/abc.html
        
        # Internal wrapper and dynamic method assignment?
        # _GBMInternal -> _XGBoostWrapper
        
    def fit(self, X, y, **kwargs):
        self.wrapped_estimator_.fit(X, y, **kwargs)
        return self
    
    def apply(self, X):
        return self.wrapped_estimator_.apply( X)
    
    def trees_to_dataframe(self):
        return self.wrapped_estimator_.trees_to_dataframe()
    
    @property
    def learning_rate(self):
        return self.wrapped_estimator_.learning_rate()
    
    @property
    def n_estimators(self):
        return self.wrapped_estimator_.n_estimators()
        
    @property
    def reg_lambda(self):
        return self.wrapped_estimator_.reg_lambda()
    
    @property
    def subsample(self):
        return self.wrapped_estimator_.subsample()
            
            
        

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
    
    
    