# -*- coding: utf-8 -*-

from .base_wrapper import _GBMWrapper
from ..trees_extraction import catboost_trees_to_dataframe


class _CatboostWrapper(_GBMWrapper):
    
    def __init__(self, estimator):
        super().__init__(estimator)
        self.params = self.estimator.get_all_params()
    
    def apply(self, X):
        return self.estimator.calc_leaf_indexes(X)
    
    def trees_to_dataframe(self):
        return catboost_trees_to_dataframe(self.estimator)
    
    def learning_rate(self):
        return self.estimator.learning_rate
    
    def n_estimators(self):
        return self.params['iterations']
    
    def reg_lambda(self):
        return self.params['l2_leaf_reg']
    
    def sumbsample(self):
        return self.params['subsample']
    
    
    