# -*- coding: utf-8 -*-

from .base_wrapper import _GBMWrapper


class _XgboostWrapper(_GBMWrapper):
    
    def apply(self, X):
        return self.estimator.apply(X)
    
    def trees_to_dataframe(self):
        pass
    
    def learning_rate(self):
        return self.estimator.learning_rate
    
    def n_estimators(self):
        pass
    
    