# -*- coding: utf-8 -*-

from .base_wrapper import _GBMWrapper
from ..trees_extraction import lightgbm_trees_to_dataframe


class _LightgbmWrapper(_GBMWrapper):
    
    def apply(self, X):
        return self.estimator.predict(X, pred_leaf=True)
    
    def trees_to_dataframe(self):
        return lightgbm_trees_to_dataframe(self.estimator)
    
    def learning_rate(self):
        return self.estimator.learning_rate
    
    def n_estimators(self):
        return self.estimator.n_estimators
    
    def reg_lambda(self):
        return self.estimator.reg_lambda
    
    def subsample(self):
        return self.estimator.subsample
    
    def _on_fit(self):
        pass
    
    