# -*- coding: utf-8 -*-

import json

from .base_wrapper import _GBMWrapper
from ..trees_extraction import xgboost_trees_to_dataframe
from ..utils import is_fitted

try:
    import xgboost as xgb
except:
    pass


class _XgboostWrapper(_GBMWrapper):
    
    def __init__(self, estimator):
        super().__init__(estimator)
        if is_fitted(estimator):
            self.config = json.loads(self.estimator.get_booster().save_config())
    
    def apply(self, X):
        return self.estimator._Booster.predict(xgb.DMatrix(X), pred_leaf=True)
    
    def trees_to_dataframe(self):
        return xgboost_trees_to_dataframe(self)
    
    def learning_rate(self):
        eta = self.config['learner']['gradient_booster']['updater'] \
            ['grow_colmaker']['train_param']['eta']
        return float(eta)
    
    def n_estimators(self):
        return self.estimator.n_estimators
    
    def reg_lambda(self):
        reg_lambda = self.config['learner']['gradient_booster']['updater'] \
            ['grow_colmaker']['train_param']['reg_lambda']
        return float(reg_lambda)
    
    def subsample(self):
        subsample = self.config['learner']['gradient_booster']['updater'] \
            ['grow_colmaker']['train_param']['subsample']
        return float(subsample)
    
    def _on_fit(self):
        self.config = json.loads(self.estimator.get_booster().save_config())
        
    
    