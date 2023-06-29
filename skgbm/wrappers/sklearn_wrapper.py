# -*- coding: utf-8 -*-

from .base_wrapper import _GBMWrapper
from ..trees_extraction import sklearn_trees_to_dataframe

import numpy as np


class _SklearnWrapper(_GBMWrapper):
    
    def apply(self, X):
        return np.squeeze(self.estimator.apply(X))
    
    def trees_to_dataframe(self):
        return sklearn_trees_to_dataframe(self)
    
    def learning_rate(self):
        return self.estimator.learning_rate
    
    def n_estimators(self):
        return self.estimator.n_estimators
    
    def reg_lambda(self):
        return 0
    
    def subsample(self):
        return self.estimator.subsample
    