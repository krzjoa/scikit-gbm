# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod


class _GBMWrapper(ABC):
    # TODO: static methods?
    
    def __init__(self, estimator):
        self.estimator = estimator
    
    @abstractmethod
    def apply(self, X):
        pass
    
    @abstractmethod
    def trees_to_dataframe(self):
        pass
    
    @abstractmethod
    def learning_rate(self):
        pass
    
    @abstractmethod
    def n_estimators(self):
        pass
    
    @abstractmethod
    def subsample(self):
        pass
    
    @abstractmethod
    def _on_fit(self):
        pass