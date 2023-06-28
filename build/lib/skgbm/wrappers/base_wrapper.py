# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod


class _GBMWrapper(ABC):
    
    # TODO: decide
    
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
    
    