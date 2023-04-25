# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 08:24:57 2023

@author: Krzysztof.Joachimiak
"""


import re


class Tree:
    """A base class to store the information about a tree structure"""
    
    # TODO: consider moving it to Cython
    # TODO: add index
    # TODO: add safe_int or maybe_int
    # TODO: to_df(), to_dict()
    # TODO: to_onnx() ? Compare:
    # http://www.xavierdupre.fr/app/sklearn-onnx/helpsphinx/auto_examples/plot_pipeline_xgboost.html
    # https://catboost.ai/en/docs/concepts/apply-onnx-ml
    # TODO: TreeEnsemble => check_is_tree_based
    
    def __init__(self, children_right, children_left, feature, threshold, value, missing=None):
        self.children_right = children_right
        self.children_left = children_left
        self.missing = missing
        self.feature = feature
        self.threshold = threshold
        self.value = value
    
    @classmethod
    def from_sklearn(cls, tree):
        tree_ = tree.tree_
        return cls(
              children_right = tree_.children_right,
              children_left  = tree_.children_left,
              feature        = tree_.feature,
              threshold      = tree_.threshold,
              value          = tree_.value
        )
    
    @classmethod
    def from_xgboost(cls, tree):
        tree = tree.split('\t')
        tree = [node.strip() for node in tree if node != '']
        
        idx = [int(node.split(':')[0]) for node in tree]
        
        prog_feat = re.compile('\:f(\d+).*')
        feature = [prog_feat.findall(node) for node in tree]
        
        prog_thr = re.compile('\:\[f\d+<(.*)?].*')
        threshold = [prog_thr.findall(node) for node in tree]
        
        prog_left = re.compile('yes=(\d+)')
        children_left = [prog_left.findall(node) for node in tree]
        
        prog_right = re.compile('yes=(\d+)')
        children_right = [prog_right.findall(node) for node in tree]
        
        prog_miss = re.compile('missing=(\d+)')
        missing = [prog_miss.findall(node) for node in tree]
        
        prog_val = re.compile('leaf=(.*)')
        value = [prog_val.findall(node) for node in tree]
        
        return cls(
              children_right = children_right,
              children_left  = children_left,
              feature        = feature,
              threshold      = threshold,
              value          = value, 
              missing        = missing
        )
        



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
    lgb = LGBMClassifier().fit(X, y)
    cab = CatBoostClassifier().fit(X, y)
    
    # sklearn
    dt = sklearn_gbm.estimators_[0][0]
    sk_tree = Tree.from_sklearn(dt)
    
    # XGBoost
    xgb_trees = xgb.get_booster().get_dump()
    xgb_tree = xgb_trees[0]
    xgb_tree_ = Tree.from_xgboost(xgb_tree)
    
    # LightGBM
    lgb_tree = lgb.booster_.dump_model()
    
    
    
    st = '0:[f17<0.216996461] yes=1,no=2,missing=1'
    prog = re.compile('.*\:\[f(\d+).*')
    prog.match(st)[0]
