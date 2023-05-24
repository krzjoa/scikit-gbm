# -*- coding: utf-8 -*-
"""
Created on Wed May 24 09:58:47 2023

@author: Krzysztof.Joachimiak
"""

import pandas as pd
import numpy as np



# =============================================================================
#                               SCIKIT-LEARN
# =============================================================================

_TREE_LEAF = -1

 # See: https://github.com/scikit-learn/scikit-learn/blob/f034f57b1ad7bc5a7a5dd342543cea30c85e74ff/sklearn/tree/_tree.pyx#L1087
def sklearn_compute_node_depths(obj):
    """Compute the depth of each node in a tree.

    .. versionadded:: 1.3

    Returns
    -------
    depths : ndarray of shape (self.node_count,), dtype=np.int64
        The depth of each node in the tree.
    """

    depths = np.empty(obj.node_count, dtype=np.int64)
    children_left = obj.children_left
    children_right = obj.children_right
    node_count = obj.node_count

    depths[0] = 1  # init root node
    for node_id in range(node_count):
        if children_left[node_id] != _TREE_LEAF:
            depth = depths[node_id] + 1
            depths[children_left[node_id]] = depth
            depths[children_right[node_id]] = depth

    return depths


def sklearn_get_parent_idx(obj):
    return

def sklearn_get_trees(obj, as_dataframe=True):
    trees =  obj.estimators_
    
    if as_dataframe:
        nodes = []
        for tree_index, tree in enumerate(sk_trees):
            tree = tree[0].tree_
            decision_type = '<='
            has_children = (tree.children_left != -1) & (tree.children_right != -1)
            
           
            df = pd.DataFrame({
                'tree_index'        : [tree_index] * tree.node_count,
                'node_depth'        : sklearn_compute_node_depths(tree),
                'node_index'        : list(range(0, tree.node_count)),
                'left_child'        : tree.children_left,
                'right_child'       : tree.children_right,
                'parent_index'      : sklearn_get_parent_idx(tree),
                'split_feature'     : tree.feature,
                'split_gain'        : None,
                'threshold'         : tree.threshold,
                'decision_type'     : [decision_type if hc else None for hc in has_children ],
                'missing_direction' : None,
                'missing_type'      : None,
                'value'             : tree.value.ravel(),
                'weight'            : None,
                'count'             : tree.n_node_samples 
            })
            nodes.append(df)
        trees = pd.concat(nodes)
    else:
        # TODO: use common representation for trees
        pass

    return trees

# =============================================================================
#                                   XGBOOST
# =============================================================================

def xgboost_get_trees(obj):
    return obj.get_booster().get_dump()


# =============================================================================
#                                  LIGHTGBM
# =============================================================================

def lightgbm_get_trees(obj, as_dataframe=True):
    if as_dataframe:
        return obj.booster_.trees_to_dataframe()
    else:
        return obj.booster_.dump_model()
    
# =============================================================================
#                                  CATBOOST
# =============================================================================

def catboost_get_trees(obj):
    n_trees = obj._object._get_tree_count()
    tree_splits = [
        obj._object._get_tree_splits(i, None) for i in range(n_trees)
    ]
    leaf_values = [
        obj._get_tree_leaf_values(i) for i in range(n_trees)
    ]

    if not obj._object._is_oblivious():
        step_nodes = [
            obj._get_tree_step_nodes(i) for i in range(n_trees) 
        ]
        node_to_leaf = [
            obj._get_tree_node_to_leaf(i) for i in range(n_trees) 
        ] 
    return 







if __name__ == '__main__':
    from sklearn.datasets import make_classification
    
    from sklearn.ensemble import GradientBoostingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    import re
    
    from sklearn.tree import plot_tree  
    
    X, y = make_classification()
    
    
    sklearn_gbm = GradientBoostingClassifier().fit(X, y)
    xgb = XGBClassifier().fit(X, y)
    lgb = LGBMClassifier().fit(X, y)
    cab = CatBoostClassifier().fit(X, y)
    
    # sklearn
    dt = sklearn_gbm.estimators_[0][0]
    #sk_tree = Tree.from_sklearn(dt)
    
    # XGBoost
    xgb_trees = xgb.get_booster().get_dump()
    xgb_tree = xgb_trees[0]
    #xgb_tree_ = Tree.from_xgboost(xgb_tree)
    
    # LightGBM
    lgb_trees = lgb.booster_.dump_model()
    lgb_tree = lgb_trees['tree_info']
    
    st = '0:[f17<0.216996461] yes=1,no=2,missing=1'
    prog = re.compile('.*\:\[f(\d+).*')
    prog.match(st)[0]
    
    # CatBoost
    
    
    
    # Creating a data frame
    
    # ['tree_index',
    #  'node_depth',
    #  'node_index',
    #  'left_child',
    #  'right_child',
    #  'parent_index',
    #  'split_feature',
    #  'split_gain',
    #  'threshold',
    #  'decision_type',
    #  'missing_direction',
    #  'missing_type',
    #  'value',
    #  'weight',
    #  'count']
    
    import pandas as pd
    
    
    sk_trees = sklearn_gbm.estimators_
    
    nodes = []
    
    for tree_index, tree in enumerate(sk_trees):
        tree = tree[0].tree_
        
        print(tree_index)
        
        decision_type = '<='
        
        has_children = (tree.children_left != -1) & (tree.children_right != -1)
        
        # See: https://github.com/scikit-learn/scikit-learn/blob/f034f57b1ad7bc5a7a5dd342543cea30c85e74ff/sklearn/tree/_tree.pyx#L1087
        df = pd.DataFrame({
            'tree_index'        : [tree_index] * tree.node_count,
            'node_depth'        : None,
            'node_index'        : list(range(0, tree.node_count)),
            'left_child'        : tree.children_left,
            'right_child'       : tree.children_right,
            'parent_index'      : None,
            'split_feature'     : tree.feature,
            'split_gain'        : None,
            'threshold'         : tree.threshold,
            'decision_type'     : [decision_type if hc else None for hc in has_children ],
            'missing_direction' : None,
            'missing_type'      : None,
            'value'             : tree.value.ravel(),
            'weight'            : None,
            'count'             : tree.n_node_samples 
        })
