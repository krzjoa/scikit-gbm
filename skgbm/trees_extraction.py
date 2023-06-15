# -*- coding: utf-8 -*-
"""
Created on Wed May 24 09:58:47 2023

@author: Krzysztof.Joachimiak
"""

import pandas as pd
import numpy as np
import json
import os


COLUMN_ORDER = [
    'tree_index', 
    'node_depth', 
    'node_index', 
    'left_child', 
    'right_child',
    'parent_index', 
    'split_feature', 
    'split_gain', 
    'threshold',
    'decision_type', 
    'missing', 
    'missing_type',  
    'value', 
    'weight', 
    'count'
] 


def _filter_cols(trees_df):
    cols = [col for col in COLUMN_ORDER if col in trees_df.columns.tolist()]
    return trees_df[cols]


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

def sklearn_trees_to_dataframe(obj):
    trees =  obj.estimators_
    
    nodes = []
    for tree_index, tree in enumerate(trees):
        tree = tree[0].tree_
        decision_type = '<='
        has_children = (tree.children_left != -1) & (tree.children_right != -1)
        
        df = pd.DataFrame({
            'tree_index'        : [tree_index] * tree.node_count,
            'node_depth'        : sklearn_compute_node_depths(tree),
            #'node_index'        : list(range(0, tree.node_count)),
            'left_child'        : tree.children_left,
            'right_child'       : tree.children_right,
            #'parent_index'      : None, #sklearn_get_parent_idx(tree),
            'split_feature'     : tree.feature,
            #'split_gain'        : None,
            'threshold'         : tree.threshold,
            'decision_type'     : [decision_type if hc else None for hc in has_children],
            #'missing_direction' : None,
            #'missing_type'      : None,
            'value'             : tree.value.ravel(),
            #'weight'            : None,
            'count'             : tree.n_node_samples 
        })
        nodes.append(df)
        
    trees_df = pd.concat(nodes)
    return _filter_cols(trees_df)
    
# =============================================================================
#                                   XGBOOST
# =============================================================================

def xgboost_trees_to_dataframe(obj):
    trees_df = obj.get_booster().trees_to_dataframe()
    
    # Rename columns
    trees_df = \
        trees_df.rename(columns={
            'Tree'     : 'tree_index',
            'ID'       : 'node_index',
            'Feature'  : 'split_feature',
            'Split'    : 'threshold',
            'Yes'      : 'left_child',
            'No'       : 'right_child',
            'Missing'  : 'missing',
            'Gain'     : 'split_gain',
            'Cover'    : 'count',
            'Category' : 'category' 
        })
    
    # Removing Node column
    trees_df = trees_df.drop(columns=['Node'])
    
    # Renaming features if needed
    feature_names = obj.get_booster().feature_names
    
    if feature_names is None:
        trees_df['split_feature'] = \
            trees_df['split_feature'].str[1:] \
            .apply(lambda x: int(x) if x != 'eaf' else None)  # eaf == trimmed Leaf
            
    return _filter_cols(trees_df)


# =============================================================================
#                                  LIGHTGBM
# =============================================================================

def lightgbm_trees_to_dataframe(obj):
    trees_df = obj.booster_.trees_to_dataframe()
    
    # Transform missing_direction
    trees_df['missing'] = np.where(
        trees_df['missing_direction'] == 'left',
        trees_df['left_child'],
        trees_df['right_child']
    )
    trees_df = trees_df.drop(columns=['missing_direction'])
    
    # Transform feature names 
    feature_names = obj.booster_.feature_name()
    
    # Heuristics: Column_0 is a generic name for feature
    if feature_names[0] == 'Column_0':
        trees_df['split_feature'] = \
            trees_df['split_feature'].str[7:] \
            .apply(lambda x: int(x) if x is not None else x)
            
    # TODO: standarize node_index? (i.e. remove L and S)
    return _filter_cols(trees_df)
    
# =============================================================================
#                                  CATBOOST
# =============================================================================

def _catboost_raw_trees(obj):
    # Saving temporary model dump and loading JSON file
    model_name = '_tmp.dump'
    obj.save_model('_tmp.dump', 'json')
    with open(model_name) as f:
        trees_json = json.load(f)
    os.remove(model_name)
    return trees_json

def _catboost_get_splits(trees_json):
    return [split['border'] 
            for tree in  trees_json['oblivious_trees']
            for split in tree['splits']
        ]


def catboost_trees_to_dataframe(obj):
    
    trees_json = _catboost_raw_trees(obj)
    
    # Getting parts of the model
    oblivious_trees = trees_json['oblivious_trees']
    features_info   = trees_json['features_info']
    
    # Parsing model dump
    df = []

    for tree_index, tree in enumerate(oblivious_trees):
        tree_depth = len(tree['splits'])
        last_max = 0
        for node_depth, split in enumerate(tree['splits']):
            level_width = 2 ** node_depth
            next_level_width = 2 ** (node_depth + 1)
            current_max = last_max + level_width
            
            if node_depth == (tree_depth - 1):
                weight = tree['leaf_weights']
                value = tree['leaf_values']
                level_width = len(tree['leaf_values'])
                left_child = [None] * level_width
                right_child = [None] * level_width
            else:
                weight = [None] * level_width
                value = [None] * level_width
                left_child = np.arange(current_max, current_max + next_level_width, 2)
                right_child = np.arange(current_max + 1, current_max + 1 + next_level_width, 2)
            
            node_index = np.arange(last_max, last_max + level_width)
            node_index = [f'{tree_index}-{node_idx}' for node_idx in node_index]
            
            level_nodes = pd.DataFrame({
                'tree_index'        : [tree_index] * level_width,
                'node_depth'        : [tree_index] * level_width,
                'node_index'        : node_index,
                'left_child'        : left_child,
                'right_child'       : right_child,
                #'parent_index'      : None,
                #'split_gain'        : None,
                'split_feature'     : [split['float_feature_index']] * level_width,
                'threshold'         : [split['border']] * level_width,
                #'missing_direction' : None,
                #'missing_type'      : None,
                'value'             : value,
                'weight'            : weight,
                #'count'             : None 
            })  
            
            df.append(level_nodes)
            last_max += level_width
    trees_df = pd.concat(df)     
    return _filter_cols(trees_df)



if __name__ == '__main__':
    pass
    
