#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from .base import GBMWrapper

def trees_to_dataframe(obj) -> pd.DataFrame:
    """
    A common interface to fetch trees as pandas DataFrame
    
    DaraFrame columns for all the models share their names, but they differ when it comes
    to the exact set of available parameters. 
    
    +-----------------+-----------+-------------+------------+--------------+
    |                 | XGBoost   |  LightGBM   |  CatBoost  | scikit-learn |         
    +=================+===========+=============+============+==============+
    | tree_index      | ✅        | ✅          | ✅         | ✅           |
    +-----------------+-----------+-------------+------------+--------------+
    | node_depth      | ❌        | ✅          | ✅         | ✅           |
    +-----------------+-----------+-------------+------------+--------------+
    | node_index      | ✅        | ✅          | ✅         | ❌           |
    +-----------------+-----------+-------------+------------+--------------+
    | left_child      | ✅        | ✅          | ✅         | ✅           |
    +-----------------+-----------+-------------+------------+--------------+
    | right_child     | ✅        | ✅          | ✅         | ✅           |
    +-----------------+-----------+-------------+------------+--------------+
    | parent_index    | ❌        | ✅          | ❌         | ❌           |
    +-----------------+-----------+-------------+------------+--------------+
    | split_feature   | ✅        | ✅          | ✅         | ✅           |
    +-----------------+-----------+-------------+------------+--------------+
    | split_gain      | ✅        | ✅          | ❌         | ❌           |
    +-----------------+-----------+-------------+------------+--------------+
    | threshold       | ✅        | ✅          | ✅         | ✅           |
    +-----------------+-----------+-------------+------------+--------------+
    | decision_type   | ❌        | ✅          | ❌         | ✅           |
    +-----------------+-----------+-------------+------------+--------------+
    | missing         | ✅        | ✅          | ❌         | ❌           |
    +-----------------+-----------+-------------+------------+--------------+
    | missing_type    | ❌        | ✅          | ❌         | ❌           |
    +-----------------+-----------+-------------+------------+--------------+
    | value           | ❌        | ✅          | ✅         | ✅           |
    +-----------------+-----------+-------------+------------+--------------+
    | weight          | ❌        | ✅          | ✅         | ❌           |
    +-----------------+-----------+-------------+------------+--------------+
    | count           | ✅        | ✅          | ❌         | ✅           |
    +-----------------+-----------+-------------+------------+--------------+
    
    
    Parameters
    ----------
    obj: object
        An XGBoost, LightGBM, CatBoost or scikit-learn GradientBoosting* model
    
    Returns
    -------
    trees_df: pd.DataFrame
        A pandas DataFrame containing information about all the trees in the model    
    
    Examples
    --------
    >>> from skgbm.tools import trees_to_dataframe
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> X, y = make_regression()
    >>> gb_reg = GradientBoostingRegressor().fit(X, y)
    >>> gb_df = trees_to_dataframe(gb_reg)
    """
    return GBMWrapper(obj).trees_to_dataframe()