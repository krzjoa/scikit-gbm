#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 22:57:10 2023

@author: krzysztof
"""

import pandas as pd

from .base import GBMWrapper

def trees_to_dataframe(obj) -> pd.DataFrame:
    """
    A common interface to fetch trees as pandas DataFrame
    
    Parameters
    ----------
    obj: object
        A GBM model from XGBoost, LightGBM, CatBoost or scikit-learn GradientBoosting* model
    
    Returns
    -------
    trees_df: pd.DataFrame
        A pandas DataFrame containing information about all the trees in the model    
    
    """
    wrapped_gbm = GBMWrapper(obj)
    return wrapped_gbm.trees_to_dataframe()