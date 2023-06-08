#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 23:02:39 2023

@author: krzysztof
"""

import pytest

import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor, XGBRanker
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMRanker
from catboost import CatBoostClassifier, CatBoostRegressor, CatBoostRanker

from skgbm.functional import trees_to_dataframe

make_classification = pytest.fixture(make_classification)
make_regression = pytest.fixture(make_regression)


def test_gbm_wrapper_classification(make_classification):
    
    classes = [
        GradientBoostingClassifier,
        LGBMClassifier,
        CatBoostClassifier
    ]
    
    X, y = make_classification
    
    n_estimators = 120
    n_samples = X.shape[0]
    
    for class_ in classes:
        model = class_(n_estimators=n_estimators)
        model.fit(X, y)   
        trees_df = trees_to_dataframe(model)
        assert type(trees_df) is pd.DataFrame
    

def test_gbm_wrapper_regression(make_regression):
    
    classes = [
        GradientBoostingRegressor,
        LGBMRegressor,
        CatBoostRegressor
    ]
    
    X, y = make_regression
    
    n_estimators = 120
    n_samples = X.shape[0]
    
    for class_ in classes:
        model = class_(n_estimators=n_estimators)
        model.fit(X, y)   
        trees_df = trees_to_dataframe(model)
        assert type(trees_df) is pd.DataFrame
    