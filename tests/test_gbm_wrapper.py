#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 17:01:06 2023

@author: krzysztof
"""

import pytest

from sklearn.datasets import make_classification, make_regression

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor, XGBRanker
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMRanker
from catboost import CatBoostClassifier, CatBoostRegressor, CatBoostRanker

from skgbm.base import GBM

make_classification = pytest.fixture(make_classification)
make_regression = pytest.fixture(make_regression)


def test_gbm_wrapper_classification(make_classification):
    
    X, y = make_classification
    
    n_estimators = 120
    n_samples = X.shape[0]
    
    # scikit-learn
    gbm_wrapped = GBM(GradientBoostingClassifier(n_estimators=n_estimators))
    gbm_wrapped.fit(X, y)    
    out = gbm_wrapped.apply(X)
    assert out.shape == (n_samples, n_estimators)
    
    # XGBoost
    gbm_wrapped = GBM(XGBClassifier(n_estimators=n_estimators))
    gbm_wrapped.fit(X, y)    
    out = gbm_wrapped.apply(X)
    assert out.shape == (n_samples, n_estimators)
    
    # LightGBM
    gbm_wrapped = GBM(LGBMClassifier(n_estimators=n_estimators))
    gbm_wrapped.fit(X, y)    
    out = gbm_wrapped.apply(X)
    assert out.shape == (n_samples, n_estimators)
    
    # CatBoost
    gbm_wrapped = GBM(CatBoostClassifier(n_estimators=n_estimators))
    gbm_wrapped.fit(X, y)    
    out = gbm_wrapped.apply(X)
    assert out.shape == (n_samples, n_estimators)
    

def test_gbm_wrapper_regression(make_regression):
    
    X, y = make_regression
    
    n_estimators = 120
    n_samples = X.shape[0]
    
    # scikit-learn
    gbm_wrapped = GBM(GradientBoostingRegressor(n_estimators=n_estimators))
    gbm_wrapped.fit(X, y)    
    out = gbm_wrapped.apply(X)
    assert out.shape == (n_samples, n_estimators)
    
    # XGBoost
    gbm_wrapped = GBM(XGBRegressor(n_estimators=n_estimators))
    gbm_wrapped.fit(X, y)    
    out = gbm_wrapped.apply(X)
    assert out.shape == (n_samples, n_estimators)
    
    # LightGBM
    gbm_wrapped = GBM(LGBMRegressor(n_estimators=n_estimators))
    gbm_wrapped.fit(X, y)    
    out = gbm_wrapped.apply(X)
    assert out.shape == (n_samples, n_estimators)
    
    # CatBoost
    gbm_wrapped = GBM(CatBoostRegressor(n_estimators=n_estimators))
    gbm_wrapped.fit(X, y)    
    out = gbm_wrapped.apply(X)
    assert out.shape == (n_samples, n_estimators)
    
    
    
    