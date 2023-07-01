# -*- coding: utf-8 -*-

import pytest

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import numpy as np

from skgbm.xai import AXIL

RTOL = 0.0001
make_regression = pytest.fixture(make_regression)

def test_axil(make_regression):

    X, y = make_regression
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    n_estimators = 120
    reg_lambda = 0    
    
    for regressor_ in [GradientBoostingRegressor,
                       LGBMRegressor, 
                       CatBoostRegressor,
                       XGBRegressor]:
        
        # Fitting regressor & generating prediction
        if regressor_ is not GradientBoostingRegressor:
            reg = regressor_(n_estimators=n_estimators, reg_lambda=reg_lambda)
        else:
            reg = regressor_(n_estimators=n_estimators)
            
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        
        # Fitting AXIL & generating weights
        axil = AXIL(reg).fit(X_train, y_train)
        k_test = axil.transform(X_test)
        
        assert np.isclose(y_pred, k_test.T @ y_train, rtol=RTOL).all()


# def test_axil_lambda(make_regression):
    
#     X, y = make_regression
#     X_train, X_test, y_train, y_test = train_test_split(X, y)
    
#     n_estimators = 120
#     reg_lambdas = [0.5, 1, 2, 3]
    
#     for reg_lambda in reg_lambdas:
#         for regressor_ in [LGBMRegressor, 
#                            CatBoostRegressor,
#                            XGBRegressor]:
            
#             # Fitting regressor & generating prediction
#             reg = regressor_(n_estimators=n_estimators, reg_lambda=reg_lambda)
                
#             reg.fit(X_train, y_train)
#             y_pred = reg.predict(X_test)
            
#             # Fitting AXIL & generating weights
#             axil = AXIL(reg).fit(X_train, y_train)
#             k_test = axil.transform(X_test)
            
#             assert np.isclose(y_pred, k_test.T @ y_train, rtol=RTOL).all()
            
