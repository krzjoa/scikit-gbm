# -*- coding: utf-8 -*-

# TODO: check if object is raw booster

XGBOOST_ESTIMATORS = [
    "<class 'xgboost.sklearn.XGBRegressor'>",
    "<class 'xgboost.sklearn.XGBClassifier'>",
    "<class 'xgboost.sklearn.XGBRanker'>"
]

LIGHTGBM_ESTIMATORS = [
    "<class 'lightgbm.sklearn.LGBMRegressor'>",
    "<class 'lightgbm.sklearn.LGBMClassifier'>",
    "<class 'lightgbm.sklearn.LGBMRanker'>"
]


CATBOOST_ESTIMATORS = [
    "<class 'catboost.core.CatBoostRegressor'>"
    "<class 'catboost.core.CatBoostClassifier'>",
    "<class 'catboost.core.CatBoostRanker'>"
]

SKLEARN_ESTIMATORS = [
    "<class 'sklearn.ensemble._gb.GradientBoostingRegressor'>",
    "<class 'sklearn.ensemble._gb.GradientBoostingClassifier'>"
]

VALIED_ESTIMATORS = \
    XGBOOST_ESTIMATORS + \
    LIGHTGBM_ESTIMATORS + \
    CATBOOST_ESTIMATORS + \
    SKLEARN_ESTIMATORS


def check_estimator(estimator):
    str_type = str(type(estimator)) 
    assert str_type in VALIED_ESTIMATORS, \
        f'Passed object {str_type} is not one of the valid class'

def is_xgboost(estimator):
    return str(type(estimator)) in XGBOOST_ESTIMATORS

def is_lightgbm(estimator):
    return str(type(estimator)) in LIGHTGBM_ESTIMATORS

def is_catboost(estimator):
    return str(type(estimator)) in CATBOOST_ESTIMATORS


def is_sklearn_gbm(estimator):
    return str(type(estimator)) in SKLEARN_ESTIMATORS        
        