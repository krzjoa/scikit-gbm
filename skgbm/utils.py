# -*- coding: utf-8 -*-

# TODO: check if object is raw booster

# =============================================================================
#                                  XGBOOST
# =============================================================================

XGB_CLASSIFIER = "XGBClassifier"
XGB_REGRESSOR =  "XGBRegressor"
XGB_RANKER =  "XGBRanker"

XGBOOST_ESTIMATORS = [
   XGB_CLASSIFIER,
   XGB_REGRESSOR,
   XGB_RANKER
]

# =============================================================================
#                                  LIGHTGBM
# =============================================================================

LGB_CLASSIFIER = "LGBMClassifier"
LGB_REGRESSOR =  "LGBMRegressor"
LGB_RANKER =  "LGBMRanker"

LIGHTGBM_ESTIMATORS = [
   LGB_CLASSIFIER,
   LGB_REGRESSOR,
   LGB_RANKER
]

# =============================================================================
#                                  CATBOOST
# =============================================================================

CATBOOST_CLASSIFIER =  "CatBoostClassifier"
CATBOOST_REGRESSOR  =  "CatBoostRegressor"
CATBOOST_RANKER     =  "CatBoostRanker"


CATBOOST_ESTIMATORS = [
   CATBOOST_CLASSIFIER,
   CATBOOST_REGRESSOR,
   CATBOOST_RANKER
]

# =============================================================================
#                                SCIKIT-LEARN
# =============================================================================

SKLEARN_CLASSIFIER = "GradientBoostingClassifier"
SKLEARN_REGRESSOR = "GradientBoostingRegressor",

SKLEARN_ESTIMATORS = [
    SKLEARN_CLASSIFIER,
    SKLEARN_REGRESSOR
]

# =============================================================================
#                              ALL ESTIMATORS
# =============================================================================

VALIED_ESTIMATORS = \
    XGBOOST_ESTIMATORS + \
    LIGHTGBM_ESTIMATORS + \
    CATBOOST_ESTIMATORS + \
    SKLEARN_ESTIMATORS

# =============================================================================
#                               ALL REGRESSORS
# =============================================================================


ALL_REGRESSORS = [
    XGB_REGRESSOR,
    LGB_REGRESSOR,
    CATBOOST_REGRESSOR,
    SKLEARN_REGRESSOR
]


def check_estimator(estimator):
    str_type = type(estimator).__name__ 
    assert str_type in VALIED_ESTIMATORS, \
        f'Passed object {str_type} is not one of the valid class'

def is_xgboost(estimator):
    return type(estimator).__name__  in XGBOOST_ESTIMATORS

def is_lightgbm(estimator):
    return type(estimator).__name__  in LIGHTGBM_ESTIMATORS

def is_catboost(estimator):
    return type(estimator).__name__ in CATBOOST_ESTIMATORS

def is_sklearn_gbm(estimator):
    return type(estimator).__name__ in SKLEARN_ESTIMATORS        

def check_is_gbm_regressor(estimator):
    str_type = type(estimator).__name__ 
    assert str_type in ALL_REGRESSORS, \
        f'Passed object {str_type} is not a GBM regressor'