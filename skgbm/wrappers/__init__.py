# -*- coding: utf-8 -*-

from typing import Final

from .sklearn_wrapper import _SklearnWrapper
from .xgboost_wrapper import _XgboostWrapper
from .lightgbm_wrapper import _LightgbmWrapper
from .catboost_wrapper import _CatboostWrapper


name_map: Final = {
    
    # scikit-learn
    'GradientBoostingClassifier' : _SklearnWrapper,
    'GradientBoostingRegressor'  : _SklearnWrapper,
    
    # XGBoost
    'XGBClassifier' : _XgboostWrapper,
    'XGBRegressor'  : _XgboostWrapper,
    'XGBRanker'     : _XgboostWrapper, # ?
    
    # LightGBM
    'LGBMClassifier' : _LightgbmWrapper,
    'LGBMRegressor'  : _LightgbmWrapper,
    'LGBMRanker'     : _LightgbmWrapper,
    
    # CatBoost
    'CatBoostClassifier' : _CatboostWrapper,
    'CatBoostRegressor'  : _CatboostWrapper,
    'CatBoostRanker'     : _CatboostWrapper
}


def wrap_estimator(obj):
    return name_map[type(obj).__name__](obj)