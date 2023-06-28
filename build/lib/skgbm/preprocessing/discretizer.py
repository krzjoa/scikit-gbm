#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import scipy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import clone

from feature_engine.discretisation import ArbitraryDiscretiser

from ..base import GBMWrapper
from ..tools import trees_to_dataframe
from ..trees_extraction import _catboost_raw_trees, _catboost_get_splits

try:
    import catboost
    CATBOOST_CLASSES = [
        catboost.CatBoostRegressor,
        catboost.CatBoostClassifier,
        catboost.CatBoostRanker
    ]
except:
    # If there is no CatBoost, any CatBoost model can't be passed anyway
    CATBOOST_CLASSES = []



class GBMDiscretizer(BaseEstimator, TransformerMixin, GBMWrapper):
    """
    Feature discretizer based on GBDT.
    
    Internally, it uses `ArbitraryDiscretiser <https://feature-engine.trainindata.com/en/1.0.x/discretisation/ArbitraryDiscretiser.html>`_
    to handle discretization step after finding the optimal thresholds.
    
    Parameters
    ----------
    estimator: object
        A gradient boosting model from scikit-learn, XGBoost, LightGBM or CatBoost library    
    one_hot: bool
        Transform the ouput categorical features using one-hot encoding
    columns: list of str
        List of column names to be transformed
    append: bool 
        Append the newly created features to the original ones
        
    References
    ----------
    .. [1] K. Semsch, `"My contribution to the tidymodels ecosystem - 
           implementing supervised discretization step with XgBoost backend" 
           <https://konradsemsch.netlify.app/2020/05/my-contribution-to-tidymodels-ecosystem-implementing-supervised-discretization-step-with-xgboost-backend/>`_, 2020.
    .. [2] A. Berrado and  G. C. Runger, `"Supervised multivariate discretization in mixed data with Random Forests"
           <https://ieeexplore.ieee.org/document/5069327>`_, 
           2009.
    .. [3] H. Maïssae, `ForestDisc: Forest Discretization (R package) 
           <https://cran.r-project.org/web/packages/ForestDisc/index.html>`_, 2022.
    
    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from skgbm.preprocessing import GBMDiscretizer
    >>> from xgboost import XGBClassifier
    >>>
    >>> iris = load_iris()
    >>> data = pd.DataFrame(
    >>>        data= np.c_[iris['data'], iris['target']],
    >>>        columns= iris['feature_names'] + ['target']
    >>>  )
    >>> data.columns = data.columns.str[:-5]
    >>> data.columns = data.columns.str.replace(' ', '_')
    >>>    
    >>> # Data splitting
    >>> X, y = data.iloc[:, :4], data.iloc[:, 4:]
    >>> X_train, X_test, y_train, y_test = train_test_split(
    >>>         X, y, test_size=0.3, random_state=0
    >>> )
    >>> X_cols = X.columns.tolist()
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> gbm_discretizer = GBMDiscretizer(CatBoostClassifier(verbose=0), 
    >>>                                  X_cols, one_hot=False)
    >>> X_train_disc = gbm_discretizer.fit_transform(X_train, y_train)
    >>> #      sepal_length  sepal_width  petal_length  petal_width
    >>> # 60              7            0             9            5
    >>> # 116            22            9            29           13
    >>> # 144            24           12            31           20
    >>> # 119            17            1            24           10
    >>> # 108            24            4            32           13
    >>> # ..            ...          ...           ...          ...
    >>> # 9               6           10             4            0
    >>> # 103            20            8            30           13
    >>> # 67             15            6            15            5
    >>> # 117            32           17            38           17
    >>> # 47              3           11             3            1
    """
    
    def __init__(self, estimator, 
                 columns: list, 
                 one_hot: bool = True, 
                 append: bool = False):
        self.one_hot = one_hot
        if one_hot:
            self.ohe = OneHotEncoder()
        self.append = append
        self.columns = columns
        self.estimators_ = {}
        self.discretizer_ = None
        super().__init__(estimator)
    
    def fit(self, X, y, **kwargs):
        """
        Fit a set GBDT models (one per each discretized feature), distil split thresholds from them
        and create an internal `ArbitraryDiscretiser <https://feature-engine.trainindata.com/en/1.0.x/discretisation/ArbitraryDiscretiser.html>`_.
        instance based on those values.
           
        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            A data frame (matrix) of all the features.
        y: array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (this is a supervised transformation).
        
        Returns
        -------
        self: object
            Fitted discretizer.
        """
                    
        # Fitting estimators (one per tranformed column)
        disc_thresholds_ = {}
        
        for col in self.columns:

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.estimators_[col] = est_ =  \
                    clone(self.estimator).fit(X[[col]], y)
            
            # Getting the data frame for CatBoost is redundant
            if type(est_) not in CATBOOST_CLASSES:
                trees = trees_to_dataframe(est_)
                splits = trees \
                    .sort_values('threshold')['threshold'] \
                    .drop_duplicates() \
                    .dropna() \
                    .tolist()
            else:
                trees = _catboost_raw_trees(est_)
                splits = _catboost_get_splits(trees)
                splits = np.sort(np.unique(splits)).tolist()
            
            splits = [-np.inf] + splits + [np.inf]
            disc_thresholds_[col] = splits
        
        self.discretizer_ = \
            ArbitraryDiscretiser(binning_dict=disc_thresholds_)
            
        X_disc = self.discretizer_.fit_transform(X)
        
        if hasattr(self, 'ohe'):
            self.ohe.fit(X_disc)
            
        return self
    
    def transform(self, X, **kwargs):
        """
        Discretize the specified subset of columns.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to discretize.
        
        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        # TODO: one_hot encoding to GBMWrapper
        output = self.discretizer_.transform(X)
        if hasattr(self, 'ohe'):
            output = self.ohe.transform(output)
        if self.append:
            output = scipy.sparse.hstack([X, output])
        return output
    
    def fit_transform(self, X, y, **kwargs):
        return self.fit(X, y, **kwargs).transform(X, **kwargs)
    
    @property       
    def binner_dict_(self):
        return self.discretizer_.binner_dict_
        
    

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    
    import pandas as pd
    import numpy as np
    
    # Loading data
    iris = load_iris()
    # https://stackoverflow.com/questions/38105539/how-to-convert-a-scikit-learn-dataset-to-a-pandas-dataset
    data = pd.DataFrame(
        data= np.c_[iris['data'], iris['target']],
        columns= iris['feature_names'] + ['target']
    )
    data.columns = data.columns.str[:-5]
    data.columns = data.columns.str.replace(' ', '_')
    
    # Data splitting
    X, y = data.iloc[:, :4], data.iloc[:, 4:]
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=0)
    X_cols = X.columns.tolist()
    
    gbm_discretizer = GBMDiscretizer(XGBClassifier(), X_cols, one_hot=False)
    gbm_discretizer = GBMDiscretizer(GradientBoostingClassifier(), X_cols, one_hot=False)
    gbm_discretizer = GBMDiscretizer(LGBMClassifier(), X_cols, one_hot=False)
    
    gbm_discretizer.fit_transform(X_train, y_train)
    
    
    

