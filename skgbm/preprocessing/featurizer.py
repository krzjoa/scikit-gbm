#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from ..base import GBMWrapper


class GBMFeaturizer(BaseEstimator, TransformerMixin, GBMWrapper):
    """
    Feature generator for any GBDT model.
    
    Parameters
    ----------
    estimator: object
        A gradient boosting model from scikit-learn, XGBoost, LightGBM or CatBoost library    
    one_hot: bool
        Transform the ouput categorical features using one-hot encoding
    append: bool 
        Append the newly created features to the original ones
    
    References
    ----------

    .. [1] X. He, J. Pan, O. Jin, T. Xu, B. Liu, T. Xu, Y. Shi, A. Atallah,
           R. Herbrich, S. Bowers, J. Q. Candela, `"Practical Lessons from Predicting Clicks on Ads at Facebook"
           <https://research.fb.com/wp-content/uploads/2016/11/practical-lessons-from-predicting-clicks-on-ads-at-facebook.pdf>`_, 2016.

    .. [2] C. Mougan, `"Feature Generation with Gradient Boosted Decision Trees"
           <https://towardsdatascience.com/feature-generation-with-gradient-boosted-decision-trees-21d4946d6ab5>`_, Towards Data Science, 2021.
    
    .. [3] David Masip, `"sktools — Helpers for scikit learn"
           <https://sktools.readthedocs.io/en/latest/sktools.html#module-sktools.preprocessing>`_
    
    .. [4] `xgboostExtension: xgboost Extension for Easy Ranking & TreeFeature
            <https://github.com/bigdong89/xgboostExtension>`_
    
    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from skgbm.preprocessing import GBMFeaturizer
    >>> from lightgbm import LGBMRegressor
    >>>
    >>> X, y = load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> gbm_featurizer = GBMFeaturizer(LGBMRegressor())
    >>> gbm_featurizer.fit(X_train, y_train)
    >>> gbm_featurizer.transform(X_test)
    
    """
    
    def __init__(self, estimator, one_hot: bool = True, append: bool = True):
        self.one_hot = one_hot
        if one_hot:
            self.ohe = OneHotEncoder()
        self.append = append
        super().__init__(estimator)
    
    def fit(self, X, y, **kwargs):
        """
        Fit a GBDT model and OneHotEncoder.
           
        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            A data frame (matrix) of all the features.
        y: array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values.
        
        Returns
        -------
        self: object
            Fitted discretizer.
        """
        super().fit(X, y, **kwargs)
        if hasattr(self, 'ohe'):
            X_ = self.apply(X)
            self.ohe.fit(X_)
        return self
    
    def transform(self, X, **kwargs):
        """
        Return features distiled from the GBM model trees.
        The number of the output features depens on `one_hot` and `append` parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to discretize.
        
        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_trees) or (n_samples, 1) 
            Transformed array.
        """
        output = self.apply(X)
        if hasattr(self, 'ohe'):
            output = self.ohe.transform(output)
        if self.append:
            output = scipy.sparse.hstack([X, output])
        return output
    
    def fit_transform(self, X, y, **kwargs):
        return self.fit(X, y, **kwargs).transform(X, **kwargs)
    


