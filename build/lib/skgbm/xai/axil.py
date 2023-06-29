# -*- coding: utf-8 -*-
# Original version by Paul Geertsema 2022 (https://github.com/pgeertsema/AXIL_paper/blob/main/axil.py)

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from numba import jit

from ..base import GBM
from ..utils import check_is_gbm_regressor, \
    is_lightgbm, is_catboost, is_sklearn_gbm


LIGHTGBM_RMSE_LOSS = [
    'regression', 
    'regression_l2', 'l2', 
    'mean_squared_error', 
    'mse', 
    'l2_root', 
    'root_mean_squared_error', 
    'rmse'
]


def check_is_supported_by_axil(estimator):
    """For the moment, AXIL only accepts models trained with RMSE."""
    
    # if is_xgboost(estimator):
    #     raise Exception("XGBoost regressors are supported yet.")
    
    # Common interface to get loss function
    accepted = True
    if is_lightgbm(estimator):
        if estimator.objective_ in LIGHTGBM_RMSE_LOSS:
            accepted = True
    elif is_catboost(estimator):
        if estimator.get_param('loss_function') == 'RMSE':
            # We're not able to verify bias!!!
            accepted = True
    elif is_sklearn_gbm(estimator):
        if estimator.loss == 'squared_error' and estimator.init_.strategy == 'mean':
            accepted = True
    
    if not accepted:
        raise Exception("Passed estimator uses loss functions other than RMSE or non-standard initial guess.")
        
        
def check_uses_lambda_reg(wrapped_estimator):
    if wrapped_estimator.reg_lambda > 0:
        raise Exception("For the moment, the AXIL explainer only handles cases with regularization parameter lambda = 0.")
        
        
def check_uses_bagging(wrapped_estimator):
    if wrapped_estimator.sample < 1:
        raise Exception("For the moment, the AXIL explainer doens't handle models with bagging turned on.")
    


# Create own symmetrical matrix???

# TODO: can be written faster?
# TODO: create lcm_symm
@jit(nopython=True)
def lcm(vector1, vector2):
    '''
    utility function to create leaf coincidence matrix L from leaf membership vectors vector1 (train) and vector2 (test)
    element l_s,f in L takes value of 1 IFF observations v1 and v2 are allocated to the same leaf (for v1 in vector1 and v2 in vector2)
    that is, vector1[v1] == vector2[v2], otherwise 0

    Input arguments vector1 and vector2 are list-like, that is, support indexing and len()
    Output L is a python matrix of dimension (len(vector1), len(vector2))
    '''
    # tree contains a list of predicted values
    N1 = len(vector1)
    N2 = len(vector2)

    L = np.full((N1, N2), np.nan)
    for v1 in range(0, N1):
        for v2 in range(0, N2):
            same = (vector1[v1] == vector2[v2])
            L[v1][v2] = same

    return L

# Inherit GBMWrapper (GBM)
class AXIL(BaseEstimator, TransformerMixin, GBM):
    """
    Additive eXplanations with Instance Loadings - instance importance for regression.
    
    A method being perhaps a first representative of a novel class of XAI techniques: instance-based explanations.
    It's application to Gradient Boosted Decsion Trees is based on the following theorem:
        
    ... Any GBM regression prediction is a linear combination of training data targets y.
    
    For the moment, it only accepts the models with reg_lambda = 0 and sample = 1 (no bagging).
    
    
    Parameters
    ----------
    estimator: object
        A gradient boosting model from scikit-learn, XGBoost, LightGBM or CatBoost library   
    
    References
    ----------
    .. [1] P. Geertsema and  H. Lu, `"Instance-based Explanations for Gradient Boosting Machine Predictions with AXIL Weights"
           <https://arxiv.org/abs/2301.01864>`_, 
           2009.
    .. [2] `AXIL_paper GitHub repository <https://github.com/pgeertsema/AXIL_paper>`_
    
    Examples
    --------
    >>> from lightgm import LGBMRegressor
    >>> from skgbm.xai import AXIL
    >>> from sklearn.datasets import fetch_data
    >>>
    >>> X, y = fetch_data()
    >>> 
    
    """    
    
    def __init__(self, estimator: object):
        check_is_gbm_regressor(estimator)
        check_is_fitted(estimator)
        check_is_supported_by_axil(estimator)
        super().__init__(estimator)
        
        # Checks on wrapped_estimator
        check_uses_lambda_reg(self.estimator)
        check_uses_bagging(self.estimator)
    
    
    def fit(self, X, y=None, **kwargs):
        """
        Parameters
        ----------
        X:
        
        """
        # check with e.g. quantile loss
        # https://github.com/pgeertsema/AXIL_paper/blob/main/axil.py
        
        # Make a table out of it:
        # * scikit- learn: GradientBoostingRegressor has no lambda parameter
        # * catboost: lambda is applied to leaves, but not to initial guess
        # * xgboost: lambda is applied to both leaves and 
        
        # number of observations
        N = len(X)
        num_trees = self.n_estimators # n_estimators?
        learning_rate = self.learning_rate
        
        # useful matrices
        ones = np.ones((N,N))
        I = np.identity(N) # ones on the diagonal mean here: 100% of the original value
        
        # Creating matrix of target-leaf membership
        self.lm_train = self._instance_leaf_membership(X, N)
        
        # Clear list of P matrices (to be used for calculating AXIL weights)
        self.P_list = []
        
        # iterations 0 model predictions (simply average of training data)
        # corresponds to "tree 0"
        P_0 = ones / N
        # P_0 = ones / (N + self.reg_lambda)
        # P_0 = np.zeros((N,N)) #<- this one is totally wrong
        
        # if is_xgboost(self.estimator) or is_lightgbm(self.estimator):
        #     P_0 = ones / (N + self.reg_lambda)
        # else:
        #     P_0 = ones / N
            
        self.P_list.append(P_0)
        G_accum = P_0 # idea of residuals
        
        # https://docs.scipy.org/doc/scipy/reference/sparse.html
        
        # do iterations for trees 1 to num_trees (inclusive)
        # note, LGB trees ingnores the first (training data mean) predictor, so offset by 1
        for i in range(1, num_trees+1):
            
            # TODO: do use sparse matrices
            # D and W are symmetric issymmetric(D, rtol=0.00001)
            # lm[i] has size (1, N)
            D = lcm(self.lm_train[i], self.lm_train[i])
            
            # Default regularization parameters:
            # * LightGBM: 0
            # * sklearn: -
            # * CatBoost: 3
            # * XGBoost: 1
            # W = D / (D.sum(axis=1) + self.reg_lambda) 
            
            W = D / D.sum(axis=1) # originally: W = D / (ones @ D)
      
            # Resid coef is a coefficient, that allows us transition from 
            # original targets to residuals at n-th iteration
            resid_coef = I-G_accum
            P = learning_rate * (W @ resid_coef)
            self. P_list.append(P)
            
            # Accumulate weights
            G_accum += P
        
        return self
    
    def transform(self, X, **kwargs):
       
        check_is_fitted(self, 'P_list')

        # list of P matices
        P = self.P_list
        
        # number of instances in training data used to estimate P
        N, _ = P[0].shape
        
        # number of instances in this data
        S = len(X)
        
        # model instance membership of tree leaves 
        lm_test = self._instance_leaf_membership(X, S)
        
        # number of trees in model
        num_trees = self.n_estimators
        
        # ones matrix with same dimensions as P
        ones_P = np.ones((N, N))
        
        # ones matrix with same dimensions as L
        ones_L = np.ones((N, S))
        
        # first tree is just sample average
        L = ones_L
        K = (P[0].T @ (L / (ones_P @ L)))
        
        # execute for 1 to num_trees (inclusive)
        for i in range(1, num_trees+1):
            L = lcm(self.lm_train[i], lm_test[i])
            # Nans in CatBoost - check the exact reason
            K = K + np.nan_to_num((P[i].T @ (L / (ones_P @ L))), 0)
        
        return K
    
    
    def _instance_leaf_membership(self, X, N):
        """Creates matrix (n_trees, n_instances)"""
        instance_leaf_membership = self.apply(X)
        # Zeroes instead of ones
        return np.concatenate((
            np.ones((1, N)), # First prediction: "one leaf" 
            instance_leaf_membership.T
        ), axis = 0) + 1
    
    


if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from lightgbm import LGBMRegressor 
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    
    from sklearn.ensemble import GradientBoostingRegressor
    
    from skgbm.xai import AXIL
    import numpy as np
    
    X, y = make_regression(n_samples=200)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    gbm = LGBMRegressor(reg_lambda=1).fit(X_train, y_train)
    gbm = GradientBoostingRegressor().fit(X_train, y_train)
    gbm = CatBoostRegressor(reg_lambda=0, subsample=1).fit(X_train, y_train)
    gbm = XGBRegressor(reg_lambda=1).fit(X_train, y_train)
    
    # https://stats.stackexchange.com/questions/372634/boosting-and-bagging-trees-xgboost-lightgbm
    # https://stackoverflow.com/questions/48011742/xgboost-leaf-scores
    # https://www.riskified.com/resources/article/boosting-comparison/
    
    axil = AXIL(gbm)
    axil.fit(X_train)
    
    # Take regularization into account
    
    # LightGBM: https://github.com/microsoft/LightGBM/blob/9edea60edc5db7785508d0f5981890ac1e1b430a/src/treelearner/feature_histogram.hpp#L991
    # 
     
    k_test = axil.transform(X_test)
    y_pred = gbm.predict(X_test)
    
    print(y_pred)
    print(k_test.T @ y_train)
    
    np.isclose(y_pred, k_test.T @ y_train, rtol=0.0001).all()
    
    # Waht is bayesian_matrix_reg in catboost?

    
        