# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..base import GBMWrapper
from ..utils import check_is_gbm_regressor

import pdb

# TODO: can be written faster?
# TODO: create lcm_symm
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

class AXIL(BaseEstimator, TransformerMixin):
    """
    Instance importance for regression.
    
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
        # TODO: check is_fitted
        # TODO: we don't have to store the data?
        check_is_gbm_regressor(estimator)
        check_is_fitted(estimator)
        self.wrapped_ = GBMWrapper(estimator)
        super().__init__()
    
    @property
    def estimator(self):
        return self.wrapped_.estimator
    
    def fit(self, X, y, **kwargs):
        # TODO: is the first estimator always mean? (for any loss function?)
        # check with e.g. quantile loss
        # https://github.com/pgeertsema/AXIL_paper/blob/main/axil.py
        
        # number of observations
        N = len(X)
        num_trees = self.wrapped_.n_estimators # n_estimators?
        learning_rate = self.wrapped_.learning_rate
        
        # useful matrices
        ones = np.ones((N,N))
        I = np.identity(N) # ones on the diagonal mean here: 100% of the original value
        
        # Creating matrix of target-leaf membership
        self.lm = self._instance_leaf_membership(X, N)
        
        # Clear list of P matrices (to be used for calculating AXIL weights)
        self.P_list = []
        
        # iterations 0 model predictions (simply average of training data)
        # corresponds to "tree 0"
        # Check if loss is l2
        
        # It makes sense anyway, becuase we don't use the predictions here
        # We only take into account the leaf the particular instance falls in
        P_0 = ones / N
        self.P_list.append(P_0)
        G_accum = P_0 # idea of residuals
        
        # https://docs.scipy.org/doc/scipy/reference/sparse.html
        
        # do iterations for trees 1 to num_trees (inclusive)
        # note, LGB trees ingnores the first (training data mean) predictor, so offset by 1
        for i in range(1, num_trees+1):
            # lm[i] has size (1, N)
            D = lcm(self.lm[i], self.lm[i])
            W = D / D.sum(axis=1)
            resid_pred = I-G_accum
            P = learning_rate * (W @ resid_pred)
            self. P_list.append(P)
            
            
            # On the main diagonal we have only 
            # W = D / (ones @ D)
            
            # D.sum(axis=1) can be considred as 
            # The number of examples in the leaf the n_th example falls in
            # (1, N)
            # Use scparse matrix instead
            # w_ = D / D.sum(axis=1)
            # So w_i is the is a raio we have to multiply the target i by to get 
            # the final prediction of the given leaf
            
            # np.diag(w_).sum() adds up to the number of leaves
            # It means: we still have to cover 0.995 of the prediction 
            # Or: np.diag((I-G_prev))
            
             # This matrices are symmetrical
             # issymmetric(D)
             # issymmetric(W)
             # issymmetric(w_)
             
             # We can then formulate the unit test quite easily: check if multiplying it works
            
            # This matrix is symmetrical
            # It means for the first exameple we're checking with relation with 
            # All the remaining elements. For the second one, we don't need to that for the first element etc.
            # P_alt_1 = W @ (I-G_accum)
            # P_alt_2 = W / np.diag((I-G_accum))
            
            # from scipy.linalg import issymmetric
            # issymmetric(P_alt_1, rtol=0.00000000001)
            # issymmetric(P_alt_1, rtol=0.000000000000001)
            
            # Sens operacji I-G_accum
            # W każdej kolejnej operacji nie liczymy już tak naprawdę średniej 
            # z targetu y_i, ale średnią z jedgo residuów
            # Musimy więc w stsowny sposób przeskalować wartość targetu i
            
            # Co z ujemnymi wartościami? Chyba to działa, bo jest odejmowanie a nie mnożenie
            # W tym przypadku macierz 1 symbolizuje 100% wartości oryginalnej
            
            P = learning_rate * (W @ (I-G_accum))
            self. P_list.append(P)
            
            # Accumulate weights
            G_accum += P
        
        self.trained = True
        return self
    
    def transform(self, X, **kwargs):
        # https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
        
        if not self.trained:
            print("Sorry, you first need to fit to training data. Use the fit() method.")
            return None
        
        # list of P matices
        P = self.P_list
        
        # number of instances in training data used to estimate P
        N, _ = P[0].shape
        
        # number of instances in this data
        S = len(X)
        
        # model instance membership of tree leaves 
        instance_leaf_membership = self.model.predict(data=X, pred_leaf=True)
        
        lm_test = np.concatenate((np.ones((1, S)), instance_leaf_membership.T), axis = 0) + 1
        
        # number of trees in model
        num_trees = self.model.num_trees()
        
        # ones matrix with same dimensions as P
        ones_P = np.ones((N, N))
        
        # ones matrix with same dimensions as L
        ones_L = np.ones((N, S))
        
        # first tree is just sample average
        L = ones_L
        K = (P[0].T @ (L / (ones_P @ L)))
        
        # execute for 1 to num_trees (inclusive)
        for i in range(1, num_trees+1):
            # Wybieramy tylko odpowiednie liście i sumujemy
            L = lcm(self.lm_train[i], lm_test[i])
            K = K + (P[i].T @ (L / (ones_P @ L)))
        
        return X
    
    
    def _instance_leaf_membership(self, X, N):
        """Creates matrix (n_trees, n_instances)"""
        instance_leaf_membership = self.wrapped_.apply(X)
        # Zeroes instead of ones
        return np.concatenate((
            np.ones((1, N)), # First prediction: "one leaf" 
            instance_leaf_membership.T
        ), axis = 0) + 1


if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from lightgbm import LGBMRegressor 
    
    from sklearn.ensemble import GradientBoostingRegressor
    
    from skgbm.xai import AXIL
    
    X, y = make_regression(n_samples=200)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    lgb_regressor = LGBMRegressor()
    lgb_regressor.fit(X_train, y_train)
    
    gbm = GradientBoostingRegressor()
    gbm.fit(X_train, y_train)
    
    axil = AXIL(lgb_regressor)
    axil.fit(X, y)
    
    
    lgb_regressor.st
    
    # Init estimator
    # GradientBoostingRegressor() -> gbm.init_ (domyślnie: DummyEstimator)
    # LGBMRegressor() # boost_from_average
    # 
    
    
        