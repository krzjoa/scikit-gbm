# scikit-gbm

[![Documentation Status](https://readthedocs.org/projects/scikit-gbm/badge/?version=latest)](https://scikit-gbm.readthedocs.io/en/latest/?badge=latest)

scikit-learn compatible tools to work with GBM models

## Installation

```
pip install scikit-gbm

# or 

pip install git+https://github.com/krzjoa/scikit-gbm.git
```

## Usage

For the moment, the only available class is `GBMFeaturezier`. It's a wrapper around
scikit-learn GBMs, XGBoost, LightGBM and CatBoost models. 

```python

# Classification
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from skgbm.preprocessing import GBMFeaturizer
from lightgbm import LGBMRegressor
from xgboost import XGBClassifier

X, y = make_classification()
# train_test_split

pipeline = \
    Pipeline([
        ('gbm_featurizer', GBMFeaturizer(XGBClassifier())),
        ('logistic_regression', LogisticRegression())
    ])

# Try also:
# ('gbm_featurizer', GBMFeaturizer(GradientBoostingClassifier())),
# ('gbm_featurizer', GBMFeaturizer(LGBMClassifier())),
# ('gbm_featurizer', GBMFeaturizer(CatBoostClassifier())),


# Regression
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Try also:
# ('gbm_featurizer', GBMFeaturizer(GradientBoostingClassifier())),
# ('gbm_featurizer', GBMFeaturizer(LGBMClassifier())),
# ('gbm_featurizer', GBMFeaturizer(CatBoostClassifier())),

pipeline = \
    Pipeline([
        ('gbm_featurizer', GBMFeaturizer(XGBClassifier())),
        ('logistic_regression', LogisticRegression())
    ])

# Training
pipeline.fit(X_train, y_train)

# Predictions for the test set
pipeline_pred = pipeline.predict(X_test)


```