# scikit-gbm

[![Documentation Status](https://readthedocs.org/projects/scikit-gbm/badge/?version=latest)](https://scikit-gbm.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/scikit-gbm.svg)](https://badge.fury.io/py/scikit-gbm)

scikit-learn compatible tools to work with GBM models

## Installation

```
pip install scikit-gbm

# or 

pip install git+https://github.com/krzjoa/scikit-gbm.git
```

## Usage

Fo the moment, you can find the following tools in the library:

* `GBMFeaturizer`
* `GBMDiscretizer`
* `trees_to_dataframe`

Take a look at the [documentation](https://scikit-gbm.readthedocs.io/en/latest/?badge=latest) to learn more.
A simple example, how to use `GBMFeaturizer` in a classification task.

```python

# Classification
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from skgbm.preprocessing import GBMFeaturizer
from lightgbm import LGBMRegressor
from xgboost import XGBClassifier

X, y = make_classification()
X_train, X_test, y_train, y_test = train_test_split(X, y)

pipeline = \
    Pipeline([
        ('gbm_featurizer', GBMFeaturizer(XGBClassifier())),
        ('logistic_regression', LogisticRegression())
    ])

# Try also:
# ('gbm_featurizer', GBMFeaturizer(GradientBoostingClassifier())),
# ('gbm_featurizer', GBMFeaturizer(LGBMClassifier())),
# ('gbm_featurizer', GBMFeaturizer(CatBoostClassifier())),

# Predictions for the test set
pipeline_pred = pipeline.predict(X_test)
```
