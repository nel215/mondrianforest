# coding:utf-8
import mondrianforest
import numpy as np
from sklearn import datasets, cross_validation, svm


def test_regression():
    np.random.seed(215)
    boston = datasets.load_boston()
    tree = mondrianforest.MondrianTreeRegressor()
    cv = cross_validation.ShuffleSplit(len(boston.data), n_iter=10, test_size=0.05)
    scores = cross_validation.cross_val_score(tree, boston.data, boston.target, cv=cv, scoring='mean_squared_error')
    assert scores.mean() > -65
