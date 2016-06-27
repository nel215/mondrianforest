# coding:utf-8
import mondrianforest
import numpy as np
from sklearn import datasets, svm, cross_validation, naive_bayes


def test_iris():
    np.random.seed(215)
    iris = datasets.load_iris()
    tree = mondrianforest.MondrianTreeClassifier()
    cv = cross_validation.ShuffleSplit(len(iris.data), n_iter=3, test_size=0.2)
    scores = cross_validation.cross_val_score(tree, iris.data, iris.target, cv=cv)
    assert scores.mean() > 0.9
