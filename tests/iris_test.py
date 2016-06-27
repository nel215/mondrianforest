# coding:utf-8
import mondrianforest
import numpy as np
from sklearn import datasets


def test_iris():
    np.random.seed(215)
    iris = datasets.load_iris()
    tree = mondrianforest.MondrianTree()
    tree.partial_fit(np.array(iris.data), iris.target)
