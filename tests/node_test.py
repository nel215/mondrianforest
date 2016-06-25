# coding:utf-8
from mondrianforest import node
import numpy as np


def test_partial_fit():
    np.random.seed(215)
    tree = node.MondrianTree()
    X = np.array([[1.0, 2.0], [0.0, -1.0]])
    y = np.array([1, 0])
    tree.partial_fit(X, y)
    assert tree.root.parent is None
    assert tree.root.min_list.tolist() == [0, -1]
    assert tree.root.max_list.tolist() == [1, 2]
