# coding:utf-8
from mondrianforest import node
import numpy as np


def test_partial_fit():
    np.random.seed(215)
    tree = node.MondrianTree()
    X = np.array([[1.0, 2.0], [0.0, -1.0], [0.0, 3.0]])
    y = np.array([1, 0, 0])
    tree.partial_fit(X, y)
    assert tree.root.parent is None
    assert tree.root.min_list.tolist() == [0, -1]
    assert tree.root.max_list.tolist() == [1, 3]
    assert tree.root.left.is_leaf is True
    assert tree.root.right.is_leaf is False
    assert tree.root.stat.stats[0]['sum'].tolist() == [0, 2]
    assert tree.root.stat.stats[0]['count'] == 2

    proba = tree.predict_proba(np.array([[1.0, 2.0]]))[0]
    assert proba[1] > proba[0]


def test_classifier_stat_update():
    stat = node.Classifier()
    stat.add(np.array([1, 2, 3]), 0)
    stat.add(np.array([1, 2, 4]), 0)
    stat.add(np.array([0, -1, -2]), 1)
    assert stat.stats[0]['sum'].tolist() == [2, 4, 7]
    assert stat.stats[0]['sq_sum'].tolist() == [2, 8, 25]
    assert stat.stats[0]['count'] == 2

    prop = stat.predict_proba(np.array([0, -1, -2]))
    assert prop[1] > prop[0]
