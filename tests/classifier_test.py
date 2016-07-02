# coding:utf-8
import mondrianforest
import numpy as np


def test_classifier_stat_update():
    stat = mondrianforest.Classifier()
    stat.add(np.array([1, 2, 3]), 0)
    stat.add(np.array([1, 2, 4]), 0)
    stat.add(np.array([0, -1, -2]), 1)
    assert stat.stats[0]['sum'].tolist() == [2, 4, 7]
    assert stat.stats[0]['sq_sum'].tolist() == [2, 8, 25]
    assert stat.stats[0]['count'] == 2

    prob = stat.predict_proba(np.array([0, -1, -2]))
    sum_prob = 0.0
    for k, v in prob.items():
        sum_prob += v
    assert abs(v - 1.0) < 1e-9
    assert prob[1] > prob[0]
