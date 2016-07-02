# coding:utf-8
import numpy as np


class ClassifierResult(object):
    def __init__(self, probs):
        self.probs = probs

    def merge(self, r):
        probs = {}
        for label in (set(r.probs.keys()) | set(self.probs.keys())):
            probs[label] = 0.0
            if label in self.probs:
                probs[label] += self.probs[label]
            if label in r.probs:
                probs[label] += r.probs[label]
        return ClassifierResult(probs)

    def get(self):
        return self.probs


class Classifier(object):
    def __init__(self):
        self.stats = {}

    def create_result(self, x, w):
        probs = self.predict_proba(x)
        for label in probs.keys():
            probs[label] *= w
        return ClassifierResult(probs)

    def add(self, x, label):
        if label not in self.stats:
            self.stats[label] = {
                'sum': np.zeros(len(x)),
                'sq_sum': 0,
                'count': 0,
            }
        self.stats[label]['sum'] += x
        self.stats[label]['sq_sum'] += x*x
        self.stats[label]['count'] += 1

    def merge(self, s):
        res = Classifier()
        labels = set(self.stats.keys()) | set(s.stats.keys())
        for label in labels:
            res.stats[label] = {}
            if label in self.stats and label in s.stats:
                res.stats[label]['sum'] = self.stats[label]['sum'] + s.stats[label]['sum']
                res.stats[label]['sq_sum'] = self.stats[label]['sq_sum'] + s.stats[label]['sq_sum']
                res.stats[label]['count'] = self.stats[label]['count'] + s.stats[label]['count']
            elif label in self.stats:
                res.stats[label]['sum'] = self.stats[label]['sum']
                res.stats[label]['sq_sum'] = self.stats[label]['sq_sum']
                res.stats[label]['count'] = self.stats[label]['count']
            else:
                res.stats[label]['sum'] = s.stats[label]['sum']
                res.stats[label]['sq_sum'] = s.stats[label]['sq_sum']
                res.stats[label]['count'] = s.stats[label]['count']
        return res

    def predict_proba(self, x):
        res = {}
        sum_prob = 0.0
        for label, stat in self.stats.items():
            # TODO: case that var is 0 and count <= 1
            avg = stat['sum']/stat['count']
            var = stat['sq_sum']/stat['count'] - avg*avg + 1e-9
            sig = stat['count']*var/(stat['count'] - 1 + 1e-9)
            z = np.power(2.0*np.pi, len(x))*np.linalg.norm(sig)
            prob = np.exp(-0.5 * np.dot(x-avg, x-avg) / np.dot(sig, sig)) / z
            sum_prob += prob
            res[label] = prob
        for label in res.keys():
            res[label] /= sum_prob
        return res

    def __repr__(self):
        return "<mondrianforest.Classifier stats={}".format(
            self.stats,
        )
