# coding:utf-8
from .node import MondrianTreeClassifier
import numpy as np


class MondrianForestClassifier(object):
    def __init__(self, n_tree):
        self.n_tree = n_tree
        self.trees = []
        self.classes = set()

        for i in range(self.n_tree):
            self.trees.append(MondrianTreeClassifier())

    def fit(self, X, y):
        for label in y:
            self.classes |= {label}
        for tree in self.trees:
            tree.fit(X, y)

    def partial_fit(self, X, y):
        for label in y:
            self.classes |= {label}
        for tree in self.trees:
            tree.partial_fit(X, y)

    def get_params(self, deep):
        return {'n_tree': self.n_tree}

    def predict_proba(self, X):
        return np.sum([tree.predict_proba(X) for tree in self.trees], axis=0) / self.n_tree

    def score(self, X, y):
        probs = self.predict_proba(X)
        classes = np.array([c for c in self.classes])
        correct = 0.0
        for prob, label in zip(probs, y):
            correct += prob.argmax() == (classes == label).argmax()
        return correct / len(X)
