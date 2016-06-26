# coding:utf-8
import numpy as np


class ClassifierStat(object):
    def __init__(self):
        self.count = 0
        self.sum = None

    def add(self, x):
        if self.sum is None:
            self.sum = np.zeros(len(x))
        self.sum += x
        self.count += 1


class Node(object):
    def __init__(self, min_list, max_list, tau, is_leaf, parent=None, delta=None, xi=None):
        self.parent = parent
        self.tau = tau
        self.is_leaf = is_leaf
        self.min_list = min_list
        self.max_list = max_list
        self.delta = delta
        self.xi = xi
        self.left = None
        self.right = None

    def get_parent_tau(self):
        if self.parent is None:
            return 0.0
        return self.parent.tau

    def __repr__(self):
        return "<mondrianforest.Node tau={} min_list={} max_list={} is_leaf={}>".format(
            self.tau,
            self.min_list,
            self.max_list,
            self.is_leaf,
        )


# TODO: extends BaseClassifier
class MondrianTree(object):
    def __init__(self):
        self.root = None

    def create_leaf(self, x, parent):
        return Node(
            min_list=x.copy(),
            max_list=x.copy(),
            is_leaf=True,
            tau=1e9,
            parent=parent,
        )

    def extend_mondrian_block(self, node, x):
        '''
            return root of sub-tree
        '''
        e_min = np.maximum(node.min_list - x, 0)
        e_max = np.maximum(x - node.max_list, 0)
        e_sum = e_min + e_max
        rate = np.sum(e_sum) + 1e-9
        E = np.random.exponential(1.0/rate)
        if node.get_parent_tau() + E < node.tau:
            e_sample = np.random.rand() * np.sum(e_sum)
            delta = (e_sum.cumsum() > e_sample).argmax()
            if x[delta] > node.min_list[delta]:
                xi = np.random.uniform(node.min_list[delta], x[delta])
            else:
                xi = np.random.uniform(x[delta], node.max_list[delta])
            parent = Node(
                min_list=np.minimum(node.min_list, x),
                max_list=np.maximum(node.max_list, x),
                is_leaf=False,
                tau=node.get_parent_tau() + E,
                parent=node.parent,
                delta=delta,
                xi=xi,
            )
            sibling = self.create_leaf(x, parent=parent)
            if x[parent.delta] <= parent.xi:
                parent.left = sibling
                parent.right = node
                node.parent = parent
            else:
                parent.left = node
                parent.right = sibling
                node.parent = parent
            return parent
        else:
            node.min_list = np.minimum(x, node.min_list)
            node.max_list = np.maximum(x, node.max_list)
            if not node.is_leaf:
                if x[node.delta] <= node.xi:
                    node.left = self.extend_mondrian_block(node.left, x)
                else:
                    node.right = self.extend_mondrian_block(node.right, x)
            return node

    def partial_fit(self, X, y):
        for x, label in zip(X, y):
            if self.root is None:
                self.root = self.create_leaf(x, parent=None)
            else:
                self.root = self.extend_mondrian_block(self.root, x)
