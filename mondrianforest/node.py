# coding:utf-8
import numpy as np


class ClassifierStat(object):
    def __init__(self):
        self.stats = {}

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
        res = ClassifierStat()
        labels = set(self.stats.keys()) | set(s.stats.keys())
        for label in labels:
            res.stats[label] = {}
            if label in self.stats and label in s.stats:
                res.stats[label]['count'] = self.stats[label]['count'] + s.stats[label]['count']
                res.stats[label]['sum'] = self.stats[label]['sum'] + s.stats[label]['sum']
            elif label in self.stats:
                res.stats[label]['count'] = self.stats[label]['count']
                res.stats[label]['sum'] = self.stats[label]['sum']
            else:
                res.stats[label]['count'] = s.stats[label]['count']
                res.stats[label]['sum'] = s.stats[label]['sum']
        return res

    def __repr__(self):
        return "<mondrianforest.ClassifierStat stats={}".format(
            self.stats,
        )


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
        self.stat = ClassifierStat()

    def update_leaf(self, x, label):
        self.stat.add(x, label)

    def update_internal(self):
        self.stat = self.left.stat.merge(self.right.stat)

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

    def create_leaf(self, x, label, parent):
        leaf = Node(
            min_list=x.copy(),
            max_list=x.copy(),
            is_leaf=True,
            tau=1e9,
            parent=parent,
        )
        leaf.update_leaf(x, label)
        return leaf

    def extend_mondrian_block(self, node, x, label):
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
            sibling = self.create_leaf(x, label, parent=parent)
            if x[parent.delta] <= parent.xi:
                parent.left = sibling
                parent.right = node
            else:
                parent.left = node
                parent.right = sibling
            node.parent = parent
            parent.update_internal()
            return parent
        else:
            node.min_list = np.minimum(x, node.min_list)
            node.max_list = np.maximum(x, node.max_list)
            if not node.is_leaf:
                if x[node.delta] <= node.xi:
                    node.left = self.extend_mondrian_block(node.left, x, label)
                else:
                    node.right = self.extend_mondrian_block(node.right, x, label)
                node.update_internal()
            else:
                node.update_leaf(x, label)
            return node

    def partial_fit(self, X, y):
        for x, label in zip(X, y):
            if self.root is None:
                self.root = self.create_leaf(x, label, parent=None)
            else:
                self.root = self.extend_mondrian_block(self.root, x, label)
