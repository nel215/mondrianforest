# coding:utf-8


class RegressorResult(object):
    def __init__(self, avg):
        self.avg = avg

    def merge(self, r):
        return RegressorResult(self.avg + r.avg)

    def get(self):
        return self.avg


class Regressor(object):
    def __init__(self):
        self.sum = 0
        self.count = 0

    def add(self, x, y):
        self.sum += y
        self.count += 1

    def merge(self, r):
        res = Regressor()
        res.sum = self.sum + r.sum
        res.count = self.count + r.count
        return res

    def predict(self, x):
        if self.count == 0:
            return 0
        return self.sum / self.count

    def create_result(self, x, w):
        return RegressorResult(self.predict(x)*w)
