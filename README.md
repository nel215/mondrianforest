# Mondrian Forest

An online random forest implementaion written in Python.

[![Build Status](https://travis-ci.org/nel215/mondrianforest.svg?branch=master)](https://travis-ci.org/nel215/mondrianforest)
[![PyPI](https://img.shields.io/pypi/v/mondrianforest.svg)](https://pypi.python.org/pypi/mondrianforest)

## Usage

```python
import mondrianforest
from sklearn import datasets, cross_validation

iris = datasets.load_iris()
forest = mondrianforest.MondrianForestClassifier(n_tree=10)
cv = cross_validation.ShuffleSplit(len(iris.data), n_iter=20, test_size=0.10)
scores = cross_validation.cross_val_score(forest, iris.data, iris.target, cv=cv)
print(scores.mean(), scores.std())
```

## License

mondrianforest is licensed under the MIT license.<br/>
Copyright (c) 2016 nel215

## References

### Papers

- [Balaji Lakshminarayanan, Daniel M. Roy, Yee Whye Teh, Mondrian Forests: Efficient Online Random Forests, Advances in Neural Information Processing Systems 27 (NIPS), pages 3140-3148, 2014](http://arxiv.org/abs/1406.2673)
- [Balaji Lakshminarayanan, Daniel M. Roy, Yee Whye Teh, Mondrian Forests for Large-Scale Regression when Uncertainty Matters, Proceedings of the 19th International Conference on Artificial Intelligence and Statistics (AISTATS) 2016, Cadiz, Spain. JMLR: W&CP volume 51](https://arxiv.org/abs/1506.03805)
- [Matej Balog, Yee Whye Teh, The Mondrian Process for Machine Learning](http://arxiv.org/abs/1507.05181)

### Slides

- [Mondrian Forests](https://project.inria.fr/bnpsi/files/2015/07/balaji.pdf)

### Videos
- [Mondrian forests: Efficient random forests for streaming data via Bayesian nonparametrics](http://videolectures.net/sahd2014_teh_mondrian_forests/)

### Code

- [balajiln/mondrianforest](https://github.com/balajiln/mondrianforest)
