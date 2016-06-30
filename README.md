# Mondrian Forest

An online random forest implementaion written in Python.

[![Build Status](https://travis-ci.org/nel215/mondrianforest.svg?branch=master)](https://travis-ci.org/nel215/mondrianforest)

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

- [Mondrian Forests for Large-Scale Regression when Uncertainty Matters](https://arxiv.org/abs/1506.03805)

### Slides

- [Mondrian Forests](https://project.inria.fr/bnpsi/files/2015/07/balaji.pdf)

### Videos
- [Mondrian forests: Efficient random forests for streaming data via Bayesian nonparametrics](http://videolectures.net/sahd2014_teh_mondrian_forests/)

### Code

- [balajiln/mondrianforest](https://github.com/balajiln/mondrianforest)
