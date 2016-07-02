#!/usr/bin/env python
# coding: utf-8
from setuptools import setup, find_packages


setup(
    name='mondrianforest',
    version='0.0.2',
    author='nel215',
    author_email='otomo.yuhei@gmail.com',
    url='https://github.com/nel215/mondrianforest',
    py_modules=['mondrianforest'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    keywords=['machine learning'],
)
