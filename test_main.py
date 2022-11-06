"""
Test goes here

"""

from mylib.calculator import add
import numpy as np
from mylib.sklearnlib import linearRegression


def test_add():
    assert add(1, 2) == 3


def test_sklearn_linear_reg():
    x = np.arange(10)
    k, b = 2, 1
    y = k * x + b
    X = x[:, np.newaxis]
    reg = linearRegression(X, y)
    assert abs(reg.coef_[0] - k) < 1e-6
    assert abs(reg.intercept_ - b) < 1e-6
