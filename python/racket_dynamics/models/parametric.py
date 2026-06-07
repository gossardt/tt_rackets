import numpy as np


def piecewise_lin_fn(x, a, b, c, d):
    x = np.asarray(x)
    return np.where(x < c, a * x + d, b * x + (a - b) * c + d)

