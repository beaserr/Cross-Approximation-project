import numpy as np
from sklearn.datasets import load_iris


def load_iris_data():
    data = load_iris()
    return data.data


def kernel_matrix(X):
    X = np.asarray(X)
    return X @ X.T


def gaussian_kernel(x, y, sigma=1.0):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.exp(-np.sum((x - y) ** 2) / (sigma ** 2))


def gaussian_kernel_matrix(X, sigma=1.0):
    X = np.asarray(X, dtype=float)
    s = np.sum(X * X, axis=1, keepdims=True)
    d2 = s + s.T - (X @ X.T)
    return np.exp(- d2 / (sigma ** 2))
