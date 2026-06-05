import numpy as np
from sklearn.datasets import load_iris


def load_iris_data():
    # Small real data set.
    data = load_iris()
    return data.data


def kernel_matrix(X):
    # Linear kernel.
    X = np.asarray(X)
    return X @ X.T


def gaussian_kernel(x, y, sigma=1.0):
    # Gaussian kernel for two points.
    x = np.asarray(x)
    y = np.asarray(y)
    return np.exp(-np.sum((x - y) ** 2) / (sigma ** 2))


def gaussian_kernel_matrix(X, sigma=1.0):
    # Vectorized Gaussian kernel matrix.
    X = np.asarray(X, dtype=float)
    s = np.sum(X * X, axis=1, keepdims=True)
    d2 = s + s.T - 2.0 * (X @ X.T)
    return np.exp(-np.maximum(d2, 0.0) / (2.0 * sigma ** 2))
