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
    d2 = s + s.T - 2.0 * (X @ X.T)
    return np.exp(-np.maximum(d2, 0.0) / (2.0 * sigma ** 2))


def low_rank_psd_noise(n, R, xi, seed=0):
    np.random.seed(seed)
    D = np.zeros(n)
    D[:R] = 1.0
    A_signal = np.diag(D)
    G = np.random.randn(n, n)
    W = G @ G.T / n
    return A_signal + xi * W


def poly_decay_matrix(n, R, p):
    diag = np.ones(n)
    for k in range(n - R):
        diag[R + k] = (k + 2) ** (-p)
    return np.diag(diag)


def exp_decay_matrix(n, R, q):
    diag = np.ones(n)
    for k in range(n - R):
        diag[R + k] = 10 ** (-(k + 1) * q)
    return np.diag(diag)


def generate_test_matrices(n=1000, seed=0):
    np.random.seed(seed)
    U, _ = np.linalg.qr(np.random.rand(n, n))
    V, _ = np.linalg.qr(np.random.rand(n, n))
    A1 = U @ exp_decay_matrix(n, R=5, q=0.15) @ V.T
    A2 = U @ poly_decay_matrix(n, R=15, p=1.8) @ V.T
    A3 = low_rank_psd_noise(n, R=10, xi=0.5, seed=seed)
    return A1, A2, A3
