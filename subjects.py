import numpy as np
from sklearn.datasets import load_iris


# Iris dataset

def load_iris_data():
    data = load_iris()
    X = data.data
    y = data.target
    return X, y, data



# Simple kernel
# K[i, j] = X[i]  X[j]
def kernel_matrix(X):
    X = np.asarray(X, dtype=float)
    return X @ X.T

# Low-rank matrix 
def low_rank_psd_noise(n, R, xi):
    D = np.zeros(n)
    D[:R] = 1.0
    A_signal = np.diag(D)
    G = np.random.randn(n, n)
    W = G @ G.T / n
    return A_signal + xi * W


# Polynomial decay 
def poly_decay_matrix(n, R, p):
    diag = np.ones(n)
    for k in range(n - R):
        diag[R + k] = (k + 2) ** (-p)
    return np.diag(diag)

#Exponential decay
def exp_decay_matrix(n, R, q):
    diag = np.ones(n)
    for k in range(n - R):
        diag[R + k] = 10 ** (-(k + 1) * q)
    return np.diag(diag)

# Generating test matrices

def test_matrices(n=150):

    # random orthogonal matrices
    U, _ = np.linalg.qr(np.random.rand(n, n))
    V, _ = np.linalg.qr(np.random.rand(n, n))

    # different matrix families
    A_1 = U @ exp_decay_matrix(n, R=5, q=0.15) @ V.T
    A_2 = U @ poly_decay_matrix(n, R=15, p=1.8) @ V.T
    A_3 = low_rank_psd_noise(n, R=10, xi=0.5)

    return A_1, A_2, A_3
