import numpy as np


def low_rank_psd_noise(n, R, xi, seed=0):
    # Low-rank signal plus PSD noise.
    np.random.seed(seed)
    D = np.zeros(n)
    D[:R] = 1.0
    A_signal = np.diag(D)
    G = np.random.randn(n, n)
    W = G @ G.T / n
    return A_signal + xi * W


def poly_decay_matrix(n, R, p):
    # Polynomial singular value decay.
    diag = np.ones(n)
    for k in range(n - R):
        diag[R + k] = (k + 2) ** (-p)
    return np.diag(diag)


def exp_decay_matrix(n, R, q):
    # Exponential singular value decay.
    diag = np.ones(n)
    for k in range(n - R):
        diag[R + k] = 10 ** (-(k + 1) * q)
    return np.diag(diag)


def generate_test_matrices(n=1000, seed=0):
    # Rotate diagonal test matrices.
    np.random.seed(seed)
    U, _ = np.linalg.qr(np.random.rand(n, n))
    V, _ = np.linalg.qr(np.random.rand(n, n))
    A1 = U @ exp_decay_matrix(n, R=5, q=0.15) @ V.T
    A2 = U @ poly_decay_matrix(n, R=15, p=1.8) @ V.T
    A3 = low_rank_psd_noise(n, R=10, xi=0.5, seed=seed)
    return A1, A2, A3
