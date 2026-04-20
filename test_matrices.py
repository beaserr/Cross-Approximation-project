def low_rank_psd_noise(n, R, xi):
    D = np.zeros(n)
    D[:R] = 1.0
    A_signal = np.diag(D)
    # Wishart noise 
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

