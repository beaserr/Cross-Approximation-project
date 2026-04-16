import numpy as np
import matplotlib.pyplot as plt


# Full pivoting cross approximation


def cross_approx(A, maxi, epsilon=1e-12):
    m, n = A.shape
    R = A.copy()
    U = np.zeros((m, maxi))
    V = np.zeros((n, maxi))
    errors_fro = []
    norm_fro = np.linalg.norm(A, 'fro')

    for k in range(maxi):
        idx = np.argmax(np.abs(R))
        pivot_row = idx // n
        pivot_col = idx % n
        piv = R[pivot_row, pivot_col]
        if abs(piv) < epsilon:
            break

        u = R[:, pivot_col].copy()
        v = R[pivot_row, :].copy() / piv

        U[:, k] = u
        V[:, k] = v

        R -= np.outer(u, v)
        S = U[:, :k+1] @ V[:, :k+1].T
        errors_fro.append(np.linalg.norm(A - S, 'fro') / norm_fro)
    k_final = len(errors_fro)
    return errors_fro


def partial_cross_approx(A, maxi, epsilon=1e-12):
    m, n = A.shape
    R = A.copy()
    U = np.zeros((m, maxi))
    V = np.zeros((n, maxi))
    errors_fro = []

    norm_fro = np.linalg.norm(A, 'fro')
    pivot_row = np.random.randint(m)  

    for k in range(maxi):
        pivot_col = np.argmax(np.abs(R[pivot_row, :]))
        pivot_row = np.argmax(np.abs(R[:, pivot_col]))
        piv = R[pivot_row, pivot_col]

        if abs(piv) < epsilon:
            break

        u = R[:, pivot_col].copy()
        v = R[pivot_row, :].copy() / piv
        U[:, k] = u
        V[:, k] = v

        R -= np.outer(u, v)

        S = U[:, :k+1] @ V[:, :k+1].T
        errors_fro.append(np.linalg.norm(A - S, 'fro') / norm_fro)
        idx = np.argmax(np.abs(R))
        pivot_row = idx // n

    k_final = len(errors_fro)
    return errors_fro


def frob_error_svd(A, max_rank):
    s = np.linalg.svd(A, compute_uv=False)
    norm_fro = np.sqrt(np.sum(s**2))

    return [
        np.sqrt(np.sum(s[k:]**2)) / norm_fro
        for k in range(1, max_rank + 1)
    ]


# different test matrices

def low_rank_psd_noise(n, R, xi):
    # signal
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

np.random.seed(0)
n = 150 
max_rank = 40  
U, _ = np.linalg.qr(np.random.rand(n, n))
V, _ = np.linalg.qr(np.random.rand(n, n))

A_1 = U @ exp_decay_matrix(n, R=5, q=0.15) @ V.T
A_2 = U @ poly_decay_matrix(n, R=15, p=1.8) @ V.T
A_3 = low_rank_psd_noise(n, R=10, xi=0.5) @ V.T

# matrices parameters

         
  


matrices = [
    ("Step with noise (R=10, xi=5e-2)", A_3),
    ("Polynomial decay (R=15, p=1.8)", A_2),
    ("Exponential decay (R=5, q=0.15)", A_1),
]



fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i in range(3):
    name, A = matrices[i]
    ax = axes[i]
    ca = cross_approx(A, max_rank)
    pa = partial_cross_approx(A, max_rank)
    max_r = min(max(len(ca), len(pa)), max_rank)
    svd_err = frob_error_svd(A, max_r)

    r = np.arange(1, len(svd_err) + 1)

    ax.semilogy(r, svd_err, label="SVD")
    ax.semilogy(range(1, len(ca)+1), ca, label="Full Pivoting Cross Approximation")
    ax.semilogy(range(1, len(pa)+1), pa, label="Partial Cross Approximation")

    ax.set_title(name)
    ax.grid(True)

axes[0].legend()

plt.show()
