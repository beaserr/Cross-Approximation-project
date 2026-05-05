import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris



def load_iris_data():
    data = load_iris()
    return data.data

def kernel_matrix(X):
    X = np.asarray(X, dtype=float)
    return X @ X.T
  
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


def generate_test_matrices(n=150, seed=0):
    np.random.seed(seed)
    U, _ = np.linalg.qr(np.random.rand(n, n))
    V, _ = np.linalg.qr(np.random.rand(n, n))
    A1 = U @ exp_decay_matrix(n, R=5, q=0.15) @ V.T
    A2 = U @ poly_decay_matrix(n, R=15, p=1.8) @ V.T
    A3 = low_rank_psd_noise(n, R=10, xi=0.5, seed=seed)
    return A1, A2, A3


def fpCA(A, max_rank, epsilon=1e-12):
    m, n = A.shape
    R = A.copy()
    U = np.zeros((m, max_rank))
    V = np.zeros((n, max_rank))
    errors = []
    normA = np.linalg.norm(A, 'fro')

    for k in range(max_rank):
        idx = np.argmax(np.abs(R))
        i = idx // n  
        j = idx % n    
        piv = R[i, j]

        if abs(piv) < epsilon:
            break

        u = R[:, j]
        v = R[i, :] / piv
        U[:, k] = u
        V[:, k] = v
        R -= np.outer(u, v)
        S = U[:, :k+1] @ V[:, :k+1].T
        err = np.linalg.norm(A - S, 'fro') / normA
        errors.append(err)
    return errors


def ppCA(A, max_rank, epsilon=1e-12):
    m, n = A.shape
    U = np.zeros((m, max_rank))
    V = np.zeros((n, max_rank))
    errors = []
    normA = np.linalg.norm(A, 'fro')
    pivot_row = np.random.randint(0, n)

    for k in range(max_rank):
        b = A[pivot_row, :].copy()
        for mu in range(k):
            b -= U[pivot_row, mu] * V[:, mu]
            
        pivot_col = np.argmax(np.abs(b))
        piv = b[pivot_col]

        if abs(piv) < epsilon:
            break

        a = A[:, pivot_col].copy()
        for mu in range(k):
            a -= U[:, mu] * V[pivot_col, mu]

        a /= piv
        U[:, k] = a
        V[:, k] = b
        pivot_row = np.argmax(np.abs(a))

        S = U[:, :k+1] @ V[:, :k+1].T
        err = np.linalg.norm(A - S, 'fro') / normA
        errors.append(err)   
    return errors

def ppCA2(A, max_rank, epsilon=1e-12):
    m, n = A.shape
    U = np.zeros((m, max_rank))
    V = np.zeros((n, max_rank))
    errors = []
    normA = np.linalg.norm(A, 'fro')
    pivot_row = 0

    for k in range(max_rank):
        for _ in range(m):
            b = A[pivot_row, :].copy()
            for mu in range(k):
                b -= U[pivot_row, mu] * V[:, mu]
            pivot_col = np.argmax(np.abs(b))
            piv = b[pivot_col]

            if abs(piv) > epsilon:
                break

            pivot_row = (pivot_row + 1) % m
        else:
            break

        a = A[:, pivot_col].copy()
        for mu in range(k):
            a -= U[:, mu] * V[pivot_col, mu]
        a /= piv

        U[:, k] = a
        V[:, k] = b

        pivot_row = np.argmax(np.abs(a))
        S = U[:, :k+1] @ V[:, :k+1].T
        err = np.linalg.norm(A - S, 'fro') / normA
        errors.append(err)
    return errors


def func_ppca(A_func, m, n, max_rank, epsilon=1e-12):
    U = np.zeros((m, max_rank))
    V = np.zeros((n, max_rank))
    errors = []
    normA = np.sqrt(sum(A_func(i, j)**2 for i in range(m) for j in range(n)))

    pivot_row = 0

    for k in range(max_rank):
        b = np.array([A_func(pivot_row, j) for j in range(n)])

        for mu in range(k):
            b -= U[pivot_row, mu] * V[:, mu]

        pivot_col = np.argmax(np.abs(b))
        piv = b[pivot_col]

        if abs(piv) < epsilon:
            break
        a = np.array([A_func(i, pivot_col) for i in range(m)])
        for mu in range(k):
            a -= U[:, mu] * V[pivot_col, mu]
        a /= piv
        U[:, k] = a
        V[:, k] = b
        pivot_row = np.argmax(np.abs(a))
        S_err = 0
        for i in range(m):
            for j in range(n):
                s = 0
                for mu in range(k + 1):
                    s += U[i, mu] * V[j, mu]
                diff = A_func(i, j) - s
                S_err += diff * diff

        err = np.sqrt(S_err) / normA
        errors.append(err)
    return errors

def svd_error(A, max_rank):
    s = np.linalg.svd(A, compute_uv=False)
    normA = np.sqrt(np.sum(s**2))
    errors = []
    for k in range(1, max_rank + 1):
        err = np.sqrt(np.sum(s[k:]**2)) / normA
        errors.append(err)
    return errors

X = load_iris_data()
K = kernel_matrix(X)

max_rank = 40
fca_err = fpCA(K, max_rank)
pca2_err = ppCA2(K, max_rank)
ppca_err = ppCA(K, max_rank)

r = min(len(fca_err), max_rank)
svd_err = svd_error(K, r)

plt.figure(figsize=(8, 5))
plt.semilogy(range(1, r+1), svd_err, label="SVD")
plt.semilogy(range(1, len(fca_err)+1), fca_err, label="FCA")
plt.semilogy(range(1, len(ppca_err)+1), ppca_err, label="ppCA")

plt.title("Iris kernel matrix approximation")
plt.xlabel("Rank")
plt.ylabel("Relative Frobenius error")
plt.legend()
plt.grid(True)
plt.show()

A1, A2, A3 = generate_test_matrices()

for i, A in enumerate([A1, A2, A3], start=1):
    fca_err = fpCA(A, max_rank)
    ppca_err = ppCA(A, max_rank)
    ppca2_err = ppCA2(A, max_rank)

    r = min(len(fca_err), max_rank)
    svd_err = svd_error(A, r)
    plt.figure(figsize=(8, 5))
    plt.semilogy(range(1, r+1), svd_err, label="SVD")
    plt.semilogy(range(1, len(fca_err)+1), fca_err, label="FCA")
    plt.semilogy(range(1, len(ppca_err)+1), ppca_err, label="ppCA")
    plt.semilogy(range(1, len(ppca2_err)+1), ppca2_err, label="ppCA2")

    plt.title(f"Test matrix A{i}")
    plt.xlabel("Rank")
    plt.ylabel("Relative Frobenius error")
    plt.legend()
    plt.grid(True)
    plt.show()
