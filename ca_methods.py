import numpy as np

def fpCA(A, max_rank, epsilon=1e-12):
    A = np.array(A)
    m, n = A.shape
    R = A.copy()
    U = np.zeros((m, max_rank))
    V = np.zeros((n, max_rank))
    errors = []
    normA = np.linalg.norm(A, "fro")

    for k in range(max_rank):
        idx = np.argmax(np.abs(R))
        pivot_row, pivot_col = divmod(idx, n)
        piv = R[pivot_row, pivot_col]

        if abs(piv) < epsilon:
            break
        u = R[:, pivot_col].copy()
        v = R[pivot_row, :].copy() / piv

        U[:, k] = u
        V[:, k] = v
        R -= np.outer(u, v)
        S = U[:, :k + 1] @ V[:, :k + 1].T
        err = np.linalg.norm(A - S, "fro") / normA
        errors.append(err)
    r = len(errors)
    return errors, U[:, :r], V[:, :r]


def ppCA(A, max_rank, epsilon=1e-12):
    m, n = A.shape
    U = np.zeros((m, max_rank))
    V = np.zeros((n, max_rank))
    errors = []
    normA = np.linalg.norm(A, 'fro')
    pivot_row = 0

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
        r = len(errors)
    return errors, U[:, :r], V[:, :r]


def frob_error_svd(A, max_rank):
    A = np.array(A)
    s = np.linalg.svd(A, compute_uv=False)
    normA = np.sqrt(np.sum(s**2))
    errors = []
    for k in range(1, max_rank + 1):
        err = np.sqrt(np.sum(s[k:] ** 2)) / normA
        errors.append(err)
    return errors
