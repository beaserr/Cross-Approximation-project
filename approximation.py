import numpy as np


# ============================================================
# Approximation functions
# These functions only compute U and V.
# They do not compute the error.
# ============================================================

def fpCA_approx(A, max_rank, epsilon=1e-12):
    m, n = A.shape
    R = A.copy()
    U = np.zeros((m, max_rank))
    V = np.zeros((n, max_rank))

    actual_rank = 0

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

        actual_rank = k + 1

    return U[:, :actual_rank], V[:, :actual_rank]


def ppCA_approx(A, max_rank, epsilon=1e-12):
    m, n = A.shape
    U = np.zeros((m, max_rank))
    V = np.zeros((n, max_rank))

    pivot_row = np.random.randint(0, m)
    actual_rank = 0

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

        a_next = np.abs(a.copy())
        a_next[pivot_row] = 0
        pivot_row = np.argmax(a_next)

        actual_rank = k + 1

    return U[:, :actual_rank], V[:, :actual_rank]


# ============================================================
# Adaptive ppCA kept close to your original code
# ============================================================

def ppCA_adaptive(A, max_rank, epsilon=1e-12):
    m, n = A.shape
    U = np.zeros((m, max_rank))
    V = np.zeros((n, max_rank))
    errors = []
    pivot_row = np.random.randint(0, m)

    normA = np.linalg.norm(A, 'fro')
    a1_norm = None
    b1_norm = None

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

        S = U[:, :k + 1] @ V[:, :k + 1].T
        err = np.linalg.norm(A - S, 'fro') / normA
        errors.append(err)

        if k == 0:
            a1_norm = np.linalg.norm(a, 2)
            b1_norm = np.linalg.norm(b, 2)

        if np.linalg.norm(a, 2) * np.linalg.norm(b, 2) <= epsilon * a1_norm * b1_norm:
            break

        a_next = np.abs(a.copy())
        a_next[pivot_row] = 0
        pivot_row = np.argmax(a_next)

    return errors


def func_ppca(A_func, m, n, max_rank, epsilon=1e-12):
    U = np.zeros((m, max_rank))
    V = np.zeros((n, max_rank))
    errors = []

    normA = np.sqrt(sum(A_func(i, j) ** 2 for i in range(m) for j in range(n)))

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

        a_next = np.abs(a.copy())
        a_next[pivot_row] = 0
        pivot_row = np.argmax(a_next)

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


def func_ppca_adaptive(A_func, m, n, max_rank, epsilon=1e-12):
    A = np.array([[A_func(i, j) for j in range(n)] for i in range(m)])
    return ppCA_adaptive(A, max_rank, epsilon)
