import numpy as np

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
