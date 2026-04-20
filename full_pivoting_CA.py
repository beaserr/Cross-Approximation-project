import numpy as np
import matplotlib.pyplot as plt

#The function cross_approx does cross approximation with full pivoting, and returns frobenius error of the algorithm


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

