import time

import numpy as np


# ============================================================
# Error analysis functions
# These functions only compute the relative Frobenius errors.
# The approximation U,V is already computed before.
# ============================================================

def fpCA_error_analysis(A, U, V):
    # Relative errors for fpCA factors.
    errors = []
    normA = np.linalg.norm(A, 'fro')

    for k in range(1, U.shape[1] + 1):
        S = U[:, :k] @ V[:, :k].T
        err = np.linalg.norm(A - S, 'fro') / normA
        errors.append(err)

    return errors


def ppCA_error_analysis(A, U, V):
    # Relative errors for ppCA factors.
    errors = []
    normA = np.linalg.norm(A, 'fro')

    for k in range(1, U.shape[1] + 1):
        S = U[:, :k] @ V[:, :k].T
        err = np.linalg.norm(A - S, 'fro') / normA
        errors.append(err)

    return errors


def time_approximation(method, A, max_rank, epsilon=1e-12):
    # Time only the approximation step.
    start = time.perf_counter()
    U, V = method(A, max_rank, epsilon)
    end = time.perf_counter()

    approximation_time = end - start
    return U, V, approximation_time


def time_error_analysis(method, A, U, V):
    # Time only the error computation.
    start = time.perf_counter()
    errors = method(A, U, V)
    end = time.perf_counter()

    error_time = end - start
    return errors, error_time


def svd_error(A, max_rank):
    # Best possible Frobenius errors.
    s = np.linalg.svd(A, compute_uv=False)
    normA = np.sqrt(np.sum(s ** 2))
    errors = []

    for k in range(1, max_rank + 1):
        err = np.sqrt(np.sum(s[k:] ** 2)) / normA
        errors.append(err)

    return errors
