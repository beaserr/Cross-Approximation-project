#The function does cross approximation algorithm with partial pivoting and returns the frobenius error computed

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
