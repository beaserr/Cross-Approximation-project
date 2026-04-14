import numpy as np
import matplotlib.pyplot as plt



# FULL CROSS APPROXIMATION 
def cross_approx(A, maxi, epsilon=1e-12):
    m, n = A.shape
    R = A.copy()

    U = np.zeros((m, maxi))
    V = np.zeros((n, maxi))

    errors_fro = []
    errors_2 = []

    norm_fro = np.linalg.norm(A, 'fro')
    norm_2 = np.linalg.norm(A, 2)

    for k in range(maxi):
        pivot_row, pivot_col = np.unravel_index(np.argmax(np.abs(R)), R.shape)
        max_val = R[pivot_row, pivot_col]

        if abs(max_val) < epsilon:
            break

        u = R[:, pivot_col].copy()
        v = R[pivot_row, :].copy() / max_val

        U[:, k] = u
        V[:, k] = v

        R -= np.outer(u, v)

        S = U[:, :k+1] @ V[:, :k+1].T
        errors_fro.append(np.linalg.norm(A - S, 'fro') / norm_fro)
        errors_2.append(np.linalg.norm(A - S, 2) / norm_2)

    k_final = len(errors_fro)
    return U[:, :k_final], V[:, :k_final], errors_fro, errors_2


# PARTIAL CROSS APPROXIMATION 

def partial_cross_approx(A, maxi, epsilon=1e-12):
    m, n = A.shape
    R = A.copy()

    U = np.zeros((m, maxi))
    V = np.zeros((n, maxi))

    errors_fro = []
    errors_2 = []

    norm_fro = np.linalg.norm(A, 'fro')
    norm_2 = np.linalg.norm(A, 2)
    pivot_row = np.argmax(np.max(np.abs(R), axis=1))

    for k in range(maxi):

        pivot_col = np.argmax(np.abs(R[pivot_row, :]))

        if abs(R[pivot_row, pivot_col]) < epsilon:
            break

        pivot_row = np.argmax(np.abs(R[:, pivot_col]))

        u = R[:, pivot_col].copy()
        v = R[pivot_row, :].copy() / R[pivot_row, pivot_col]

        U[:, k] = u
        V[:, k] = v

        R -= np.outer(u, v)

        S = U[:, :k+1] @ V[:, :k+1].T

        errors_fro.append(np.linalg.norm(A - S, 'fro') / norm_fro)
        errors_2.append(np.linalg.norm(A - S, 2) / norm_2)

    k_final = len(errors_fro)
    return U[:, :k_final], V[:, :k_final], errors_fro, errors_2



# SVD ERROR
def frob_error_svd(A, max_rank):
    s = np.linalg.svd(A, compute_uv=False)
    norm_fro = np.sqrt(np.sum(s**2))

    return [
        np.sqrt(np.sum(s[k:]**2)) / norm_fro
        for k in range(1, max_rank + 1)
    ]


# matrix
m, n = 50, 50
r = 50
max_rank = 10

np.random.seed(42)
U_rand, _ = np.linalg.qr(np.random.rand(m, m))
V_rand, _ = np.linalg.qr(np.random.rand(n, n))

R_eff = 10
eps = 1e-3

sigma_exp = np.ones(r)
sigma_exp[R_eff:] = 10.0 ** (-0.25 * np.arange(1, r - R_eff + 1))

sigma_poly = np.ones(r)
sigma_poly[R_eff:] = 1.0 / np.arange(1, r - R_eff + 1)

sigma_step = np.ones(r)
sigma_step[R_eff:] = eps

A_exp = U_rand[:, :r] @ np.diag(sigma_exp) @ V_rand[:, :r].T
A_poly = U_rand[:, :r] @ np.diag(sigma_poly) @ V_rand[:, :r].T
A_step = U_rand[:, :r] @ np.diag(sigma_step) @ V_rand[:, :r].T



_, _, ca_fro_exp, _ = cross_approx(A_exp, max_rank)
_, _, pa_fro_exp, _ = partial_cross_approx(A_exp, max_rank)

_, _, ca_fro_poly, _ = cross_approx(A_poly, max_rank)
_, _, pa_fro_poly, _ = partial_cross_approx(A_poly, max_rank)

_, _, ca_fro_step, _ = cross_approx(A_step, max_rank)
_, _, pa_fro_step, _ = partial_cross_approx(A_step, max_rank)

#plot

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

cases = [
    ("exp", A_exp, ca_fro_exp, pa_fro_exp),
    ("poly", A_poly, ca_fro_poly, pa_fro_poly),
    ("step", A_step, ca_fro_step, pa_fro_step),
]

for ax, (name, A, ca, pa) in zip(axes, cases):

    max_r = min(max(len(ca), len(pa)), max_rank)
    svd_errors = frob_error_svd(A, max_r)

    r1 = np.arange(1, len(svd_errors) + 1)

    ax.plot(r1, svd_errors, label="svd")
    ax.plot(range(1, len(ca)+1), ca, label="full")
    ax.plot(range(1, len(pa)+1), pa, label="partial")

    ax.set_title(name)
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig('cross_approximation_comparison.png', dpi=150, bbox_inches='tight')
print("Plot saved as 'cross_approximation_comparison.png'")
