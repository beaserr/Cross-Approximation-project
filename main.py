from cross_approx import cross_approx
from partial_cross_approx import  partial_cross_approx
from frob_error_svd import frob_error_svd
from test_matrices import test_matrices

matrices = [
    ("Step with noise", A_3),
    ("Polynomial decay", A_2),
    ("Exponential decay", A_1),
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

axes.legend()

plt.show()
