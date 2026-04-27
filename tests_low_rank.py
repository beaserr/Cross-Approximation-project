import numpy as np
import matplotlib.pyplot as plt

from ca_methods import fpCA, ppCA, frob_error_svd
from subjects import test_matrices


def run_low_rank_test(max_rank=40):
    A_1, A_2, A_3 = test_matrices(n=150, seed=0)
    matrices = {
        "Exponential decay": A_1,
        "Polynomial decay": A_2,
        "Low-rank + noise": A_3
    }

    for name, A in matrices.items():
        ca_full, _, _ = fpCA(A, max_rank)
        ca_partial, _, _ = ppCA(A, max_rank, start_row=0)
        r = min(len(ca_full), len(ca_partial), max_rank)
        svd_err = frob_error_svd(A, r)
        plt.figure(figsize=(8, 5))

        plt.semilogy(np.arange(1, r + 1), svd_err, label="SVD")
        plt.semilogy(np.arange(1, len(ca_full) + 1), ca_full, label="FCA")
        plt.semilogy(np.arange(1, len(ca_partial) + 1), ca_partial, label="ppCA")
        plt.title(f"Low-rank approximation: {name}")
        plt.xlabel("Rank")
        plt.ylabel("Relative Frobenius error")
        plt.grid(True)
        plt.legend()

        plt.show()
