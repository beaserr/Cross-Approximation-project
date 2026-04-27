import numpy as np
import matplotlib.pyplot as plt

from ca_methods import fpCA, ppCA, frob_error_svd
from subjects import load_iris_data, kernel_matrix


def run_iris_test(max_rank=40):
    X, y, data = load_iris_data()

    K = linear_kernel_matrix(X)
    ca_full, _, _ = fpCA(K, max_rank)
    ca_partial, _, _ = ppCA(K, max_rank, start_row=0)
    r = min(len(ca_full), len(ca_partial), max_rank)
    svd_err = frob_error_svd(K, r)

    ranks = np.arange(1, r + 1)

    plt.figure(figsize=(8, 5))
    plt.semilogy(ranks, svd_err, label="SVD")
    plt.semilogy(np.arange(1, len(ca_full) + 1), ca_full, label="FCA")
    plt.semilogy(np.arange(1, len(ca_partial) + 1), ca_partial, label="ppCA")
    plt.title("Iris kernel matrix approximation")
    plt.xlabel("Rank")
    plt.ylabel("Relative Frobenius error")
    plt.grid(True)
    plt.legend()
    plt.show()
