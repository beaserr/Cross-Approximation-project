import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from approximation import fpCA_approx
from approximation import func_ppca
from approximation import ppCA_adaptive
from approximation import ppCA_approx
from data import gaussian_kernel_matrix
from data import kernel_matrix
from data import load_iris_data
from error_analysis import fpCA_error_analysis
from error_analysis import ppCA_error_analysis
from error_analysis import svd_error
from error_analysis import time_approximation
from error_analysis import time_error_analysis
from synthetic import generate_test_matrices

# Iris data set test
X = load_iris_data()
K = gaussian_kernel_matrix(X, sigma=1.0)
K2 = kernel_matrix(X)
max_rank = 150

fca_U, fca_V, fca_time = time_approximation(fpCA_approx, K, max_rank)
ppca_U, ppca_V, ppca_time = time_approximation(ppCA_approx, K, max_rank)

fca_err, fca_error_time = time_error_analysis(fpCA_error_analysis, K, fca_U, fca_V)
ppca_err, ppca_error_time = time_error_analysis(ppCA_error_analysis, K, ppca_U, ppca_V)

ppca_adapt_err = ppCA_adaptive(K, max_rank)

fca_U2, fca_V2, fca_time2 = time_approximation(fpCA_approx, K2, max_rank)
ppca_U2, ppca_V2, ppca_time2 = time_approximation(ppCA_approx, K2, max_rank)
fca_err2, fca_error_time2 = time_error_analysis(fpCA_error_analysis, K2, fca_U2, fca_V2)
ppca_err2, ppca_error_time2 = time_error_analysis(ppCA_error_analysis, K2, ppca_U2, ppca_V2)

ppca_adapt_err2 = ppCA_adaptive(K2, max_rank)

print("\nIris Gaussian kernel")
print(f"fpCA approximation time: {fca_time:.4f} seconds")
print(f"fpCA error analysis time: {fca_error_time:.4f} seconds")
print(f"ppCA approximation time: {ppca_time:.4f} seconds")
print(f"ppCA error analysis time: {ppca_error_time:.4f} seconds")

print("\nIris linear kernel")
print(f"fpCA approximation time: {fca_time2:.4f} seconds")
print(f"fpCA error analysis time: {fca_error_time2:.4f} seconds")
print(f"ppCA approximation time: {ppca_time2:.4f} seconds")
print(f"ppCA error analysis time: {ppca_error_time2:.4f} seconds")

r = min(len(fca_err), max_rank)
svd_err = svd_error(K, r)
r2 = min(len(fca_err2), max_rank)
svd_err2 = svd_error(K2, r2)

plt.figure(figsize=(8, 5))
plt.semilogy(range(1, r + 1), svd_err, label="SVD")
plt.semilogy(range(1, len(fca_err) + 1), fca_err, label="fCA")
plt.semilogy(range(1, len(ppca_err) + 1), ppca_err, label="ppCA")
plt.semilogy(range(1, len(ppca_adapt_err) + 1), ppca_adapt_err, label="ppCA adaptive")
plt.title("Iris kernel matrix approximation with Gaussian kernel")
plt.xlabel("Rank")
plt.ylabel("Relative Frobenius error")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.semilogy(range(1, r2 + 1), svd_err2, label="SVD")
plt.semilogy(range(1, len(fca_err2) + 1), fca_err2, label="fCA")
plt.semilogy(range(1, len(ppca_err2) + 1), ppca_err2, label="ppCA")
plt.semilogy(range(1, len(ppca_adapt_err2) + 1), ppca_adapt_err2, label="ppCA adaptive")
plt.title("Iris kernel matrix approximation with linear kernel")
plt.xlabel("Rank")
plt.ylabel("Relative Frobenius error")
plt.legend()
plt.grid(True)
plt.show()


#Synthetic matrices test
n_big = 1000
max_rank_big = 150

A1, A2, A3 = generate_test_matrices(n=n_big, seed=0)
