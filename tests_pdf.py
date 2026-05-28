import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing

from approximation import fpCA_approx, func_ppca, ppCA_adaptive, ppCA_approx
from data import gaussian_kernel_matrix, kernel_matrix, load_iris_data
from error_analysis import (
    fpCA_error_analysis,
    ppCA_error_analysis,
    svd_error,
    time_approximation,
    time_error_analysis,
)
from synthetic import generate_test_matrices


PLOT_DIR = Path("plots")


def save_current_figure(filename, output_dir=PLOT_DIR):
    """Save the current Matplotlib figure as a PDF and close it."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    plt.tight_layout()
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()
    return path


def plot_error_curves(title, filename, svd_err, fca_err, ppca_err, ppca_adapt_err, output_dir=PLOT_DIR):
    """Plot SVD, fpCA, ppCA, and adaptive ppCA relative Frobenius errors."""
    plt.figure(figsize=(8, 5))
    plt.semilogy(range(1, len(svd_err) + 1), svd_err, label="SVD")
    plt.semilogy(range(1, len(fca_err) + 1), fca_err, label="fpCA")
    plt.semilogy(range(1, len(ppca_err) + 1), ppca_err, label="ppCA")
    plt.semilogy(range(1, len(ppca_adapt_err) + 1), ppca_adapt_err, label="ppCA adaptive")
    plt.title(title)
    plt.xlabel("Rank")
    plt.ylabel("Relative Frobenius error")
    plt.legend()
    plt.grid(True)
    return save_current_figure(filename, output_dir)


def run_matrix_experiment(name, A, max_rank, output_filename, output_dir=PLOT_DIR):
    """Run fpCA, ppCA, adaptive ppCA, and SVD comparison for one matrix."""
    print(f"\n{name}, size {A.shape}")

    fca_U, fca_V, fca_time = time_approximation(fpCA_approx, A, max_rank)
    ppca_U, ppca_V, ppca_time = time_approximation(ppCA_approx, A, max_rank)

    fca_err, fca_error_time = time_error_analysis(fpCA_error_analysis, A, fca_U, fca_V)
    ppca_err, ppca_error_time = time_error_analysis(ppCA_error_analysis, A, ppca_U, ppca_V)
    ppca_adapt_err = ppCA_adaptive(A, max_rank, epsilon=1e-12)

    print(f"fpCA approximation time: {fca_time:.4f} seconds")
    print(f"fpCA error analysis time: {fca_error_time:.4f} seconds")
    print(f"ppCA approximation time: {ppca_time:.4f} seconds")
    print(f"ppCA error analysis time: {ppca_error_time:.4f} seconds")

    r = min(len(fca_err), max_rank)
    svd_err_values = svd_error(A, r)

    path = plot_error_curves(
        title=name,
        filename=output_filename,
        svd_err=svd_err_values,
        fca_err=fca_err,
        ppca_err=ppca_err,
        ppca_adapt_err=ppca_adapt_err,
        output_dir=output_dir,
    )
    print(f"Saved plot to {path}")


def run_iris_experiments(output_dir=PLOT_DIR):
    X = load_iris_data()
    max_rank = 150

    K_gaussian = gaussian_kernel_matrix(X, sigma=1.0)
    K_linear = kernel_matrix(X)

    run_matrix_experiment(
        "Iris kernel matrix approximation with Gaussian kernel",
        K_gaussian,
        max_rank,
        "iris_gaussian_kernel.pdf",
        output_dir,
    )
    run_matrix_experiment(
        "Iris kernel matrix approximation with linear kernel",
        K_linear,
        max_rank,
        "iris_linear_kernel.pdf",
        output_dir,
    )


def run_synthetic_experiments(output_dir=PLOT_DIR):
    n_big = 1000
    max_rank_big = 150
    A1, A2, A3 = generate_test_matrices(n=n_big, seed=0)

    for i, A in enumerate([A1, A2, A3], start=1):
        run_matrix_experiment(
            f"Synthetic test matrix A{i}, n={n_big}",
            A,
            max_rank_big,
            f"synthetic_A{i}.pdf",
            output_dir,
        )


def run_california_experiment(output_dir=PLOT_DIR):
    max_rank_big = 150
    n_data = 1000

    data = fetch_california_housing()
    X_big = data.data[:n_data]
    K_big = gaussian_kernel_matrix(X_big, sigma=1.0)

    run_matrix_experiment(
        f"California housing Gaussian kernel matrix, n={n_data}",
        K_big,
        max_rank_big,
        "california_gaussian_kernel.pdf",
        output_dir,
    )


def run_matrix_free_experiment(output_dir=PLOT_DIR):
    np.random.seed(0)
    X = load_iris_data()
    sigma = 1.0

    def K_func(i, j):
        diff = X[i] - X[j]
        return np.exp(-np.dot(diff, diff) / (2.0 * sigma**2))

    errors = func_ppca(K_func, m=X.shape[0], n=X.shape[0], max_rank=10, epsilon=1e-12)

    plt.figure(figsize=(8, 5))
    plt.semilogy(range(1, len(errors) + 1), errors, marker="o", label="matrix-free ppCA")
    plt.xlabel("Iteration")
    plt.ylabel("Relative Frobenius error")
    plt.title("Matrix-free Iris Gaussian kernel approximation")
    plt.legend()
    plt.grid(True)
    path = save_current_figure("matrix_free_iris_gaussian_kernel.pdf", output_dir)
    print(f"\nMatrix-free Iris Gaussian kernel")
    print(f"Saved plot to {path}")


def main():
    parser = argparse.ArgumentParser(description="Run cross approximation numerical experiments and save plots as PDF files.")
    parser.add_argument("--output-dir", default="plots", help="Directory where PDF plots are saved.")
    parser.add_argument(
        "--skip-california",
        action="store_true",
        help="Skip the California housing experiment if the dataset is unavailable or runtime is a concern.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    run_iris_experiments(output_dir)
    run_synthetic_experiments(output_dir)

    if not args.skip_california:
        try:
            run_california_experiment(output_dir)
        except Exception as exc:
            print(f"\nSkipped California housing experiment because the dataset could not be loaded: {exc}")

    run_matrix_free_experiment(output_dir)


if __name__ == "__main__":
    main()
