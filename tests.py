import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from approximation import (
    fpCA_approx,
    func_ppca,
    natural_CA_diagonal_noise,
    ppCA_adaptive,
    ppCA_approx,
    ppCA_random_uniform,
    ppCA_random_weighted,
)
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


def save_error_plot(filename, title, curves, output_dir=PLOT_DIR, max_display_rank=None):
    """Save a semilog plot containing any number of error curves."""
    plt.figure(figsize=(8, 5))

    for label, errors in curves:
        errors = np.asarray(errors)

        if len(errors) == 0:
            print(f"Warning: empty error curve for {label}; skipping it in {filename}")
            continue

        if max_display_rank is not None:
            errors = errors[:max_display_rank]

        plt.semilogy(range(1, len(errors) + 1), errors, label=label)

    plt.title(title)
    plt.xlabel("Rank")
    plt.ylabel("Relative Frobenius error")
    plt.legend()
    plt.grid(True)
    return save_current_figure(filename, output_dir)


# tests fucntions

def plot_standard_error_curves(title, filename, svd_err, fca_err, ppca_err, output_dir=PLOT_DIR):
    max_display_rank = max(1, len(fca_err), len(ppca_err))

    return save_error_plot(
        filename,
        title,
        [
            ("SVD", svd_err),
            ("fpCA", fca_err),
            ("ppCA", ppca_err),
        ],
        output_dir,
        max_display_rank=max_display_rank,
    )


def run_matrix_experiment(name, A, max_rank, output_filename, output_dir=PLOT_DIR):
    """Run fpCA, ppCA, and SVD comparison for one matrix."""
    print(f"\n{name}, size {A.shape}")

    fca_U, fca_V, fca_time = time_approximation(fpCA_approx, A, max_rank)
    ppca_U, ppca_V, ppca_time = time_approximation(ppCA_approx, A, max_rank)

    fca_err, fca_error_time = time_error_analysis(fpCA_error_analysis, A, fca_U, fca_V)
    ppca_err, ppca_error_time = time_error_analysis(ppCA_error_analysis, A, ppca_U, ppca_V)
    print(f"fpCA approximation time: {fca_time:.4f} seconds")
    print(f"fpCA error analysis time: {fca_error_time:.4f} seconds")
    print(f"ppCA approximation time: {ppca_time:.4f} seconds")
    print(f"ppCA error analysis time: {ppca_error_time:.4f} seconds")

    r = min(max_rank, A.shape[0], A.shape[1])
    svd_err_values = svd_error(A, r)

    path = plot_standard_error_curves(
        title=name,
        filename=output_filename,
        svd_err=svd_err_values,
        fca_err=fca_err,
        ppca_err=ppca_err,
        output_dir=output_dir,
    )
    print(f"Saved plot to {path}")


def run_adaptive_experiment(name, A, max_rank, output_filename, output_dir=PLOT_DIR):
    """Run only the adaptive ppCA curve."""
    print(f"\n{name}: adaptive ppCA, size {A.shape}")

    ppca_adapt_err = ppCA_adaptive(A, max_rank, epsilon=1e-12)
    path = save_error_plot(
        output_filename,
        f"Adaptive ppCA: {name}",
        [("ppCA adaptive", ppca_adapt_err)],
        output_dir,
    )
    print(f"Saved plot to {path}")


#Other pivoting methods

def run_random_and_natural_experiment(
    name,
    A,
    max_rank,
    output_filename,
    output_dir=PLOT_DIR,
    seed=0,
    max_display_rank=40,
):
    
    print(f"\n{name}: random and natural pivoting, size {A.shape}")

    ppca_U, ppca_V, ppca_time = time_approximation(ppCA_approx, A, max_rank)

    uniform_U, uniform_V, uniform_time = time_approximation(
        lambda B, r, eps: ppCA_random_uniform(B, r, eps, seed=seed),
        A,
        max_rank,
    )

    weighted_U, weighted_V, weighted_time = time_approximation(
        lambda B, r, eps: ppCA_random_weighted(B, r, eps, seed=seed, alpha=1.0),
        A,
        max_rank,
    )

    weighted2_U, weighted2_V, weighted2_time = time_approximation(
        lambda B, r, eps: ppCA_random_weighted(B, r, eps, seed=seed, alpha=2.0),
        A,
        max_rank,
    )

    natural_U, natural_V, natural_time = time_approximation(
        lambda B, r, eps: natural_CA_diagonal_noise(B, r, eps, diagonal_noise=1e-12),
        A,
        max_rank,
    )

    ppca_err, _ = time_error_analysis(ppCA_error_analysis, A, ppca_U, ppca_V)
    uniform_err, _ = time_error_analysis(ppCA_error_analysis, A, uniform_U, uniform_V)
    weighted_err, _ = time_error_analysis(ppCA_error_analysis, A, weighted_U, weighted_V)
    weighted2_err, _ = time_error_analysis(ppCA_error_analysis, A, weighted2_U, weighted2_V)
    natural_err, _ = time_error_analysis(ppCA_error_analysis, A, natural_U, natural_V)

    r = min(max_rank, A.shape[0], A.shape[1])
    svd_err_values = svd_error(A, r)

    print(f"ppCA approximation time: {ppca_time:.4f} seconds")
    print(f"random uniform ppCA time: {uniform_time:.4f} seconds")
    print(f"random weighted ppCA, alpha=1, time: {weighted_time:.4f} seconds")
    print(f"random weighted ppCA, alpha=2, time: {weighted2_time:.4f} seconds")
    print(f"natural pivoting with diagonal noise time: {natural_time:.4f} seconds")

    path = save_error_plot(
        output_filename,
        f"Random and natural pivoting: {name}",
        [
            ("SVD", svd_err_values),
            ("ppCA", ppca_err),
            ("random uniform ppCA", uniform_err),
            ("random weighted ppCA alpha=1", weighted_err),
            ("random weighted ppCA alpha=2", weighted2_err),
            ("natural pivots + diagonal noise", natural_err),
        ],
        output_dir,
        max_display_rank=max_display_rank,
    )
    print(f"Saved plot to {path}")
    return [uniform_err, weighted_err, weighted2_err, natural_err]


def average_random_errors(method, A, max_rank, n_runs=10):
    all_errors = []

    for seed in range(n_runs):
        U, V = method(A, max_rank, seed)
        errors = ppCA_error_analysis(A, U, V)

        if len(errors) > 0:
            all_errors.append(errors)

    if len(all_errors) == 0:
        return np.array([]), np.array([]), np.array([])

    min_len = min(len(errors) for errors in all_errors)
    all_errors = np.array([errors[:min_len] for errors in all_errors])

    mean_errors = np.mean(all_errors, axis=0)
    min_errors = np.min(all_errors, axis=0)
    max_errors = np.max(all_errors, axis=0)

    return mean_errors, min_errors, max_errors


def run_averaged_random_experiment(name, A, max_rank, output_filename, output_dir=PLOT_DIR, n_runs=20):
    """Average the random methods over several seeds to see typical behavior."""
    print(f"\n{name}: averaged random pivoting over {n_runs} runs")

    uniform_mean, _, _ = average_random_errors(
        lambda B, r, seed: ppCA_random_uniform(B, r, seed=seed),
        A,
        max_rank,
        n_runs=n_runs,
    )

    weighted_mean, _, _ = average_random_errors(
        lambda B, r, seed: ppCA_random_weighted(B, r, seed=seed, alpha=1.0),
        A,
        max_rank,
        n_runs=n_runs,
    )

    weighted2_mean, _, _ = average_random_errors(
        lambda B, r, seed: ppCA_random_weighted(B, r, seed=seed, alpha=2.0),
        A,
        max_rank,
        n_runs=n_runs,
    )

    ppca_U, ppca_V = ppCA_approx(A, max_rank)
    ppca_err = ppCA_error_analysis(A, ppca_U, ppca_V)

    r = min(max_rank, A.shape[0], A.shape[1])
    svd_err_values = svd_error(A, r)

    path = save_error_plot(
        output_filename,
        f"Average randomized ppCA behavior: {name}",
        [
            ("SVD", svd_err_values),
            ("ppCA", ppca_err),
            ("uniform random mean", uniform_mean),
            ("weighted random alpha=1 mean", weighted_mean),
            ("weighted random alpha=2 mean", weighted2_mean),
        ],
        output_dir,
        max_display_rank=min(40, max_rank),
    )
    print(f"Saved plot to {path}")


def save_mean_of_curves_plot(name, curve_groups, output_filename, output_dir=PLOT_DIR, max_display_rank=40):
    """Save one average curve over all previously computed non-standard pivot curves."""
    curves = []

    for group in curve_groups:
        for errors in group:
            errors = np.asarray(errors)

            if len(errors) > 0:
                curves.append(errors)

    if len(curves) == 0:
        print(f"Warning: no curves available for the mean plot {output_filename}")
        return None

    min_len = min(len(errors) for errors in curves)
    mean_curve = np.mean(np.array([errors[:min_len] for errors in curves]), axis=0)

    path = save_error_plot(
        output_filename,
        f"Mean curve over other pivoting methods: {name}",
        [("mean over other pivoting methods", mean_curve)],
        output_dir,
        max_display_rank=max_display_rank,
    )
    print(f"Saved plot to {path}")
    return path


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
    run_adaptive_experiment(
        "Iris kernel matrix approximation with Gaussian kernel",
        K_gaussian,
        max_rank,
        "adaptive_iris_gaussian_kernel.pdf",
        output_dir,
    )

    run_matrix_experiment(
        "Iris kernel matrix approximation with linear kernel",
        K_linear,
        max_rank,
        "iris_linear_kernel.pdf",
        output_dir,
    )
    run_adaptive_experiment(
        "Iris kernel matrix approximation with linear kernel",
        K_linear,
        max_rank,
        "adaptive_iris_linear_kernel.pdf",
        output_dir,
    )


def run_synthetic_experiments(output_dir=PLOT_DIR):
    n_big = 1000
    max_rank_big = 150
    A1, A2, _ = generate_test_matrices(n=n_big, seed=0)

    for i, A in enumerate([A1, A2], start=1):
        run_matrix_experiment(
            f"Synthetic test matrix A{i}, n={n_big}",
            A,
            max_rank_big,
            f"synthetic_A{i}.pdf",
            output_dir,
        )
        run_adaptive_experiment(
            f"Synthetic test matrix A{i}, n={n_big}",
            A,
            max_rank_big,
            f"adaptive_synthetic_A{i}.pdf",
            output_dir,
        )


def run_random_and_natural_synthetic_experiments(output_dir=PLOT_DIR):

    n_exp = 300
    max_rank_exp = 80
    A1, A2, _ = generate_test_matrices(n=n_exp, seed=1)

    matrices = [
        ("Synthetic A1 exponential decay", A1, "A1"),
        ("Synthetic A2 polynomial decay", A2, "A2"),
    ]

    all_other_pivot_curves = []

    for name, A, short_name in matrices:
        curves = run_random_and_natural_experiment(
            name,
            A,
            max_rank_exp,
            f"random_natural_synthetic_{short_name}.pdf",
            output_dir,
        )
        all_other_pivot_curves.append(curves)

    # A2 is a good case for averaging because polynomial decay is harder.
    run_averaged_random_experiment(
        "Synthetic A2 polynomial decay",
        A2,
        max_rank_exp,
        "random_pivoting_average_synthetic_A2.pdf",
        output_dir,
        n_runs=20,
    )

    save_mean_of_curves_plot(
        "Synthetic A1 and A2",
        all_other_pivot_curves,
        "other_pivoting_mean_synthetic_A1_A2.pdf",
        output_dir,
    )


def run_california_experiment(output_dir=PLOT_DIR):
    # Import here so --skip-california can still work if this dataset is unavailable.
    from sklearn.datasets import fetch_california_housing

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
    run_adaptive_experiment(
        f"California housing Gaussian kernel matrix, n={n_data}",
        K_big,
        max_rank_big,
        "adaptive_california_gaussian_kernel.pdf",
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
    print("\nMatrix-free Iris Gaussian kernel")
    print(f"Saved plot to {path}")


def run_forest_covertypes_kernel_function_experiment(output_dir=PLOT_DIR):
    # Imports stay local so the other experiments can run even if this dataset is unavailable.
    from sklearn.datasets import fetch_covtype
    from sklearn.preprocessing import StandardScaler

    n_data = 500
    max_rank = 20

    data = fetch_covtype()
    X = data.data[:n_data].astype(float)
    X = StandardScaler().fit_transform(X)
    sigma = np.sqrt(X.shape[1])

    def K_func(i, j):
        diff = X[i] - X[j]
        return np.exp(-np.dot(diff, diff) / (2.0 * sigma**2))

    errors = func_ppca(K_func, m=n_data, n=n_data, max_rank=max_rank, epsilon=1e-12)

    plt.figure(figsize=(8, 5))
    plt.semilogy(range(1, len(errors) + 1), errors, marker="o", label="matrix-free ppCA")
    plt.xlabel("Iteration")
    plt.ylabel("Relative Frobenius error")
    plt.title(f"Matrix-free Forest Covertypes Gaussian kernel, n={n_data}")
    plt.legend()
    plt.grid(True)
    path = save_current_figure("forest_covertypes_kernel_function.pdf", output_dir)
    print(f"\nMatrix-free Forest Covertypes Gaussian kernel, n={n_data}")
    print(f"Saved plot to {path}")


# ============================================================
# Command line interface
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run cross approximation numerical experiments and save plots as PDF files."
    )
    parser.add_argument("--output-dir", default="plots", help="Directory where PDF plots are saved.")
    parser.add_argument(
        "--skip-california",
        action="store_true",
        help="Skip the California housing experiment if the dataset is unavailable or runtime is a concern.",
    )
    parser.add_argument(
        "--skip-original",
        action="store_true",
        help="Skip the original Iris and n=1000 synthetic experiments.",
    )
    parser.add_argument(
        "--skip-new",
        action="store_true",
        help="Skip the random-pivot and natural-pivot experiments.",
    )
    parser.add_argument(
        "--skip-forest-covertypes",
        action="store_true",
        help="Skip the Forest Covertypes matrix-free kernel function experiment.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_original:
        run_iris_experiments(output_dir)
        run_synthetic_experiments(output_dir)

    if not args.skip_california:
        try:
            run_california_experiment(output_dir)
        except Exception as exc:
            print(f"\nSkipped California housing experiment because the dataset could not be loaded: {exc}")

    if not args.skip_new:
        run_random_and_natural_synthetic_experiments(output_dir)

    run_matrix_free_experiment(output_dir)

    if not args.skip_forest_covertypes:
        try:
            run_forest_covertypes_kernel_function_experiment(output_dir)
        except Exception as exc:
            print(f"\nSkipped Forest Covertypes experiment because the dataset could not be loaded: {exc}")


if __name__ == "__main__":
    main()
