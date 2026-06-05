from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from approximation import fpCA_approx, func_ppca, natural_CA_diagonal_noise, ppCA_adaptive, ppCA_approx, ppCA_random_uniform, ppCA_random_weighted
from data import gaussian_kernel_matrix, kernel_matrix, load_iris_data
from error_analysis import fpCA_error_analysis, ppCA_error_analysis, svd_error, time_approximation, time_error_analysis
from synthetic import generate_test_matrices


output = Path("plots")


def save_plot(filename, output=output):
    # Save each figure as a PDF.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output / filename, format="pdf", bbox_inches="tight")
    plt.close()


def plot_errors(filename, title, curves, output=output, max_rank=None):
    # Standard error plot.
    plt.figure(figsize=(8, 5))

    for label, errors in curves:
        errors = np.asarray(errors, dtype=float)

        if len(errors) == 0:
            continue

        if max_rank is not None:
            errors = errors[:max_rank]

        errors = np.maximum(errors, np.finfo(float).tiny)
        plt.semilogy(range(1, len(errors) + 1), errors, label=label)

    plt.title(title)
    plt.xlabel("Rank")
    plt.ylabel("Relative Frobenius Error")
    plt.legend()
    plt.grid(True, which="both")
    save_plot(filename, output)


def plot_three(title, filename, svd_err, fca_err, ppca_err, output=output):
    max_rank = max(1, len(fca_err), len(ppca_err))
    plot_errors(filename, title, [("SVD", svd_err), ("fpCA", fca_err), ("ppCA", ppca_err)], output, max_rank)


def test_matrix(name, a, max_rank, filename, output=output):
    # Compare fpCA, ppCA, and SVD.
    print(name)

    fca_u, fca_v, fca_time = time_approximation(fpCA_approx, a, max_rank)
    ppca_u, ppca_v, ppca_time = time_approximation(ppCA_approx, a, max_rank)

    fca_err, fca_error_time = time_error_analysis(fpCA_error_analysis, a, fca_u, fca_v)
    ppca_err, ppca_error_time = time_error_analysis(ppCA_error_analysis, a, ppca_u, ppca_v)

    print(f"fpca time: {fca_time:.4f}")
    print(f"fpca error time: {fca_error_time:.4f}")
    print(f"ppca time: {ppca_time:.4f}")
    print(f"ppca error time: {ppca_error_time:.4f}")

    r = min(max_rank, a.shape[0], a.shape[1])
    svd_err = svd_error(a, r)

    plot_three(name, filename, svd_err, fca_err, ppca_err, output)


def test_adaptive(name, a, max_rank, filename, output=output):
    print(name)

    errors = ppCA_adaptive(a, max_rank, epsilon=1e-12)
    plot_errors(filename, name, [("Adaptive ppCA", errors)], output)


def test_pivots(name, a, max_rank, filename, output=output, seed=0):
    # Compare pivoting strategies.
    print(name)

    ppca_u, ppca_v, ppca_time = time_approximation(ppCA_approx, a, max_rank)

    uniform_u, uniform_v, uniform_time = time_approximation(lambda b, r, eps: ppCA_random_uniform(b, r, eps, seed=seed), a, max_rank)
    weighted_u, weighted_v, weighted_time = time_approximation(lambda b, r, eps: ppCA_random_weighted(b, r, eps, seed=seed, alpha=1.0), a, max_rank)
    natural_u, natural_v, natural_time = time_approximation(lambda b, r, eps: natural_CA_diagonal_noise(b, r, eps, diagonal_noise=1e-12), a, max_rank)

    ppca_err, _ = time_error_analysis(ppCA_error_analysis, a, ppca_u, ppca_v)
    uniform_err, _ = time_error_analysis(ppCA_error_analysis, a, uniform_u, uniform_v)
    weighted_err, _ = time_error_analysis(ppCA_error_analysis, a, weighted_u, weighted_v)
    natural_err, _ = time_error_analysis(ppCA_error_analysis, a, natural_u, natural_v)

    r = min(max_rank, a.shape[0], a.shape[1])
    svd_err = svd_error(a, r)

    print(f"ppca time: {ppca_time:.4f}")
    print(f"uniform time: {uniform_time:.4f}")
    print(f"weighted time: {weighted_time:.4f}")
    print(f"natural time: {natural_time:.4f}")

    plot_errors(filename, name, [("SVD", svd_err), ("ppCA", ppca_err), ("Uniform random", uniform_err), ("Weighted random", weighted_err), ("Natural", natural_err)], output, max_rank)


def randomized_error_stats(method, a, max_rank, runs=20):
    # Repeat randomized methods over several seeds.
    all_errors = []

    for seed in range(runs):
        u, v = method(a, max_rank, seed)
        errors = np.asarray(ppCA_error_analysis(a, u, v), dtype=float)

        errors = errors[np.isfinite(errors)]
        errors = np.maximum(errors, np.finfo(float).tiny)

        if len(errors) > 0:
            all_errors.append(errors)

    if len(all_errors) == 0:
        empty = np.array([])
        return empty, empty, empty

    min_len = min(len(errors) for errors in all_errors)
    all_errors = np.array([errors[:min_len] for errors in all_errors])

    mean = np.mean(all_errors, axis=0)
    q25 = np.percentile(all_errors, 25, axis=0)
    q75 = np.percentile(all_errors, 75, axis=0)

    return mean, q25, q75


def test_average(name, a, max_rank, filename, output=output, runs=20):
    print(name)

    uniform_mean, _, _ = randomized_error_stats(lambda b, r, seed: ppCA_random_uniform(b, r, seed=seed), a, max_rank, runs)
    weighted_mean, _, _ = randomized_error_stats(lambda b, r, seed: ppCA_random_weighted(b, r, seed=seed, alpha=1.0), a, max_rank, runs)

    ppca_u, ppca_v = ppCA_approx(a, max_rank)
    ppca_err = ppCA_error_analysis(a, ppca_u, ppca_v)

    r = min(max_rank, a.shape[0], a.shape[1])
    svd_err = svd_error(a, r)

    if len(uniform_mean) > 0:
        print(f"Uniform random pivoting mean error: {uniform_mean[-1]:.4e}")
    if len(weighted_mean) > 0:
        print(f"Weighted random pivoting mean error: {weighted_mean[-1]:.4e}")

    plot_errors(filename, name, [("SVD", svd_err), ("ppCA", ppca_err), ("Uniform mean", uniform_mean), ("Weighted mean", weighted_mean)], output, max_rank)


def test_error_bars(name, a, max_rank, filename, output=output, runs=20):
    # Show variability of randomized pivoting.
    print(name)

    uniform_mean, uniform_q25, uniform_q75 = randomized_error_stats(lambda b, r, seed: ppCA_random_uniform(b, r, seed=seed), a, max_rank, runs)
    weighted_mean, weighted_q25, weighted_q75 = randomized_error_stats(lambda b, r, seed: ppCA_random_weighted(b, r, seed=seed, alpha=1.0), a, max_rank, runs)

    plt.figure(figsize=(8, 5))

    for label, mean, q25, q75 in [("Uniform", uniform_mean, uniform_q25, uniform_q75), ("Weighted", weighted_mean, weighted_q25, weighted_q75)]:
        if len(mean) == 0:
            continue

        step = max(1, len(mean) // 8)
        indices = np.arange(0, len(mean), step)
        ranks = indices + 1

        y = np.maximum(mean[indices], np.finfo(float).tiny)
        y_low = np.maximum(q25[indices], np.finfo(float).tiny)
        y_high = np.maximum(q75[indices], y_low * (1.0 + 1e-12))

        lower = np.maximum(y - y_low, 0.0)
        upper = np.maximum(y_high - y, 0.0)

        plt.errorbar(ranks, y, yerr=[lower, upper], marker="o", capsize=3, label=label)

    plt.yscale("log")
    plt.title(name)
    plt.xlabel("Rank")
    plt.ylabel("Relative Frobenius Error")
    plt.legend(title="Method")
    plt.grid(True, which="both")
    save_plot(filename, output)


def test_mean_pivots(name, a, max_rank, filename, output=output, runs=20):
    print(name)

    uniform_mean, _, _ = randomized_error_stats(lambda b, r, seed: ppCA_random_uniform(b, r, seed=seed), a, max_rank, runs)
    weighted_mean, _, _ = randomized_error_stats(lambda b, r, seed: ppCA_random_weighted(b, r, seed=seed, alpha=1.0), a, max_rank, runs)

    natural_u, natural_v = natural_CA_diagonal_noise(a, max_rank, epsilon=1e-12, diagonal_noise=1e-12)
    natural_err = ppCA_error_analysis(a, natural_u, natural_v)

    plot_errors(filename, name, [("Uniform mean", uniform_mean), ("Weighted mean", weighted_mean), ("Natural", natural_err)], output, max_rank)


def iris(output=output):
    # Real data test.
    from sklearn.preprocessing import StandardScaler

    x = StandardScaler().fit_transform(load_iris_data().astype(float))
    sigma = np.sqrt(x.shape[1])
    max_rank = 150

    k1 = gaussian_kernel_matrix(x, sigma=sigma)
    k2 = kernel_matrix(x)

    test_matrix("Iris Gaussian", k1, max_rank, "iris_gaussian.pdf", output)
    test_adaptive("Iris Gaussian Adaptive", k1, max_rank, "iris_gaussian_adaptive.pdf", output)

    test_matrix("Iris Linear", k2, max_rank, "iris_linear.pdf", output)
    test_adaptive("Iris Linear Adaptive", k2, max_rank, "iris_linear_adaptive.pdf", output)


def synthetic(output=output):
    # Artificial matrix tests.
    n = 1000
    max_rank = 250

    a1, a2, _ = generate_test_matrices(n=n, seed=0)

    test_matrix("Synthetic A1", a1, max_rank, "synthetic_a1.pdf", output)
    test_adaptive("Synthetic A1 Adaptive", a1, max_rank, "synthetic_a1_adaptive.pdf", output)

    test_matrix("Synthetic A2", a2, max_rank, "synthetic_a2.pdf", output)
    test_adaptive("Synthetic A2 Adaptive", a2, max_rank, "synthetic_a2_adaptive.pdf", output)


def pivots(output=output):
    n = 300
    max_rank = 80

    a1, a2, _ = generate_test_matrices(n=n, seed=1)

    test_pivots("Pivots A1", a1, max_rank, "pivots_a1.pdf", output)
    test_pivots("Pivots A2", a2, max_rank, "pivots_a2.pdf", output)

    test_average("Average A2", a2, max_rank, "average_a2.pdf", output)
    test_error_bars("Random Pivot Error Bars A2", a2, max_rank, "random_pivot_error_bars_a2.pdf", output)
    test_mean_pivots("Mean Pivots A2", a2, max_rank, "mean_pivots.pdf", output)


def california(output=output):
    # Larger real data test.
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler

    n = 1000
    max_rank = 250

    data = fetch_california_housing()
    x = StandardScaler().fit_transform(data.data[:n].astype(float))
    sigma = np.sqrt(x.shape[1])
    k = gaussian_kernel_matrix(x, sigma=sigma)

    test_matrix("California Scaled Gaussian", k, max_rank, "california.pdf", output)
    test_adaptive("California Scaled Gaussian Adaptive", k, max_rank, "california_adaptive.pdf", output)


def matrix_free(output=output):
    x = load_iris_data()
    sigma = 1.0

    def k(i, j):
        diff = x[i] - x[j]
        return np.exp(-np.dot(diff, diff) / (2.0 * sigma**2))

    errors = func_ppca(k, m=x.shape[0], n=x.shape[0], max_rank=10, epsilon=1e-12)

    plt.figure(figsize=(8, 5))
    plt.semilogy(range(1, len(errors) + 1), errors, marker="o", label="ppCA")
    plt.xlabel("Rank")
    plt.ylabel("Relative Frobenius Error")
    plt.title("Matrix Free Iris")
    plt.legend()
    plt.grid(True, which="both")
    save_plot("matrix_free_iris.pdf", output)


def forest(output=output):
    from sklearn.datasets import fetch_covtype
    from sklearn.preprocessing import StandardScaler

    n = 500
    max_rank = 20

    data = fetch_covtype()
    x = StandardScaler().fit_transform(data.data[:n].astype(float))
    sigma = np.sqrt(x.shape[1])

    def k(i, j):
        diff = x[i] - x[j]
        return np.exp(-np.dot(diff, diff) / (2.0 * sigma**2))

    errors = func_ppca(k, m=n, n=n, max_rank=max_rank, epsilon=1e-12)

    plt.figure(figsize=(8, 5))
    plt.semilogy(range(1, len(errors) + 1), errors, marker="o", label="ppCA")
    plt.xlabel("Rank")
    plt.ylabel("Relative Frobenius Error")
    plt.title("Forest")
    plt.legend()
    plt.grid(True, which="both")
    save_plot("forest.pdf", output)


def main(output=output):
    # Run all experiments.
    iris(output)
    synthetic(output)
    california(output)
    pivots(output)
    matrix_free(output)
    forest(output)


if __name__ == "__main__":
    main()
