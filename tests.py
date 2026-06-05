from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from approximation import fpCA_approx, func_ppca, natural_CA_diagonal_noise, ppCA_adaptive, ppCA_approx, ppCA_random_uniform, ppCA_random_weighted
from data import gaussian_kernel_matrix, kernel_matrix, load_iris_data
from error_analysis import fpCA_error_analysis, ppCA_error_analysis, svd_error, time_approximation, time_error_analysis
from synthetic import generate_test_matrices


output = Path("plots")


def save_plot(filename, output=output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output / filename, format="pdf", bbox_inches="tight")
    plt.close()


def plot_errors(filename, title, curves, output=output, max_rank=None):
    plt.figure(figsize=(8, 5))

    for label, errors in curves:
        errors = np.asarray(errors)

        if len(errors) == 0:
            continue

        if max_rank is not None:
            errors = errors[:max_rank]

        plt.semilogy(range(1, len(errors) + 1), errors, label=label)

    plt.title(title)
    plt.xlabel("rank")
    plt.ylabel("error")
    plt.legend()
    plt.grid(True)
    save_plot(filename, output)


def plot_three(title, filename, svd_err, fca_err, ppca_err, output=output):
    max_rank = max(1, len(fca_err), len(ppca_err))
    plot_errors(filename, title, [("svd", svd_err), ("fpca", fca_err), ("ppca", ppca_err)], output, max_rank)


def test_matrix(name, a, max_rank, filename, output=output):
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
    plot_errors(filename, name, [("adaptive ppca", errors)], output)


def test_pivots(name, a, max_rank, filename, output=output, seed=0):
    print(name)

    ppca_u, ppca_v, ppca_time = time_approximation(ppCA_approx, a, max_rank)

    uniform_u, uniform_v, uniform_time = time_approximation(lambda b, r, eps: ppCA_random_uniform(b, r, eps, seed=seed), a, max_rank)
    weighted_u, weighted_v, weighted_time = time_approximation(lambda b, r, eps: ppCA_random_weighted(b, r, eps, seed=seed, alpha=1.0), a, max_rank)
    weighted2_u, weighted2_v, weighted2_time = time_approximation(lambda b, r, eps: ppCA_random_weighted(b, r, eps, seed=seed, alpha=2.0), a, max_rank)
    natural_u, natural_v, natural_time = time_approximation(lambda b, r, eps: natural_CA_diagonal_noise(b, r, eps, diagonal_noise=1e-12), a, max_rank)

    ppca_err, _ = time_error_analysis(ppCA_error_analysis, a, ppca_u, ppca_v)
    uniform_err, _ = time_error_analysis(ppCA_error_analysis, a, uniform_u, uniform_v)
    weighted_err, _ = time_error_analysis(ppCA_error_analysis, a, weighted_u, weighted_v)
    weighted2_err, _ = time_error_analysis(ppCA_error_analysis, a, weighted2_u, weighted2_v)
    natural_err, _ = time_error_analysis(ppCA_error_analysis, a, natural_u, natural_v)

    r = min(max_rank, a.shape[0], a.shape[1])
    svd_err = svd_error(a, r)

    print(f"ppca time: {ppca_time:.4f}")
    print(f"uniform time: {uniform_time:.4f}")
    print(f"weighted time: {weighted_time:.4f}")
    print(f"weighted 2 time: {weighted2_time:.4f}")
    print(f"natural time: {natural_time:.4f}")

    plot_errors(
        filename,
        name,
        [
            ("svd", svd_err),
            ("ppca", ppca_err),
            ("uniform", uniform_err),
            ("weighted", weighted_err),
            ("weighted 2", weighted2_err),
            ("natural", natural_err),
        ],
        output,
        40,
    )

    return [uniform_err, weighted_err, weighted2_err, natural_err]


def average_errors(method, a, max_rank, runs=10):
    all_errors = []

    for seed in range(runs):
        u, v = method(a, max_rank, seed)
        errors = ppCA_error_analysis(a, u, v)

        if len(errors) > 0:
            all_errors.append(errors)

    if len(all_errors) == 0:
        return np.array([])

    min_len = min(len(errors) for errors in all_errors)
    all_errors = np.array([errors[:min_len] for errors in all_errors])

    return np.mean(all_errors, axis=0)


def test_average(name, a, max_rank, filename, output=output, runs=20):
    print(name)

    uniform = average_errors(lambda b, r, seed: ppCA_random_uniform(b, r, seed=seed), a, max_rank, runs)
    weighted = average_errors(lambda b, r, seed: ppCA_random_weighted(b, r, seed=seed, alpha=1.0), a, max_rank, runs)
    weighted2 = average_errors(lambda b, r, seed: ppCA_random_weighted(b, r, seed=seed, alpha=2.0), a, max_rank, runs)

    ppca_u, ppca_v = ppCA_approx(a, max_rank)
    ppca_err = ppCA_error_analysis(a, ppca_u, ppca_v)

    r = min(max_rank, a.shape[0], a.shape[1])
    svd_err = svd_error(a, r)

    plot_errors(filename, name, [("svd", svd_err), ("ppca", ppca_err), ("uniform", uniform), ("weighted", weighted), ("weighted 2", weighted2)], output, 40)


def test_mean(name, groups, filename, output=output):
    curves = []

    for group in groups:
        for errors in group:
            errors = np.asarray(errors)

            if len(errors) > 0:
                curves.append(errors)

    if len(curves) == 0:
        return

    min_len = min(len(errors) for errors in curves)
    mean = np.mean(np.array([errors[:min_len] for errors in curves]), axis=0)

    plot_errors(filename, name, [("mean", mean)], output, 40)


def iris(output=output):
    x = load_iris_data()
    max_rank = 150

    k1 = gaussian_kernel_matrix(x, sigma=1.0)
    k2 = kernel_matrix(x)

    test_matrix("iris gaussian", k1, max_rank, "iris_gaussian.pdf", output)
    test_adaptive("iris gaussian adaptive", k1, max_rank, "iris_gaussian_adaptive.pdf", output)

    test_matrix("iris linear", k2, max_rank, "iris_linear.pdf", output)
    test_adaptive("iris linear adaptive", k2, max_rank, "iris_linear_adaptive.pdf", output)


def synthetic(output=output):
    n = 1000
    max_rank = 150

    a1, a2, _ = generate_test_matrices(n=n, seed=0)

    test_matrix("synthetic a1", a1, max_rank, "synthetic_a1.pdf", output)
    test_adaptive("synthetic a1 adaptive", a1, max_rank, "synthetic_a1_adaptive.pdf", output)

    test_matrix("synthetic a2", a2, max_rank, "synthetic_a2.pdf", output)
    test_adaptive("synthetic a2 adaptive", a2, max_rank, "synthetic_a2_adaptive.pdf", output)


def pivots(output=output):
    n = 300
    max_rank = 80

    a1, a2, _ = generate_test_matrices(n=n, seed=1)

    curves = []

    curves.append(test_pivots("pivots a1", a1, max_rank, "pivots_a1.pdf", output))
    curves.append(test_pivots("pivots a2", a2, max_rank, "pivots_a2.pdf", output))

    test_average("average a2", a2, max_rank, "average_a2.pdf", output)
    test_mean("mean pivots", curves, "mean_pivots.pdf", output)


def california(output=output):
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler

    n = 1000
    max_rank = 250

    data = fetch_california_housing()
    x = StandardScaler().fit_transform(data.data[:n].astype(float))
    sigma = np.sqrt(x.shape[1])
    k = gaussian_kernel_matrix(x, sigma=sigma)

    test_matrix("california scaled gaussian", k, max_rank, "california.pdf", output)
    test_adaptive("california scaled gaussian adaptive", k, max_rank, "california_adaptive.pdf", output)


def matrix_free(output=output):
    x = load_iris_data()
    sigma = 1.0

    def k(i, j):
        diff = x[i] - x[j]
        return np.exp(-np.dot(diff, diff) / (2.0 * sigma**2))

    errors = func_ppca(k, m=x.shape[0], n=x.shape[0], max_rank=10, epsilon=1e-12)

    plt.figure(figsize=(8, 5))
    plt.semilogy(range(1, len(errors) + 1), errors, marker="o", label="ppca")
    plt.xlabel("rank")
    plt.ylabel("error")
    plt.title("matrix free iris")
    plt.legend()
    plt.grid(True)
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
    plt.semilogy(range(1, len(errors) + 1), errors, marker="o", label="ppca")
    plt.xlabel("rank")
    plt.ylabel("error")
    plt.title("forest")
    plt.legend()
    plt.grid(True)
    save_plot("forest.pdf", output)


def main(output=output):
    iris(output)
    synthetic(output)
    california(output)
    pivots(output)
    matrix_free(output)
    forest(output)


if __name__ == "__main__":
    main()
