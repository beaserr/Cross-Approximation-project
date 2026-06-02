import numpy as np


def fpca(a, max_rank, eps=1e-12):
    m, n = a.shape
    r = a.copy()
    u = np.zeros((m, max_rank))
    v = np.zeros((n, max_rank))
    rank = 0

    for k in range(max_rank):
        i, j = np.unravel_index(np.argmax(np.abs(r)), r.shape)
        pivot = r[i, j]

        if abs(pivot) < eps:
            break

        col = r[:, j]
        row = r[i, :] / pivot

        u[:, k] = col
        v[:, k] = row

        r = r - np.outer(col, row)
        rank = k + 1

    return u[:, :rank], v[:, :rank]


def ppca(a, max_rank, eps=1e-12):
    m, n = a.shape
    u = np.zeros((m, max_rank))
    v = np.zeros((n, max_rank))
    row = 0
    rank = 0

    for k in range(max_rank):
        b = a[row, :].copy()

        for i in range(k):
            b = b - u[row, i] * v[:, i]

        col = np.argmax(np.abs(b))
        pivot = b[col]

        if abs(pivot) < eps:
            break

        c = a[:, col].copy()

        for i in range(k):
            c = c - u[:, i] * v[col, i]

        c = c / pivot

        u[:, k] = c
        v[:, k] = b

        next_row = np.abs(c)
        next_row[row] = 0
        row = np.argmax(next_row)

        rank = k + 1

    return u[:, :rank], v[:, :rank]


def ppca_error(a, max_rank, eps=1e-12):
    errors = []
    norm = np.linalg.norm(a, "fro")

    for k in range(1, max_rank + 1):
        u, v = ppca(a, k, eps)
        approx = u @ v.T
        error = np.linalg.norm(a - approx, "fro") / norm
        errors.append(error)

    return errors


def ppca_function(f, m, n, max_rank, eps=1e-12):
    a = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            a[i, j] = f(i, j)

    return ppca_error(a, max_rank, eps)


def random_ppca(a, max_rank, eps=1e-12, seed=0):
    m, n = a.shape
    u = np.zeros((m, max_rank))
    v = np.zeros((n, max_rank))
    rng = np.random.default_rng(seed)
    rank = 0

    for k in range(max_rank):
        row = rng.integers(0, m)
        b = a[row, :].copy()

        for i in range(k):
            b = b - u[row, i] * v[:, i]

        cols = np.where(np.abs(b) > eps)[0]

        if len(cols) == 0:
            break

        col = rng.choice(cols)
        pivot = b[col]

        if abs(pivot) < eps:
            break

        c = a[:, col].copy()

        for i in range(k):
            c = c - u[:, i] * v[col, i]

        c = c / pivot

        u[:, k] = c
        v[:, k] = b
        rank = k + 1

    return u[:, :rank], v[:, :rank]


def weighted_ppca(a, max_rank, eps=1e-12, seed=0):
    m, n = a.shape
    u = np.zeros((m, max_rank))
    v = np.zeros((n, max_rank))
    rng = np.random.default_rng(seed)
    row = rng.integers(0, m)
    rank = 0

    for k in range(max_rank):
        b = a[row, :].copy()

        for i in range(k):
            b = b - u[row, i] * v[:, i]

        weights = np.abs(b)
        weights[weights < eps] = 0

        if np.sum(weights) < eps:
            break

        col = rng.choice(n, p=weights / np.sum(weights))
        pivot = b[col]

        if abs(pivot) < eps:
            break

        c = a[:, col].copy()

        for i in range(k):
            c = c - u[:, i] * v[col, i]

        c = c / pivot

        u[:, k] = c
        v[:, k] = b

        weights = np.abs(c)
        weights[row] = 0
        weights[weights < eps] = 0

        if np.sum(weights) < eps:
            break

        row = rng.choice(m, p=weights / np.sum(weights))
        rank = k + 1

    return u[:, :rank], v[:, :rank]


def natural_ppca(a, max_rank, eps=1e-12):
    m, n = a.shape
    r = min(m, n, max_rank)
    u = np.zeros((m, r))
    v = np.zeros((n, r))
    rank = 0

    for k in range(r):
        b = a[k, :].copy()

        for i in range(k):
            b = b - u[k, i] * v[:, i]

        pivot = b[k]

        if abs(pivot) < eps:
            break

        c = a[:, k].copy()

        for i in range(k):
            c = c - u[:, i] * v[k, i]

        c = c / pivot

        u[:, k] = c
        v[:, k] = b
        rank = k + 1

    return u[:, :rank], v[:, :rank]