# K[i, j] = X[i] X[j]

def kernel_matrix(X):
    X = np.asarray(X)
    return X @ X.T   
K = kernel_matrix(X)
