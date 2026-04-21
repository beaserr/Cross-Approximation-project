def kernel_test_matrix(n = 150, r = 10)       
K = np.random.randn(n, r)
A = K @ K.T
