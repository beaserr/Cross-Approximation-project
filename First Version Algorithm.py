import numpy as np
import matplotlib.pyplot as plt

def cross_approx(A, maxi, epsilon = 1e-6):
    m, n = A.shape
    R = A
    U = np.zeros((m, maxi))
    V = np.zeros((n, maxi))
    for k in range(maxi):
        maxi = 0
        pivot_row = 0
        pivot_col = 0
        for i in range(m):
            for j in range(n):
                if abs(R[i, j]) > maxi:
                    maxi = abs(R[i, j])
                    pivot_row = i
                    pivot_col = j
        if maxi < epsilon:
            print(k)
            break
        U[:, k] = R[:, pivot_col]
        V[:, k] = R[pivot_row, :]/ R[pivot_row, pivot_col]
        R = R - np.outer(U[:, k], V[:, k]) 
    return U[:, :k], V[:, :k]       


#Example with an matrix 

np.random.seed(0)


A = np.random.rand(100, 100)
U, V = cross_approx(A, maxi =100)
A_approx = U @ V.T
error = np.linalg.norm(A - A_approx, 'fro')/ np.linalg.norm(A, 'fro')
print("Frobenius norm of the error:", error)  
#Plot singular values and compare to SVD
#Error in the 2-norm 

#Exponential decay of singular values
m, n = 50, 50
r= 10
U, _ = np.linalg.qr(np.random.rand(m, m))
V, _ = np.linalg.qr(np.random.rand(n, n))
singular_values = np.exp(-np.arange(r))
A = U[:, :r] @ np.diag(singular_values) @ V[:, :r].T
U_approx, V_approx = cross_approx(A, maxi =10)
A_approx = U_approx @ V_approx.T
error = np.linalg.norm(A - A_approx, 'fro')/ np.linalg.norm(A, 'fro')
print("Frobenius norm of the error for exponential decay:", error)   

