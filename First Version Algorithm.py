import numpy as np
import matplotlib.pyplot as plt


# Cross Approximation version1 with full pivoting 

def cross_approx(A, maxi, epsilon=1e-6):
    m, n = A.shape
    R = A.copy()
    U = np.zeros((m, maxi))
    V = np.zeros((n, maxi))

    errors_fro = []
    errors_2   = []

    for k in range(maxi):
        max_val    = 0
        pivot_row  = 0
        pivot_col  = 0

        for i in range(m):
            for j in range(n):
                if abs(R[i, j]) > max_val:
                    max_val   = abs(R[i, j])
                    pivot_row = i
                    pivot_col = j

        if max_val < epsilon:
            break

        U[:, k] = R[:, pivot_col]
        V[:, k] = R[pivot_row, :] / R[pivot_row, pivot_col]

        R  = R - np.outer(U[:, k], V[:, k])
        A_approx = U[:, :k+1]@ V[:, :k+1].T

        errors_fro.append(np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro'))
        errors_2  .append(np.linalg.norm(A - A_approx,   2 ) / np.linalg.norm(A,   2 ))

    return U[:, :k+1], V[:, :k+1], errors_fro, errors_2


# Adaptive Cross Approximation; firstversion with partial Pivoting 

def partial_cross_approx(A, maxi, epsilon=1e-6):
    m, n = A.shape
    R = A.copy()
    U = np.zeros((m, maxi))
    V = np.zeros((n, maxi))

    errors_fro = []
    errors_2   = []

    pivot_row = 0                           

    for k in range(maxi):
        pivot_col = np.argmax(np.abs(R[pivot_row, :]))         
        if abs(R[pivot_row, pivot_col]) < epsilon:
            break
        pivot_row = np.argmax(np.abs(R[:, pivot_col]))        

        U[:, k] = R[:, pivot_col]
        V[:, k] = R[pivot_row, :] / R[pivot_row, pivot_col]

        R        = R - np.outer(U[:, k], V[:, k])
        A_approx = U[:, :k+1] @ V[:, :k+1].T

        errors_fro.append(np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro'))
        errors_2  .append(np.linalg.norm(A - A_approx,   2 ) / np.linalg.norm(A,   2 ))

    return U[:, :k+1], V[:, :k+1], errors_fro, errors_2


# SVD error functions

def frob_error_svd(A, max_rank):
    s = np.linalg.svd(A, compute_uv=False)
    errors = []
    for k in range(1, max_rank + 1):
        errors.append(np.sqrt(np.sum(s[k:]**2)) / np.sqrt(np.sum(s**2)))
    return errors


def two_norm_error_svd(A, max_rank):
    s = np.linalg.svd(A, compute_uv=False)
    errors = []
    for k in range(1, max_rank + 1):
        errors.append(s[k] / s[0] if k < len(s) else 0.0)
    return errors


# Matrices to test  

m, n     = 50, 50
r        = 50
max_rank = 10

np.random.seed(42)
U_rand, _ = np.linalg.qr(np.random.rand(m, m))
V_rand, _ = np.linalg.qr(np.random.rand(n, n))

# Singular value  different decays : exponential, step, and polynomial
singular_values_exp  = np.concatenate ([np.ones(r-20), np.exp(-np.arange(r-30))  
#Smaller steps with the matrices 
singular_values_step = np.concatenate([np.ones(r // 2), np.zeros(r - r // 2)]) 
singular_values_poly = 1.0 / (np.arange(1, r + 1))                        

# Different low-rank matrices
A_exp  = U_rand[:, :r]@ np.diag(singular_values_exp)@ V_rand[:, :r].T
A_step = U_rand[:, :r]@ np.diag(singular_values_step)@ V_rand[:, :r].T
A_poly = U_rand[:, :r]@ np.diag(singular_values_poly)@ V_rand[:, :r].T


# Running approximations 

_, _, ca_fro_step,  ca_2_step = cross_approx(A_step, maxi=max_rank)
_, _, ca_fro_poly,  ca_2_poly = cross_approx(A_poly, maxi=max_rank)
_, _, ca_fro_exp,   ca_2_exp  = cross_approx(A_exp,  maxi=max_rank)

_, _, part_ca_fro_step, part_ca_2_step=partial_cross_approx(A_step, maxi=max_rank)
_, _, part_ca_fro_poly, part_ca_2_poly= partial_cross_approx(A_poly, maxi=max_rank)
_, _, part_ca_fro_exp,  part_ca_2_exp = partial_cross_approx(A_exp,  maxi=max_rank)

ranks = np.arange(1, max_rank + 1)

#Plot 1: Full Pivoting Cross Approximation
plt.figure()

plt.subplot(1, 3, 1)
plt.semilogy(ranks, frob_error_svd(A_step, max_rank), label='SVD Frobenius')
plt.semilogy(ranks, two_norm_error_svd(A_step, max_rank), label='SVD 2-Norm')
#plt.semilogy(ranks, ca_fro_step, label='Cross Approx Frobenius')
#plt.semilogy(ranks,   ca_2_step,   label='Cross Approx 2-Norm')
plt.title('Step Decay')
plt.xlabel('Rank k')
plt.ylabel('Relative Error')
plt.legend()

plt.subplot(1, 3, 2)
plt.semilogy(ranks, frob_error_svd(A_poly, max_rank), label='SVD Frobenius')
plt.semilogy(ranks, two_norm_error_svd(A_poly, max_rank), label='SVD 2-Norm')
#plt.semilogy(ranks, ca_fro_poly, label='Cross Approx Frobenius')
#plt.semilogy(ranks,  ca_2_poly,   label='Cross Approx 2-Norm')
plt.title('Polynomial Decay')
plt.xlabel('Rank k')
plt.legend()

plt.subplot(1, 3, 3)
plt.semilogy(ranks, frob_error_svd(A_exp, max_rank), label='SVD Frobenius')
plt.semilogy(ranks, two_norm_error_svd(A_exp, max_rank), label='SVD 2-Norm')
plt.semilogy(ranks, ca_fro_exp, label='Cross Approx Frobenius')
plt.semilogy(ranks,   ca_2_exp,   label='Cross Approx 2-Norm')
plt.title('Exponential Decay')
plt.xlabel('Rank k')
plt.legend()

plt.suptitle('Full Pivoting Cross Approximation vs SVD')
plt.show()


# Plotting Adaptive Cross Approximation 
plt.figure()

plt.subplot(1,3,1)
plt.semilogy(ranks, frob_error_svd(A_step, max_rank), label='SVD Frobenius')
plt.semilogy(ranks, two_norm_error_svd(A_step, max_rank), label='SVD 2-Norm')
plt.semilogy(ranks, part_ca_fro_step, label='Partial cross Approximation with Frobenius')
plt.semilogy(ranks,  part_ca_2_step,   label='Partial cross Approximation with 2-Norm')
plt.title('Step Decay')
plt.xlabel('Rank k')
plt.ylabel('Relative Error')
plt.legend()

plt.subplot(1, 3, 2)
plt.semilogy(ranks, frob_error_svd(A_poly, max_rank), label='SVD Frobenius')
plt.semilogy(ranks, two_norm_error_svd(A_poly, max_rank), label='SVD 2-Norm')
plt.semilogy(ranks, part_ca_fro_poly, label='Partial cross Approximation with Frobenius')
plt.semilogy(ranks,  part_ca_2_poly,   label='Partial cross Approximation with 2-Norm')
plt.title('Polynomial Decay')
plt.xlabel('Rank k')
plt.legend()

plt.subplot(1, 3, 3)
plt.semilogy(ranks, frob_error_svd(A_exp, max_rank), label='SVD Frobenius')
plt.semilogy(ranks, two_norm_error_svd(A_exp, max_rank), label='SVD 2-Norm')
plt.semilogy(ranks, part_ca_fro_exp, label='Partial cross Approximation with Frobenius')
plt.semilogy(ranks,  part_ca_2_exp,   label='Partial cross Approximation with 2-Norm')
plt.title('Exponential Decay')
plt.xlabel('Rank k')
plt.legend()
plt.suptitle('Partial Cross Approximation compared to SVD')
plt.show()
