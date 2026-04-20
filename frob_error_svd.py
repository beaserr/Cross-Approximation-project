#Calculate the froebenius error of SVD method 

def frob_error_svd(A, max_rank):
    s = np.linalg.svd(A, compute_uv=False)
    norm_fro = np.sqrt(np.sum(s**2))

    return [
        np.sqrt(np.sum(s[k:]**2)) / norm_fro
        for k in range(1, max_rank + 1)
    ]
