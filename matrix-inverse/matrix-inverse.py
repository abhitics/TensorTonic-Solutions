import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I, or None if invalid.
    """
    A_arr = np.asarray(A)
    
    # 1. Return None if the matrix is not 2D or not square
    if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
        return None
        
    # 2. Attempt inversion, catch singular matrix errors safely
    try:
        return np.linalg.inv(A_arr)
    except np.linalg.LinAlgError:
        return None
