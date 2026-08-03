import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    A_arr = np.asarray(A)
    
    # Validation: Ensure the matrix is 2D and square
    if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
        return None
        
    # Return the sum of diagonal elements as a scalar
    return float(np.trace(A_arr))
