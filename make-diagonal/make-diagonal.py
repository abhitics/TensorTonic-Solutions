import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    # Convert input to a standard 1D numpy array
    v_arr = np.asarray(v)
    
    # Create the square matrix with v on the main diagonal
    return np.diag(v_arr)
