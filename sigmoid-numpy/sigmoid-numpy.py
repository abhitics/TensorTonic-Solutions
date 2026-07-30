import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function supporting both lists and NumPy arrays.
    """
    # Force the input to be a NumPy array so math operations work
    x = np.asarray(x)
    
    # Numerically stable calculation
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))
