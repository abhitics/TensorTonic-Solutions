import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x_arr = np.asarray(x)
    p_arr = np.asarray(p)
    
    # 1. Validation: Ensure x and p have the same number of elements
    if x_arr.shape != p_arr.shape:
        raise ValueError("Inputs x and p must have the same shape.")
        
    # 2. Validation: Ensure individual probabilities are between 0 and 1
    if np.any(p_arr < 0) or np.any(p_arr > 1):
        raise ValueError("Probabilities must be between 0 and 1 inclusive.")
        
    # 3. Validation: Ensure all probabilities sum up to 1 (using float tolerance)
    if not np.isclose(np.sum(p_arr), 1.0):
        raise ValueError("The sum of probabilities must equal 1.0.")
        
    # 4. Compute expected value safely
    return float(np.sum(x_arr * p_arr))
