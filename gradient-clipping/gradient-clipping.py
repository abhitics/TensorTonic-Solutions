import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping, matching strict platform requirements.
    """
    # 1. Safely convert input to a numpy array
    g_arr = np.asarray(g)
    
    # 2. Compute the global L2 norm
    total_norm = np.linalg.norm(g_arr)
    
    # 3. Handle Edge Cases: non-positive max_norm or exactly zero norm
    if max_norm <= 0 or total_norm == 0:
        return g_arr.copy()
        
    # 4. Check if norm is within limits (Hint 2: use g.copy() to avoid in-place modification)
    if total_norm <= max_norm:
        return g_arr.copy()
        
    # 5. Scale gradients cleanly (Hint 3)
    return g_arr * (max_norm / total_norm)
