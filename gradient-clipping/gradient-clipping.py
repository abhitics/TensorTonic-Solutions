import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping, matching strict platform requirements.
    """
    g_arr = np.asarray(g)
    
    total_norm = np.linalg.norm(g_arr)
    
    if max_norm <= 0 or total_norm == 0:
        return g_arr.copy()
        
    if total_norm <= max_norm:
        return g_arr.copy()

    return g_arr * (max_norm / total_norm)
