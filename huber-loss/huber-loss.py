import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    yt=np.asarray(y_true)
    yp=np.asarray(y_pred)
    e=yt-yp
    e=np.absolute(e)
    err=np.where(e<=delta , 0.5*(e**2) , delta*(e - 0.5*delta))
    return np.mean(err)




