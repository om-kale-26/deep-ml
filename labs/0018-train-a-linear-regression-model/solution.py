# --- cell 1 ---
import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # 1. Add a column of ones to X to account for the bias term (b)
    ones = np.ones((X.shape[0], 1))
    X_b = np.concatenate([ones, X], axis=1)
    
    # 2. Solve the linear regression equation directly using least squares
    theta = np.linalg.lstsq(X_b, y, rcond=None)[0]
    
    # 3. Separate the bias (first element) and the weights (remaining elements)
    b_trained = theta[0]
    W_trained = theta[1:]
    
    return W_trained, b_trained
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.