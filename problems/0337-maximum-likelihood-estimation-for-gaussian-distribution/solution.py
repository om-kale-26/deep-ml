import numpy as np

def gaussian_mle(data: np.ndarray) -> tuple:
    """
    Compute Maximum Likelihood Estimates for Gaussian distribution parameters.
    
    Args:
        data: 1D numpy array of observations
        
    Returns:
        Tuple of (mean_mle, variance_mle)
    """
    # Your code here
    # 1. Calculate the MLE mean (the average of the array)
    mean_mle = np.mean(data)
    
    # 2. Calculate the MLE variance (divide by n, using np.var with ddof=0)
    variance_mle = np.var(data)
    
    # 3. Return both as a tuple
    return (mean_mle, variance_mle)
    