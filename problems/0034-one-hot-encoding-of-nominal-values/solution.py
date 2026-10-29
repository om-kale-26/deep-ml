import numpy as np

def to_categorical(x: np.ndarray, n_col: int = None) -> np.ndarray:
    """
    Perform one-hot encoding of nominal values.
    """
    if n_col is None:
        n_col = np.max(x) + 1
        
    n_rows = len(x)
    
    # Initialize a matrix of zeros
    one_hot = np.zeros((n_rows, n_col), dtype=float)
    
    # Set 1 at the indices specified by x for each row
    one_hot[np.arange(n_rows), x] = 1.0
    
    return one_hot