# --- cell 1 ---
import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """Split rows into train/val/test folds, fit something on the training fold, predict the test fold.

    Returns
    -------
    test_predictions : np.ndarray, shape (len(test_idx),)
    train_idx, val_idx, test_idx : 1-D integer arrays forming a partition of range(len(y))
    """
    
    n = len(y)
    
    # 1. Shuffle and partition indices using the seed
    indices = np.random.default_rng(seed).permutation(n)
    
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    # 2. Fit a predictor on the training set only (using mean target as a robust baseline)
    train_mean = np.mean(y[train_idx])
    
    # 3. Predict on the test set
    test_predictions = np.full(shape=(len(test_idx),), fill_value=train_mean)
    
    return test_predictions, train_idx, val_idx, test_idx
    pass