import numpy as np

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Impute missing values in a 2D array using the specified strategy.
    
    Args:
        data: 2D numpy array with missing values represented as np.nan
        strategy: Imputatinp.on strategy - 'mean', 'median', or 'mode'
        
    Returns:
        2D numpy array with missing values imputed
    """
    # Your code here

    result  = data.copy()
    n_cols = result.shape[1]

    for col_idx in range(n_cols):
        col = result[:,col_idx]
        non_missing = col[~np.isnan(col)]

        if strategy == 'mean':
            fill_value = np.mean(non_missing)
        elif strategy == 'median':
            fill_value = np.median(non_missing)
        elif strategy == 'mode':
            vals, counts = np.unique(non_missing, return_counts=True)
            max_count = np.max(counts)
            modes = vals[counts == max_count]
            fill_value = np.min(modes)

        col[np.isnan(col)] = fill_value
        result[:,col_idx] = col

    return result

