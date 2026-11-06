import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    # 1. Create indices [0, 1, ..., n_samples - 1]
    indices = np.arange(n_samples)
    
    # 2. Shuffle if requested
    if shuffle:
        np.random.shuffle(indices)
        
    # 3. Determine fold sizes (handles cases where n_samples isn't perfectly divisible by k)
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    
    # 4. Slice indices into folds
    folds = []
    current = 0
    for size in fold_sizes:
        folds.append(indices[current:current + size].tolist())
        current += size
        
    # 5. Create train/test index pairs for each fold
    result = []
    for i in range(k):
        test_indices = folds[i]
        # Train indices are all folds except the current test fold
        train_indices = []
        for j in range(k):
            if i != j:
                train_indices.extend(folds[j])
                
        result.append((train_indices, test_indices))
        
    return result