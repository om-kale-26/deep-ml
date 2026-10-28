import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    # 1. Initialize an array of zeros with the exact same shape as x
    numerical_grad = np.zeros_like(x, dtype=float)
    
    # 2. Iterate through every single element coordinate in the array x
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        
        # Save the original value at this index position
        old_val = x[idx]
        
        # Compute f(x + epsilon)
        x[idx] = old_val + epsilon
        fx_plus = f(x.copy())
        
        # Compute f(x - epsilon)
        x[idx] = old_val - epsilon
        fx_minus = f(x.copy())
        
        # Reset the value back to its original state
        x[idx] = old_val
        
        # Calculate the centered finite difference approximation
        numerical_grad[idx] = (fx_plus - fx_minus) / (2 * epsilon)
        
        it.iternext()
        
    # 3. Calculate Euclidean L2 norms for the relative error formula
    diff_norm = np.linalg.norm(analytical_grad - numerical_grad)
    analytical_norm = np.linalg.norm(analytical_grad)
    numerical_norm = np.linalg.norm(numerical_grad)
    
    # 4. Handle the edge case where both gradients are exactly zero
    denominator = analytical_norm + numerical_norm
    if denominator == 0.0:
        relative_error = 0.0
    else:
        relative_error = diff_norm / denominator
        
    return numerical_grad, float(relative_error)
