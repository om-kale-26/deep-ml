import numpy as np

def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)

def l1_regularization_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4):
    n_samples, n_features = X.shape
    
    # Initialize weights and bias
    w = np.zeros(n_features)
    b = 0.0
    
    for _ in range(max_iter):
        y_pred = X @ w + b
        
        # Gradients
        dw = -(2 / n_samples) * (X.T @ (y - y_pred))
        db = -(2 / n_samples) * np.sum(y - y_pred)
        
        # Gradient descent step
        w_old = w.copy()
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Soft-thresholding for L1 penalty
        w = soft_threshold(w, learning_rate * alpha)
        
        # Convergence check
        weight_change = np.linalg.norm(w - w_old)
        if weight_change < tol:
            break
            
    return w, float(b)