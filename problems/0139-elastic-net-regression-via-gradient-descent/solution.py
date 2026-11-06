import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    # Implement Elastic Net regression here

    n_samples, n_features = X.shape

    w = np.zeros(n_features)
    b = 0.0

    for _ in range(max_iter):

        e = np.dot(X, w) + b - y

        grad_w = (1 / n_samples) * np.dot(X.T, e) + alpha1 * np.sign(w) + 2 * alpha2 * w
        grad_b = (1 / n_samples) * np.sum(e)
        
        # 3. Update weights and bias using gradient descent
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
        
        # 4. Check convergence (stopping condition)
        if np.sum(np.abs(grad_w)) < tol:
            break
            
    return w, b







    pass