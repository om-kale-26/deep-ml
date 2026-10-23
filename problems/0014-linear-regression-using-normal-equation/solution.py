import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    # Convert inputs to NumPy arrays
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)  # Ensure y is a column vector
    
    # Apply the normal equation: theta = (X^T * X)^(-1) * X^T * y
    # np.linalg.pinv is used instead of np.linalg.inv for numerical stability (handles singular matrices)
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    
    # Flatten the result to a 1D list and round to 4 decimal places
    return [round(float(val), 4) for val in theta.flatten()]