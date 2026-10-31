import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	# Your code here

	a = np.mean((y_true - X @ w)**2)

	b = alpha * np.sum((w)**2)

	return a + b
