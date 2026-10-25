import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    m, n = X.shape  # m = number of samples, n = number of features
    y = np.array(y)
    
    for epoch in range(n_epochs):
        
        # --- METHOD 1: BATCH GRADIENT DESCENT ---
        if method == 'batch':
            predictions = X @ weights
            error = predictions - y
            # Added the factor of 2 for standard MSE gradient
            gradient = (2 / m) * (X.T @ error)
            weights = weights - learning_rate * gradient

        # --- METHOD 2: STOCHASTIC GRADIENT DESCENT ---
        elif method == 'stochastic':
            for i in range(m):
                Xi = X[i:i+1]       # Shape (1, n)
                yi = y[i]           # Scalar value
                
                prediction = Xi @ weights
                error = prediction - yi
                # For a single sample, gradient includes 2 * error * Xi
                gradient = 2 * error[0] * Xi[0]
                weights = weights - learning_rate * gradient

        # --- METHOD 3: MINI-BATCH GRADIENT DESCENT ---
        elif method == 'mini_batch':
            for i in range(0, m, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                batch_m = X_batch.shape[0]
                predictions = X_batch @ weights
                error = predictions - y_batch
                # Added the factor of 2 here as well
                gradient = (2 / batch_m) * (X_batch.T @ error)
                weights = weights - learning_rate * gradient
                
    return weights