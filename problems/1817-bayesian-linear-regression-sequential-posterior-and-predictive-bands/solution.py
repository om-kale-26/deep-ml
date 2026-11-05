import numpy as np

def posterior(X, y, m0, S0, sigma2):
    # Ensure y is a column vector (N, 1) if it's 1D
    y = np.atleast_2d(y).T
    X = np.atleast_2d(X)
    
    # 1. Prior precision: S0^(-1)
    S0_inv = np.linalg.inv(S0)
    
    # 2. Posterior covariance: SN = (S0^(-1) + (1 / sigma2) * X^T * X)^(-1)
    data_precision = (1.0 / sigma2) * np.dot(X.T, X)
    SN = np.linalg.inv(S0_inv + data_precision)
    
    # 3. Posterior mean: mN = SN * (S0^(-1) * m0 + (1 / sigma2) * X^T * y)
    term1 = np.dot(S0_inv, m0)
    term2 = (1.0 / sigma2) * np.dot(X.T, y).flatten()
    mN = np.dot(SN, term1 + term2)
    
    return mN, SN

def sequential_posterior(X, y, m0, S0, sigma2):
    X = np.atleast_2d(X)
    y = np.atleast_2d(y).flatten()
    
    N, D = X.shape
    means = np.zeros((N, D))
    
    current_m = m0
    current_S = S0
    
    # Process rows one at a time sequentially
    for i in range(N):
        xi = X[i:i+1, :]
        yi = np.array([y[i]])
        
        current_m, current_S = posterior(xi, yi, current_m, current_S, sigma2)
        means[i] = current_m
        
    return current_m, current_S, means

def predictive(X_test, mN, SN, sigma2):
    X_test = np.atleast_2d(X_test)
    
    # Mean: mN^T * x for each test row -> X_test @ mN
    mean = np.dot(X_test, mN)
    
    # Variance: sigma2 + x^T * SN * x for each test row
    # Computed efficiently row-by-row using matrix multiplication
    var = sigma2 + np.sum(np.dot(X_test, SN) * X_test, axis=1)
    
    return mean, var