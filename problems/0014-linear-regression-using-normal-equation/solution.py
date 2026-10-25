import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round

	#convert into numpy arrays 
	X = np.array(X)
	y = np.array(y)

	#X transpose 
	X_transpose = X.T

	#X_transpose * X 
	XT_X = X_transpose @ X 

	#inverse of XT_X
	XT_X_inv = np.linalg.inv(XT_X)

	#X_transpose * y 
	XT_y = X_transpose @ y 

	# theta 
	theta = XT_X_inv @ XT_y

	return theta