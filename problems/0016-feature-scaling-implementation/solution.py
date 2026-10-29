import numpy as np 

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here

	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)

	# standardization techinque
	standardized = np.where(std == 0, 0.0, (data - mean) / std)
	standardized = np.round(standardized, 4)

	# min max normalization

	min_val = np.min(data, axis = 0)
	max_val = np.max(data, axis = 0)

	den = max_val - min_val

	normalized = np.where(den == 0, 0.0, (data - min_val) / den)
	normalized = np.round(normalized, 4) 






	return standardized, normalized