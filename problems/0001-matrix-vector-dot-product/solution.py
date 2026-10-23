def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:

	num_rows = len(a)
	num_cols = len(a[0])
	vector_length = len(b)

	if num_cols != vector_length:
		return -1

	result = []
    
	for row in a:
		sum = 0
		for i in range(num_cols):
			sum += row[i] * b[i]

		result.append(sum)

	return result 	


	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	#pass