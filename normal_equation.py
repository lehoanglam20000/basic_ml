'''Normal Equation
Exact Solution: The Normal Equation 
 y computes the parameters (slope and intercept) directly, providing the exact values that minimize the cost function (e.g., Mean Squared Error).
Closed-form: It is a closed-form solution, meaning it doesn't require iterative steps. It uses matrix operations to find the optimal parameters in one go.'
''
'''

import numpy as np

# Example dataset
x = np.array([5,7,10])
y = np.array([4,6,8])

# Number of data points
size = len(x)

# Create the design matrix A
A = np.vstack([x, np.ones(size)]).T

print (A)
# Compute the coefficients using the normal equation
coefficients = np.linalg.inv(A.T @ A) @ A.T @ y

# Extract the slope (m) and intercept (n)
m, n = coefficients

print(f"Slope (m): {m}")
print(f"Intercept (n): {n}")

'''
 Matrix inversion is computationally expensive, especially for large matrices. The time complexity of matrix inversion is approximately O(n3)
Practically, for dense matrices, handling matrices larger than 10,000 x 10,000 elements can be challenging on a typical desktop computer.

'''