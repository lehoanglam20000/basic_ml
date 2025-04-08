
'''
Gradient Descent
Iterative Approach: Gradient descent is an iterative optimization algorithm that gradually adjusts the parameters to minimize the cost function. It starts with initial guesses and updates the parameters in small steps based on the gradient.
Approximate Solution: While gradient descent can converge to the optimal solution, it may take many iterations and depends on the learning rate and convergence criteria. It is particularly useful for large datasets where matrix inversion (required by the Normal Equation) is computationally expensive.
    
    '''
import numpy as np

# Example dataset
x = np.array([5,7,10])
y = np.array([4,6,8])

# Initialize parameters
m = 0
n = 0

# Set learning rate and convergence threshold
alpha = 0.01
epsilon = 1e-6

# Maximum numner of iterations
N_max = 10000 # 10000 will see it approximate the exact value, 1000 doesnot reach a good enough solution

# Function to compute gradients
def compute_gradients(m, n, x, y):
    size = len(x)
    y_pred = m * x + n
    dm = -2/size * np.sum(x * (y - y_pred))
    dn = -2/size * np.sum(y - y_pred)
    return {'dm': dm, 'dn': dn}

# Lists to store gradients
dm_list = []
dn_list = []
ep_list = []
# Gradient descent loop
for i in range(N_max):
    gradients = compute_gradients(m, n, x, y)
    dm_list.append(gradients['dm'])
    dn_list.append(gradients['dn'])
    
    m -= alpha * gradients['dm']
    n -= alpha * gradients['dn']
    
    # Check convergence
    ep  = np.linalg.norm([gradients['dm'], gradients['dn']]) # Euclidean value
    ep_list.append(ep)
    if  ep < epsilon:
        break

# Print the results
print (dm_list[:10])
print (dn_list[:10])
print (ep_list[:10])
print(f"Converged after {i+1} iterations")
print(f"Slope (m): {m}")
print(f"Intercept (n): {n}")