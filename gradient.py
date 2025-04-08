import numpy as np

# Example dataset
x = np.array([5,7,10])
y = np.array([4,6,8])

# Initialize parameters
m = 0
b = 0

# Set learning rate and convergence threshold
alpha = 0.01
epsilon = 1e-6

# Maximum number of iterations
N_max = 1000 # 10000 will see it approximate the exact value

# Function to compute gradients
def compute_gradients(m, b, x, y):
    n = len(x)
    y_pred = m * x + b
    dm = -2/n * np.sum(x * (y - y_pred))
    db = -2/n * np.sum(y - y_pred)
    return {'dm': dm, 'db': db}

# Lists to store gradients
dm_list = []
db_list = []
ep_list = []
# Gradient descent loop
for i in range(N_max):
    gradients = compute_gradients(m, b, x, y)
    dm_list.append(gradients['dm'])
    db_list.append(gradients['db'])
    
    m -= alpha * gradients['dm']
    b -= alpha * gradients['db']
    
    # Check convergence
    ep  = np.linalg.norm([gradients['dm'], gradients['db']]) # Euclidean value
    ep_list.append(ep)
    if  ep < epsilon:
        break

# Print the results
print (dm_list[:10])
print (db_list[:10])
print (ep_list[:10])
print(f"Converged after {i+1} iterations")
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")