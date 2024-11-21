import numpy as np

a = [1, 2, 3]
b = [2, 3, 4]

a = np.array([a])
# Transpose row vector b into a column vector
b = np.array([b]).T

# Should be [[20]]
print(np.dot(a, b))
