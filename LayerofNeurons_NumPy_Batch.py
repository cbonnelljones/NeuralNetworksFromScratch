import numpy as np

inputs = [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]
weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]
biases = [2.0, 3.0, 0.5]

outputs = np.dot(np.array(inputs), np.array(weights).T) + biases

# Should be 
# [[ 4.8    1.21   2.385]
# [ 8.9   -1.81   0.2  ]
# [ 1.41   1.051  0.026]]
print(outputs)