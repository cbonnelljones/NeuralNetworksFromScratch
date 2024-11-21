import numpy as np

inputs = [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]
weights1 = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]
biases1 = [2.0, 3.0, 0.5]

weights2 = [
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13]
]
biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(np.array(inputs), np.array(weights1).T) + biases1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

# Should be 
# [[ 0.5031  -1.04185 -2.03875]
# [ 0.2434  -2.7332  -5.7633 ]
# [-0.99314  1.41254 -0.35655]]
print(layer2_outputs)