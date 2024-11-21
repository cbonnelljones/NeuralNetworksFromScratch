import numpy as np

dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])

# We have 3 sets of inputs
inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])

# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# One bias for each neuron
# biases are the row vector the the shape (1, neurons)
biases = np.array([[2, 3, 0.5]])

# Forward pass
layer_outputs = np.dot(inputs, weights) + biases # dense layer
relu_outputs = np.maximum(0, layer_outputs) # ReLU activation

# Let's optimize and test backpropagation here
# ReLU activation - simulates derivative with respect to input values
# from next layer passed to current layer during backpropagation
drelu = relu_outputs.copy()
drelu[layer_outputs <= 0] = 0

# Dense layer
# dinputs - multiply by weights
dinputs = np.dot(drelu, weights.T)
# dweights - multiply by inputs
dweights = np.dot(inputs.T, drelu)
# dbiases - sum values, do this over samples (first axis), keepdims
# since this by default will produce a plain list - 
# we explained this in chapter 4
dbiases = np.sum(drelu, axis=0, keepdims=True)

# Update parameters
weights += -0.001 * dweights
biases += -0.001 * dbiases

print(weights)
print(biases)

exit()
# Sum weights related to the given input multiplied by
# the gradient related to the given neurons
#dx0 = sum([weights[0][0]*dvalues[0][0], weights[0][1]*dvalues[0][1], weights[0][2]*dvalues[0][2]])
#dx1 = sum([weights[1][0]*dvalues[0][0], weights[1][1]*dvalues[0][1], weights[1][2]*dvalues[0][2]])
#dx2 = sum([weights[2][0]*dvalues[0][0], weights[2][1]*dvalues[0][1], weights[2][2]*dvalues[0][2]])
#dx3 = sum([weights[3][0]*dvalues[0][0], weights[3][1]*dvalues[0][1], weights[3][2]*dvalues[0][2]])

# Simplified version
#dx0 = sum(weights[0]*dvalues[0])
#dx1 = sum(weights[1]*dvalues[0])
#dx2 = sum(weights[2]*dvalues[0])
#dx3 = sum(weights[3]*dvalues[0])

#dinputs = np.array([dx0, dx1, dx2, dx3])

# Even more simple
dinputs = np.dot(dvalues, weights.T)
dweights = np.dot(inputs.T, dvalues)
dbiases = np.sum(dvalues, axis=0, keepdims=True)

#print(dinputs)
#print(dweights)
print(dbiases)