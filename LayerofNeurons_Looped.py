inputs = [1, 2, 3, 2.5]

# List of weights
weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

biases = [2, 3, 0.5]

# Output of current layer
layer_outputs = []
# For each neuron
#print(tuple(zip(weights, biases)))
for neuron_weights, neuron_bias in zip(weights, biases):
    # Zeroed output of given neuron
    neuron_output = 0
    # for each input and weight to the neuron
    #print("Neuron Weight: ", neuron_weights)
    #print("Neuron Bias: ", neuron_bias)
    for n_input, weight in zip(inputs, neuron_weights):
        # Multiply this input by associated weight
        # and add to the neuron's output variable
        neuron_output += n_input*weight
    # Add bias
    neuron_output += neuron_bias
    # Add result to layer's output list
    layer_outputs.append(neuron_output)

# Should be [4.8, 1.21, 2.385]
print(layer_outputs)