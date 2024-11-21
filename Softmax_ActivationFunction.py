import numpy as np

#layer_outputs = [4.8, 1.21, 2.385]

#E = 2.71828182846 # Euler's number
#exp_values = []
#for output in layer_outputs:
#    exp_values.append(E ** output)
#print(exp_values)

#norm_base = sum(exp_values)
#norm_values = []
#for value in exp_values:
#    norm_values.append(value / norm_base)
#print(norm_values)

# With numpy
#exp_values = np.exp(layer_outputs)
#norm_values = exp_values / np.sum(exp_values)
#print(norm_values)

class Activation_Softmax:

    def forward(self, inputs):

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

inputs = [-2, -1, 0] #[1, 2, 3]
softmax = Activation_Softmax()
softmax.forward([inputs])
print(softmax.output)