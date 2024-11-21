import numpy as np
import cv2
import os
import pickle
import copy
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import sine_data
import matplotlib.pyplot as plt

nnfs.init()

# Dense layer
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        #self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases
    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Gradients on regularization
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights 
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases 
        
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
    
    def get_parameters(self):
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs

class Layer_Dropout:
    # Init
    def __init__(self, rate):
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate        

    def forward(self, inputs, training):
        self.inputs = inputs
        
        if not training:
            self.output = inputs.copy()
            return
        
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

class Activation_ReLU:
    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from input
        self.output = np.maximum(0, inputs)
    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify the original variable, 
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
    def predictions(self, outputs):
        return outputs

class Activation_Linear:
    def forward(self, inputs, training):
        # Just remember values
        self.inputs = inputs
        self.output = inputs
    def backward(self, dvalues):
        # Devivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()
    def predictions(self, outputs):
        return outputs

class Activation_Sigmoid:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
    def predictions(self, outputs):
        return (outputs > 0.5) * 1

class Activation_Softmax:
    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate amlge-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)
    
# SGC optimizer
class Optimizer_SGD:
    # Initialize optimizer - set settings
    # learning rate of 1 is defualt for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    # Update parameters
    def update_params(self, layer):
        # If we use momentum
        if self.momentum:
            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either
                layer.bias_momentums = np.zeros_like(layer.biases)
            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            # Build bias updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        # Vanilla SGD updates (as before momentum updates)
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        layer.weights += weight_updates
        layer.biases += bias_updates

    # Call once after any parameter updates 
    def post_update_params(self):
        self.iterations += 1

# Adagrad optimizer
class Optimizer_Adagrad:
    # Initiate optimizer - set settings
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before and parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once oafter any parameter updates
    def post_update_params(self):
        self.iterations += 1

# RMSprop optimizer
class Optimizer_RMSprop:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square root cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# Adam optimizer
class Optimizer_Adam:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need it to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradient
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1- self.beta_2 ** (self.iterations +1))

        # Vanilla SGD parameter update + normalization
        # with square root cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Loss:

    def regularization_loss(self):
        regularization_loss = 0
        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        return regularization_loss
    
    # Set/remember trainable layer
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers 
    
    # Calculates the data and regularization losses
    # given model output and ground truth values

    def calculate(self, output, y, *, include_regularization=False):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean losses
        data_loss = np.mean(sample_losses)
        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        if not include_regularization:
            return data_loss
        # Return loss
        return data_loss, self.regularization_loss()
    
    # Calculate accumulated loss
    def calculate_accumulated(self, *, include_regularization=False):
        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count
        # If just data loss - return it
        if not include_regularization:
            return data_loss
        # Return the data and regularization losses
        return data_loss, self.regularization_loss()
    
    # Reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # Number of samples in class
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # clip both sides so as to not drag the mean toward any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        # Probabilities for target values
        # Only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # One-hot encoded
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Loss_BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses
    def backward(self, dvalues, y_true):
        # NUmber of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples

class Loss_MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples

# Softmax classifier - combined Softmax activation 
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():   
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded, 
        # turn them into discrete labels
        if len(y_true.shape) == 2:
            y_true - np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Accuracy:
    # Calculate an accuracy
    # given predictions and ground truth values
    def calculate(self, predictions, y):
        # Get comparison results
        comparisons = self.compare(predictions, y)
        # Calculate an accuracy
        accuracy = np.mean(comparisons)
        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return accuracy
    
    # Calculate accumulated accuracy
    def calculate_accumulated(self):
        # Calculate mean loss
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy
    
    # Reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Accuracy_Regression(Accuracy):
    def __init__(self):
        # Create precisin property
        self.precision = None
    
    # Calculate precision values
    # based on passed-n ground truth
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
    
    # Compare predictions to ground truth
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision

class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        # Binary mode?
        self.binary = binary
    # No initialization needed
    def init(self, y):
        pass
    # Compare predictions to ground truth values
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

class Model:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None
    
    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    def set_parameters(self, parameters):
        # Iterate over the parameters and layers
        # and update each layer with each set of the parameters
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)
    
    def get_parameters(self):
        # Create list of parameters
        parameters = []
        # Iterate trainable layers and get their prameters
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters

    def save_parameters(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
    
    def load_parameters(self, path):
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))
    
    def save(self, path):
        model = copy.deepcopy(self)
        model.loss.new_pass()
        model.accuracy.new_pass()
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def finalize(self):
        # Create and set the input layer
        self.input_layer = Layer_Input()
        # Count all the objects
        layer_count = len(self.layers)
        # Initialize a list containing trainable layers
        self.trainable_layers = []
        # Iterate the objects
        for i in range(layer_count):
            # If it's the first layer,
            # there previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            # If layer contains attribute called "weights",
            # it's a trainable layer, add to "trainable_layers" list
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        # Update loss object with trainable layers
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

    def evaluate(self, X_val, y_val, *, batch_size=None):
        validation_steps = 1
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]

            # Perform forward pass
            output = self.forward(batch_X, training=False)
            # Calculate the loss
            self.loss.calculate(output, batch_y)
            # Get predictions and calculate an accuracy
            predeictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predeictions, batch_y)

        # Get and print validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f'Validation: ' + 
                f'Accuracy: {validation_accuracy:.3f}, ' +
                f'Loss: {validation_loss:.3f}')
            
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        
        # initialize accuracy object
        self.accuracy.init(y)
        
        # Default value if batch size is not being set
        train_steps = 1
        
        # If there is validation data passes,
        # set default number of steps for validation as well
        if validation_data is not None:
            validation_steps = 1
            # For better readability
            X_val, y_val = validation_data
        
        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1
            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        # Main training loop
        for epoch in range(1, epochs + 1):
            
            print(f'Epoch: {epoch}')

            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over steps
            for step in range(train_steps):

                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                # Perform forward pass
                output = self.forward(batch_X, training=True)
                # Calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss
                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)
                # Perform backward pass
                self.backward(output, batch_y)
                # Optimize
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not step % print_every or step == train_steps - 1:
                    print(f'Step: {step}, ' +
                        f'Accuracy: {accuracy:.3f}, ' +
                        f'Loss: {loss:.3f}, (' +
                        f'Data Loss: {data_loss:.3f}, ' +
                        f'Reg Loss: {regularization_loss:.3f}), ' +
                        f'Lr: {self.optimizer.current_learning_rate}')

        # Get and print epoch loss and accuracy
        epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
        epoch_loss = epoch_data_loss + epoch_regularization_loss
        epoch_accuracy = self.accuracy.calculate_accumulated()

        print(f'Training, ' +
              f'Accuracy: {epoch_accuracy:.3f}, ' +
              f'Loss: {epoch_loss:.3f}, (' +
              f'Data Loss: {epoch_data_loss:.3f}, ' +
              f'Reg Loss: {epoch_regularization_loss:.3f}), ' +
              f'Lr: {self.optimizer.current_learning_rate}')
        
        if validation_data is not None:
            self.evaluate(*validation_data, batch_size=batch_size)
        
    def forward(self, X, training):

        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)
        # Call forward method on every object in a chain
        # Pass output of the previous object as a parameter
        i = 0
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        # "layer" is now the last object from the list, 
        # return it's output
        return layer.output
    
    def backward(self, output, y):
        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method
            # on the combined activation/loss
            # this will set the dinputs property
            self.softmax_classifier_output.backward(output, y)
            # Since we'll not call backward method of the last layer
            # which is Softmax activation
            # as we used combined activation/loss
            # object, let's set dinputs in the object
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            # Call backward method going through
            # all the objects but the last
            # in reverse order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        # First call backward method on the loss
        # this will set dinputs properly that the last
        # layer will try to access shortly
        self.loss.backward(output, y)
        # Call backward method going through all the objects
        # in reverse order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def predict(self, X, *, batch_size=None):

        # Default value if batch size is not being set 
        predictions_steps = 1
        
        # Calculate number of predictions_steps
        if batch_size is not None:
            predictions_steps = len(X) // batch_size
            # Dividing rounds down. If there are some remaining
            # data, but not a full batch, this won't include it.
            # Add 1 to include this not full batch
            if predictions_steps * batch_size < len(X):
                predictions_steps += 1
        
        # Model outputs
        output = []

        # Iterate over steps
        for step in range(predictions_steps):
            # if batch size is not set
            # train using one step and full dataset
            if batch_size is None:
                batch_X = X
            
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]
            
            # Perform the forward pass
            batch_output = self.forward(batch_X, training=False)
        
            # Append batch prediction to the list of predictions
            output.append(batch_output)

        # Stack and return results
        return np.vstack(output)

def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))

    X = []
    y = []

    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)

            X.append(image)
            y.append(label)
        
    return np.array(X), np.array(y).astype('uint8')

def create_data_mnist(path):

    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    return X, y, X_test, y_test

fashion_mnist_labels = {
    0: "T-shirt/top",
    1: "Pants",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Shoe",
    8: "Bag",
    9: "Boot"
}


max_scale = 255
half_scale = max_scale / 2

# Read an image
image_data = cv2.imread("pants.jpg", cv2.IMREAD_GRAYSCALE)

# Resize to the same size as the Fashion MNIST images
image_data = cv2.resize(image_data, (28,28))

# Invert image colors
image_data = 225 - image_data

plt.imshow(image_data, cmap="gray")
plt.show()

# Reshape and scale pixel data
image_data = (image_data.reshape(1, -1).astype(np.float32) - half_scale) / half_scale

# Create dataset
# X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

## Shuffle training data set
#keys = np.array(range(X.shape[0]))
#np.random.shuffle(keys)

#X = X[keys]
#y = y[keys]

# Scale and reshape samples
# X = (X.reshape(X.shape[0], -1).astype(np.float32) - half_scale) / half_scale
# X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - half_scale) / half_scale

model = Model.load('fasion_mnist.model')
#confidences = model.predict(X_test[:5])
confidences = model.predict(image_data)
predictions = model.output_layer_activation.predictions(confidences)
#print(confidences)
#print(y_test[:5])
print(predictions)
for prediction in predictions:
    print(fashion_mnist_labels[prediction])

#model.evaluate(X_test, y_test)

'''
model = Model()
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())
#model.add(Activation_Sigmoid())
#model.add(Layer_Dense(64, 64))
#model.add(Activation_ReLU())
#model.add(Layer_Dense(64, 1))
#model.add(Activation_Linear())

model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=1e-3),
    accuracy=Accuracy_Categorical()
)

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)
#parameters = model.get_parameters()

model.save('fasion_mnist.model')
model.save_parameters('fasion_mnist.params')


model = Model()
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())
#model.add(Activation_Sigmoid())
#model.add(Layer_Dense(64, 64))
#model.add(Activation_ReLU())
#model.add(Layer_Dense(64, 1))
#model.add(Activation_Linear())

model.set(
    loss=Loss_CategoricalCrossentropy(),
    accuracy=Accuracy_Categorical()
)

model.finalize()
#model.set_parameters(parameters)
model.load_parameters('fasion_mnist.params')
model.evaluate(X_test, y_test)
'''