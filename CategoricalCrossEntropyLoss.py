import math
import numpy as np

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        
        # Number of samples in class
        samples = len(y_pred)

        # Clip data to prevent division by 0, clip both sides so as to not drag the mean toward any value
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


softmax_output = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]
])
class_targets = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0]
])
loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(softmax_output, class_targets)
print(loss)