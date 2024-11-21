import numpy as np
import cv2
import os

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

# Create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# Scale features
# image data range between 0-255
max_scale = 255
half_scale = max_scale / 2
X = (X.astype(np.float32) - half_scale) / half_scale
X_test = (X_test.astype(np.float32) - half_scale) / half_scale

print(X.min(), X.max())
print(X.shape)

# Reshape to vectors
# Current shape (6000, 28, 28)
# Flatten to 6000, 784 (28*28)
# image data needs to gofrom 2d to 1d (i.e. vector)
X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print(X.shape)

# Shuffle
keys = np.array(range(X.shape[0]))
print(keys[:10])

np.random.shuffle(keys)
print(keys[:10])

X = X[keys]
y = y[keys]

print(y[:15])