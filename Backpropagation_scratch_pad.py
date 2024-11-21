# forward pass

x = [1.0, -2.0, 3.0] # inputs
w = [-3.0, -1.0, 2.0] # weights
b = 1.0 # bias

# Multiplying inputs by weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
#print(xw0, xw1, xw2)

# Adding weighted inputs and a bias, creating neuron's output
z = xw0 + xw1 + xw2 + b
#print(z)

# Apply ReLU activation function to neuron output
y = max(z, 0)
print(y)

# backward pass

# The derivative from the next layer
dvalue = 1 # made up value for demonstration purposes

# Derivative of ReLU and the chain rule
drelu_dz = dvalue * (1. if z>0 else 0.)
#print(drelu_dz)

# Partial derivative of the sum opperation, always 1, no matter the inputs, the chain rule
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1
drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db
#print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

# Partial derivatives of the multiplication, the chain rule
# partial derivative of f with respect to x equals y
# following this rule, the partial derivative of the first weighted input with respect to the input
# equals the weight (the other input of this function)
# then we apply the chain rule
dmul_dx0 = w[0] # partial derivative with respect to input
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0] # partial derivative with respect to weight
dmul_dw1 = x[1]
dmul_dw2 = x[2]

drelu_dx0 = drelu_dxw0 * dmul_dx0 # chain rule
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw2 = drelu_dxw2 * dmul_dw2

#print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)

# Gradient, partial derivatives combined into a vector
dx = [drelu_dx0, drelu_dx1, drelu_dx2] # gradient on input
dw = [drelu_dw0, drelu_dw1, drelu_dw2] # gradient on weight
db = drelu_db # gradient on bias, just 1 bias here

#print(w, b)

# Apply a fraction of the gradient to these values
w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
b += -0.001 * db

#print(w, b)

# Second forward pass
# Multiplying inputs by weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
#print(xw0, xw1, xw2)

# Adding weighted inputs and a bias, creating neuron's output
z = xw0 + xw1 + xw2 + b
#print(z)

# Apply ReLU activation function to neuron output
y = max(z, 0)
print(y)