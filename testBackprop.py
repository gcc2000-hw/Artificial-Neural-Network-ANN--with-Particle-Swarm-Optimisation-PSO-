import numpy as np
import matplotlib.pyplot as plt
from Activation import *
from Loss import *
from NetworkBuilder import ANNBuilder
import pandas as pd
from Activation import *
from GradientDescend import *
from Layer import *
from Swarm import Swarm
from Adapter import adapter
from sklearn.model_selection import train_test_split 

# Assuming your ANNBuilder and Layer classes are already defined as shown previously
# and initial_input is the size of the input layer

# Build the network
# Prepare a test input and target
test_input = np.array([[0.5], [-0.2]])  # Example input, reshape as necessary
test_target = np.array([[1]])            # Example target for binary classification

# Ensure the input is in the correct shape for the network
initial_input = test_input.shape[0]  # This should match the input layer size
ann = ANNBuilder.build(3, np.array([2, 2, 1]), np.array([1, 1, 1]), initial_input)

# Define a simple MSE loss function for testing
def mse_loss(output, target):
    return ((output - target) ** 2).mean()

def mse_loss_derivative(output, target):
    return 2 * (output - target) / output.size

# Define a function for gradient checking
def check_gradients(network, sample_input, sample_target, epsilon=1e-5, rate=0.01):
    # Forward and backward passes
    output = network.forward(sample_input)
    loss_grad = Mse.Derivate(sample_target, output)
    network.backward(loss_grad, rate)

    # Store the computed gradients
    computed_gradients = [(layer.dw, layer.db) for layer in network.layers]

    # Numerical gradient checking
    for idx, layer in enumerate(network.layers):
        layer_numerical_gradient_w = np.zeros_like(layer.W)
        layer_numerical_gradient_b = np.zeros_like(layer.B)

        # Checking weight gradients
        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                original_value = layer.W[i, j]
                layer.W[i, j] = original_value + epsilon
                loss_plus_epsilon = Mse.Evaluate(network.forward(sample_input), sample_target)
                layer.W[i, j] = original_value - epsilon
                loss_minus_epsilon = Mse.Evaluate(network.forward(sample_input), sample_target)
                layer_numerical_gradient_w[i, j] = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)
                layer.W[i, j] = original_value

        # Checking bias gradients
        for i in range(layer.B.shape[0]):
            original_value = layer.B[i]
            layer.B[i] = original_value + epsilon
            loss_plus_epsilon = Mse.Evaluate(network.forward(sample_input), sample_target)
            layer.B[i] = original_value - epsilon
            loss_minus_epsilon = Mse.Evaluate(network.forward(sample_input), sample_target)
            layer_numerical_gradient_b[i] = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)
            layer.B[i] = original_value

        # Output the comparison results
        print(f"Layer {idx + 1}")
        print(f"Computed weight gradients:\n{computed_gradients[idx][0]}")
        print(f"Numerical weight gradients:\n{layer_numerical_gradient_w}")
        print(f"Computed bias gradients:\n{computed_gradients[idx][1]}")
        print(f"Numerical bias gradients:\n{layer_numerical_gradient_b}")
        print()

# Prepare a test sample and target from your dataset
test_input = np.array([[0.5, -0.2]]).T  # Assuming 2 features in the input
test_target = np.array([[1]])  # Assuming binary classification

# Run gradient checking
check_gradients(ann, test_input, test_target)
