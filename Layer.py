import numpy as np
from Activation import Sigmoid, ReLU, TanH, Activation

activation_map = {
        1: Sigmoid,
        2: ReLU,
        3: TanH
    }
#2,1 
class Layer:
    def __init__(self, nodes, input_size, activation):
        self.nodes = nodes
        # self.X_in = input
        self.W = np.random.uniform(-1, 1, (nodes, input_size))
        self.B = np.random.uniform(-1, 1, nodes)
        self.activation_fn = activation_map[activation]
        self.output = 0
        self.dw = np.zeros((nodes,input_size))
        self.db = np.zeros(nodes)

    def forward(self, X_in):
        if X_in.ndim == 1:
            X_in = X_in.reshape(-1, 1)
        self.X_in = X_in
        weighted_sum = np.dot(self.W, X_in)+self.B[:, np.newaxis]
        out = self.activation_fn.evaluation(weighted_sum)
        self.output = out
        # out = Activation(self, np.dot(self.W, X_in)+self.B)
        return out
    
    def backward(self, output_gradient, rate):
        print("Output gradient shape:", output_gradient.shape)
        print("Input (X_in) shape:", self.X_in.shape)
        # dz is the derivative
        dz = self.activation_fn.derivative(self.output)
        # weighted_sum = np.dot(self.W, self.X_in) + self.B
        # dz = self.activation_fn.derivative(weighted_sum)
        
        delta = output_gradient * dz
        print("Delta shape:", delta.shape)
        if dz is None:
            raise ValueError("The derivative of the activation function returned None.")
        
        # dw is the derivative of the weighted sum
        dw = np.dot(delta, self.X_in.T)
        # db is the derivative of the bias
        db = np.sum(delta, axis=1)
        self.W -= rate * dw
        self.B -= rate * db
        input_gradient = np.dot(self.W.T, delta)
        return input_gradient