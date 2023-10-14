import numpy as np
from Activation import *
activation_map = {
        1: Sigmoid,
        2: ReLU,
        3: TanH
    }
#2,1 
class Layer:
    def __init__(self, nodes, activation):
        print(nodes,  "  fjjfjf")
        self.nodes = nodes
        self.X_in = input
        self.W = np.random.randn(nodes)
        print(self.W , " : W")
        self.B = 1
        self.activation_fn = activation_map[activation]

    def forward(self, X_in):
        self.X_in = X_in
        out = self.activation_fn.evaluation(np.dot(self.W, X_in)+self.B)
        return out
    def backward(self, delta, rate):
        # dz is the derivative of the weighted sum
        dz = self.activation_fn.derivative(self.W * self.X_in) * delta
        # dw is the derivative of the
        dw = self.X_in * dz
        db = dz
        delta = self.W * dz
        self.W -= rate * dw
        self.B-= rate * db
        return delta