import numpy as np
from Activation import Activation

class Layer:
    def __init__(self, nodes, activation):
        self.nodes = nodes
        self.X_in = input
        self.W = [np.random()]
        self.B = 1
        self.activation = activation

    def forward(self, X_in):
        self.X_in = X_in
        out = Activation.evaluation(self.W*X_in+self.B)
        return out
    def backward(self, delta, rate):

        # dz is the derivative of the weighted sum
        dz = Activation.derivative(self.W * self.X_in) * delta
        # dw is the derivative of the
        dw = self.X_in * dz
        db = dz
        delta = self.W * dz
        self.W -= rate * dw
        self.B-= rate * db
        return delta