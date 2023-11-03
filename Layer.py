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
        self.dw = np.zeros((nodes,input_size))
        self.db = np.zeros(nodes)

    def forward(self, X_in):
        self.X_in = X_in
        weighted_sum = np.dot(self.W, X_in)+self.B
        out = self.activation_fn.evaluation(self,weighted_sum)
        # out = Activation(self, np.dot(self.W, X_in)+self.B)
        return out








    # def backward(self, delta, rate):
    #     # dz is the derivative of the weighted sum
    #     dz = self.activation_fn.derivative(self.W * self.X_in) * delta
    #     # dw is the derivative of the
    #     dw = self.X_in * dz
    #     db = dz
    #     delta = self.W * dz
    #     self.W -= rate * dw
    #     self.B-= rate * db
    #     return delta