import numpy as np
from Activation import Sigmoid, ReLU, TanH

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
        if(activation != 2):
            self.W = np.random.randn(nodes, input_size) * np.sqrt(1. / input_size)
        elif(activation == 1 or activation == 3):
            self.W = np.random.randn(nodes, input_size) * np.sqrt(2. / input_size)
        else:
            self.W = np.random.uniform(-0.1, 0.1, (nodes, input_size))

        self.B = np.zeros(nodes)
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
        # dz is the derivative
        dz = self.activation_fn.derivative(self.output)
        # weighted_sum = np.dot(self.W, self.X_in) + self.B
        # dz = self.activation_fn.derivative(weighted_sum)
        
        if dz is None:
            raise ValueError("The derivative of the activation function returned None.")
        delta = output_gradient * dz

        # dw is the derivative of the weighted sum
        self.dw = np.dot(delta, self.X_in.T)
        # if output_gradient.ndim == 1:
        #     output_gradient = output_gradient[:, np.newaxis]
        # db is the derivative of the bias
        self.db = np.sum(delta, axis=1)
        self.W -= rate * self.dw
        self.B -= rate * self.db
        input_gradient = np.dot(self.W.T, delta)
     
        return input_gradient