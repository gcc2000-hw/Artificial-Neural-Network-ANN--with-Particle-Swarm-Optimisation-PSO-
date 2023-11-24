import numpy as np
from Activation import Sigmoid, ReLU, TanH, Softmax
#A map that maps all the activations to an index
activation_map = {
        1: Sigmoid,
        2: ReLU,
        3: TanH,
        4: Softmax
    } 
#
#A class that defines the layers of the ANN
class Layer:
    #we recieve the number of nodes in the layer, size of the input being passed to the layer, activation for the layer
    def __init__(self, nodes, input_size, activation):
        # we check the activation and initialize weights accordingly 
        #if its sigmoid or tanH we use xavier initialization and for ReLU we use 'He' initialization
        #for any other we generate a random distribution between -0.1 to 0.1
        #From Kumar, S.K., 2017. On weight initialization in deep neural networks. arXiv preprint arXiv:1704.08863.
        self.nodes = nodes
        # self.X_in = input
        if(activation != 2):
            self.W = np.random.randn(nodes, input_size) * np.sqrt(1. / input_size)
        elif(activation == 1 or activation == 3):
            self.W = np.random.randn(nodes, input_size) * np.sqrt(2. / input_size)
        else:
            self.W = np.random.uniform(-0.1, 0.1, (nodes, input_size))
        # initialize bias as zero
        self.B = np.zeros(nodes)
        # obtain the activation function from the map
        self.activation_fn = activation_map[activation]
        # (for back prop) save the output of forward
        self.output = 0
        # (for back prop) save the derivative of the activation function
        self.dw = np.zeros((nodes,input_size))
        self.db = np.zeros(nodes)

#Function to run forward on the layer
    def forward(self, X_in):
        #if the dimention of input is n, , reshape it to 2D (nx1)
        if X_in.ndim == 1:
            X_in = X_in.reshape(-1, 1)
        #if the weights dimention is a mismatch with the input, transpose the input
        if self.W.shape[1] != X_in.shape[0]:
            X_in = X_in.T
        self.X_in = X_in
        #calculate the weighted sum
        weighted_sum = np.dot(self.W, X_in)+self.B[:, np.newaxis]
        #pass the weighted sum to the activation for the layer and get the output
        out = self.activation_fn.evaluation(weighted_sum)
        #(for back prop) store the output
        self.output = out
        #return the value of forward (prediction)
        return out