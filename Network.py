import numpy as np

class Network:
    def __init__(self):
        self.layers = []

    def append(self, layer):
        #  initializes an empty list of layers
        self.layers.append(layer)

    # function for forward propagation
    def forward (self, data_in):
        # output of each layer is the input of next
        out = data_in
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def get_param(self):
        #  flattens and concatenates weights and biases for each layer
        param = [np.concatenate([i.W.flatten(),i.B.flatten()]) for i in self.layers]
        return np.concatenate(param)
    
    def update_param(self,param):
        start_section = 0
        # iterates through each layer and updates the parameters
        for i in self.layers:
            # calculates the end index of the layers weights
            end = start_section + i.W.size 
            # updates weights
            i.W = param[start_section:end].reshape(i.W.shape) 
            # update start section for biases
            start_section = end
            # gets end index for biases
            end = start_section + i.B.size
            # updates
            i.B = param[start_section:end].reshape(i.B.shape)
            start_section = end

    def get_dimension(self):
        return len(self.get_param())
    