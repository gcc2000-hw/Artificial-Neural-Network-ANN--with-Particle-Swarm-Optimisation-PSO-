import numpy as np
class Network:
    def __init__(self):
        self.layers = []
    def append(self, layer):
        self.layers.append(layer)
    def forward (self, data_in):
        out = data_in
        for layer in self.layers:
            out = layer.forward(out)
        return out
    def get_param(self):
        param = [np.concatenate([i.W.flatten(),i.B.flatten()]) for i in self.layers]
        return np.concatenate(param)
    def update_param(self,param):
        start_section = 0
        for i in self.layers:
            end = start_section + i.W.size 
            i.W = param[start_section:end].reshape(i.W.shape) 
            start_section = end
            end = start_section + i.B.size
            i.B = param[start_section:end].reshape(i.B.shape)
            start_section = end
    def get_dimension(self):
        return len(self.get_param())
    def backward(self, delta, rate):
        for layer in reversed(self.layers):
            delta = layer.backward(delta, rate)