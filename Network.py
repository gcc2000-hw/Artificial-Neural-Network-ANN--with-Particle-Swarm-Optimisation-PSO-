class Network:
    def __init__(self) -> None:
        self.layers = []
    def append(self, layer):
        self.layers.append(layer)
    def forward (self, data_in):
        out = data_in
        for layer in self.layers:
            out = layer.forward(out)
        return out
    def backpropagate(self, delta, rate):
        for layer in self.layers.reverse():
            delta = layer.backpropagate(delta, rate)