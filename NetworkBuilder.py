from Network import Network
from Layer import Layer

class ANNBuilder:
    def build(nb_layers, list_nodes, list_functions):
        ann = Network()
        for i in range(nb_layers):
            layer = Layer(list_nodes[i], list_functions[i])
            ann.append(layer)
        return ann