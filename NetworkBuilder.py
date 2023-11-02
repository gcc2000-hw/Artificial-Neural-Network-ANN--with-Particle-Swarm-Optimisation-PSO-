from Network import Network
from Layer import Layer

class ANNBuilder:
    def build(nb_layers, list_nodes, list_functions):
        ann = Network()
        input_size = 4
        for i in range(nb_layers):
            if i > 0:
                input_size = list_nodes[i - 1]
            layer = Layer(list_nodes[i], input_size, list_functions[i])
            ann.append(layer)
        return ann