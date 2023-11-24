from Network import Network
from Layer import Layer

class ANNBuilder:
    def build(nb_layers, list_nodes, list_functions, input_size):
        # nb_layers: Number of layers in the network
        # list_nodes: List containing the number of nodes for each layer
        # list_functions: List of activation functions for each layer
        # input_size: Size of the input layer

        # creates an instance of the network class
        ann = Network()
        # iterates over the number of layers
        for i in range(nb_layers):
            # iterates over the number of layers
            if i > 0:
                # set input size as number of nodes in previous layer
                input_size = list_nodes[i - 1]
            # creates a layer with specified number of nodes, input size and activation functions
            layer = Layer(list_nodes[i], input_size, list_functions[i])
            # append layer to the network
            ann.append(layer)
        return ann