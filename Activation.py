import numpy as np

class Activation:
    def __init__(self, weighted_sum):
        self.weighted_sum = weighted_sum

    def evaluation(weighted_sum):
        pass
    def derivative(weighted_sum):
        pass

class Sigmoid(Activation):
    def evaluation(weighted_sum):
        return 1/(1+np.exp(-weighted_sum))
    def derivative(weighted_sum):
       sigmoid_output = 1/(1+np.exp(-weighted_sum))
       return sigmoid_output * (1 - sigmoid_output)
        
class ReLU(Activation):
    def evaluation(weighted_sum):
        # return weighted_sum * (weighted_sum > 0)
        return np.maximum(0, weighted_sum)
    def derivative(weighted_sum):
        return np.where(weighted_sum > 0, 1, 0)
        # x[x<=0] = 0
        # x[x>0] = 1
        # return x
    
class TanH(Activation):
    def evaluation(weighted_sum):
        return np.tanh(weighted_sum)
    def derivative(weighted_sum):
        return 1 - np.tanh(weighted_sum)**2


class Softmax(Activation):
    def evaluation(weighted_sum):
        e = np.exp(weighted_sum - np.max(weighted_sum))
        return e/np.sum(e,axis=0)
    def derivative(weighted_sum):
        pass
    
class Linear(Activation):
    def evaluation(weighted_sum):
        return weighted_sum
    def derivative(weighted_sum):
        return 1.0

'''
References
1. https://dustinstansbury.github.io/theclevermachine/derivation-common-neural-network-activation-functions

'''