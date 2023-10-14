import numpy as np

class Activation:
    def __init__(self, weighted_sum):
        self.weighted_sum = weighted_sum

    def evaluation(weighted_sum):
        pass
    def derivative(weighted_sum):
        pass

class Sigmoid(Activation):
    def evaluation(self, weighted_sum):
        return 1/(1+np.exp(-weighted_sum))
    def derivative(self, weighted_sum):
       return self.evaluation(self, weighted_sum) * (1 - self.evaluation(self, weighted_sum))
        
class ReLU(Activation):
    def evaluation(self, weighted_sum):
        return weighted_sum * (weighted_sum > 0)
    def derivative(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    
class TanH(Activation):
    def evaluation(self, weighted_sum):
        return np.tanh(weighted_sum)
    def derivative(self, weighted_sum):
        return (1 - (self.evaluation(self,weighted_sum)**2))

print("1." ,Sigmoid.derivative(Sigmoid,np.array([5,7])))
print("2. " ,ReLU.derivative(ReLU,np.array([5,7])))
print("3. " ,TanH.derivative(TanH,np.array([5,7])))
print(ReLU.derivative(ReLU,np.array([1,-2,3,4])))
'''
References
1. https://dustinstansbury.github.io/theclevermachine/derivation-common-neural-network-activation-functions

'''