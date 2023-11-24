import numpy as np
#generic activation class
class Activation:
    def __init__(self, weighted_sum):
        self.weighted_sum = weighted_sum
#generic method to evalutate the activation function with the weighted sum
    def evaluation(weighted_sum):
        pass
#generic method to evaluate the derivate of the activation function with weighted sum
    def derivative(weighted_sum):
        pass

#Class for the sigmoid activation function (subclass of the generic activation class)
class Sigmoid(Activation):
    #Evaluates the sigmoid function A(x) = 1/(1 + e-x)
    def evaluation(weighted_sum):
        return 1/(1+np.exp(-weighted_sum))
    #Evaluates the derivative of the sigmoid function A'(x) = A(x) * (1-A(x))
    def derivative(weighted_sum):
       sigmoid_output = 1/(1+np.exp(-weighted_sum))
       return sigmoid_output * (1 - sigmoid_output)
    
#Class for the ReLU activation function (subclass of the generic activation class)
class ReLU(Activation):
    #Evaluates the ReLU function A(x)=max(0,x)
    def evaluation(weighted_sum):
        return np.maximum(0, weighted_sum)
    #Evaluates the derivative of the ReLU function A'(x) = 0 if x < 0 else 1
    def derivative(weighted_sum):
        return np.where(weighted_sum > 0, 1, 0)
#Class for the TanH activation function (subclass of the generic activation class)
class TanH(Activation):
    #Evaluates the TanH function A(x)=tanh(x)
    def evaluation(weighted_sum):
        return np.tanh(weighted_sum)
    #Evaluates the derivative of the TanH function A'(x) = 1 - tanh(x)^2
    def derivative(weighted_sum):
        return 1 - np.tanh(weighted_sum)**2

#Class for the TanH activation function (subclass of the generic activation class) Used to get the probability distribution of multiclass predictions
class Softmax(Activation):
    #Evaluates the Softmax function A(x)= e^x/sigma(e^x)
    def evaluation(weighted_sum):
        #Normalization step, Get the exponential of each feature. By substracting the maximum weighted sum value from all of the features
        #Gives us the largest exponent value as 0 which avoid overflow
        #We set the dimensions as the same by keepdims = True
        # Here, by taking the max, we reduce the dimentionality of the weighted sum to say n, by using keepdims =True we make sure it says 2D ie 2x1
        # Thereby avoiding dimention mismatch 
        e = np.exp(weighted_sum - np.max(weighted_sum, axis = 1, keepdims=True))
        return e/np.sum(e,axis=1, keepdims=True)
    # We dont define the gradient as we use softmax on the final layer for PSO 
    def derivative(weighted_sum):
        pass
#Class for the Linear activation function (subclass of the generic activation class)
class Linear(Activation):
    #Evaluates the Linear function A(x)= x
    def evaluation(weighted_sum):
        return weighted_sum
    #Evaluates the derivative of the linear function A'(x) = 1
    def derivative(weighted_sum):
        return 1.0

'''
References
1. https://dustinstansbury.github.io/theclevermachine/derivation-common-neural-network-activation-functions

'''