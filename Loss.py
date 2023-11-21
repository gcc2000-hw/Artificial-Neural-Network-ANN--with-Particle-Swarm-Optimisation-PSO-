import numpy as np
from Activation import *

class Loss:
    def __init__(self, expected, predicted):
        self.expected = expected
        self.predicted = predicted
    # def Evaluate(self):
    #     pass
    # def Derivate(self):
    #     pass

class Mse(Loss):
    def Evaluate(expected, predicted):
        return 1/2 * (np.square(expected - predicted))
    def Derivate(expected, predicted):
        return predicted - expected

class BinaryCrossEntropyLoss(Loss):
    def Evaluate(expected, predicted):
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        term0 = (1-expected) * np.log(1-predicted) 
        term1 = expected * np.log(predicted) 
        return -(term0 + term1)
    def Derivate(expected, predicted):
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return (predicted - expected) / (predicted * (1 - predicted))
    
class CrossEntropyLoss(Loss):
    def Evaluate(self):
        pass
        

class Hinge(Loss):
    def Evaluate(expected, predicted):
        return np.mean(np.maximum(0, 1-expected * predicted))
    def Derivate(expected, predicted):
        return np.where(expected*predicted<1,-expected,0)