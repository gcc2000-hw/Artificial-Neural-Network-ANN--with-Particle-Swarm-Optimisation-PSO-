import numpy as np
from Activation import *

class Loss:
    def __init__(self, expected, predicted):
        self.expected = expected
        self.predicted = predicted
    def Evaluate(self):
        pass
    def Derivate(self):
        pass

class Mse(Loss):
    def Evaluate(expected, predicted):
        return 1/2 * (np.square(expected - predicted))
    def Derivate(expected, predicted):
        return expected - predicted

class BinaryCrossEntropyLoss(Loss):
    def Evaluate(expected, predicted):
        term0 = (1-expected) * np.log(1-expected + 1e-7) 
        term1 = predicted * np.log(expected + 1e-7) 
        return -(term0 + term1)
    def Derivate(expected, predicted):
        return predicted/expected + (1 - predicted)/(1-expected)
    
class CrossEntropyLoss(Loss):
    def Evaluate(self):
        pass
        

class Hinge(Loss):
    def Evaluate(expected, predicted):
        return np.mean(np.maximum(0, 1-expected * predicted))
    def Derivate(expected, predicted):
        return np.where(expected*predicted<1,-expected,0)