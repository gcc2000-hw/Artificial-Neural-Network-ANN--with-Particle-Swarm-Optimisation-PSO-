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
    def Evaluate(self):
        return 1/2(np.square(self.expected - self.predicted))
    def Derivate(self):
        return self.expected - self.predicted

class BinaryCrossEntropyLoss(Loss):
    def Evaluate(self):
        term0 = (1-self.expected) * np.log(1-self.expected + 1e-7) 
        term1 = self.predicted * np.log(self.expected + 1e-7) 
        return -(term0 + term1)
    def Derivate(self):
        return self.predicted/self.expected + (1 - self.predicted)/(1-self.expected)

class Hinge(Loss):
    def Evaluate(self):
        return np.mean(np.maximum(0, 1- self.expected * self.predicted))
    def Derivate(self):
        return np.where(self.expected*self.predicted<1,-self.expected,0)