import numpy as np
from Activation import *
#generic class for loss
class Loss:
    # initialize the expected label and predicted label
    def __init__(self, expected, predicted):
        self.expected = expected
        self.predicted = predicted

# Class for Mse loss (subclass of the generic loss function)
class Mse(Loss):
    #function to evaluate the Mean square error loss where Loss(ex, pe) = 1/2 (ex - pe)^2 (for a single loss) we dont take mean
    #as we dont calculate the aggregate loss
    def Evaluate(expected, predicted):
        return 1/2 * (np.square(expected - predicted))
    
# class to evaluate the binary cross entropy loss
class BinaryCrossEntropyLoss(Loss):
    # function to evaluate binary cross entropy
    def Evaluate(expected, predicted):
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        term0 = (1-expected) * np.log(1-predicted)
        term1 = expected * np. log(predicted)
        return -(term0 + term1)

# class to evaluate the cross entropy loss
class CrossEntropyLoss(Loss):
    # function to evaluate cross entropy
    def Evaluate(expected,predicted):
        if predicted.shape != expected.shape:
            predicted = predicted.T
        epsilon = 1e-12
        predicted = np.clip(predicted, epsilon, 1. - epsilon)
        ce_loss = -np.sum(expected * np.log(predicted)) / expected.shape[0]
        return ce_loss

# class to evaluate the hinge loss
class Hinge(Loss):
    def Evaluate(expected, predicted):
        return np.mean(np.maximum(0, 1-expected * predicted))
