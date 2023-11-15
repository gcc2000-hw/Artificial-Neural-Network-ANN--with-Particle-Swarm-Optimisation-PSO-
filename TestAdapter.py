import numpy as np
class TestAdapter:
    def __init__(self):
        pass

    def get_param(self):
        # Return initial parameter for x
        return np.array([np.random.randn()])

    def set_param(self, param):
        # This example does not need to store parameters
        pass

    def evaluate(self, params):
        # Assuming the function is x^2
        x = params[0]
        return x**2