#PSO adapter that passes any function to be opimized by PSO
class adapter: 
    # it intializes and save the fitness calculation function of the function to be optimized
    # it also stores the the parameters of the function 
    def __init__(self,func,param):
        self.func = func
        self.param = param
    #getter for the parameters
    def get_param(self):
        return self.param
    #setter for the parameters
    def set_param(self,param_new):
        self.param = param_new
    #Passes inputs into the fitness calculation function for the fitness to be calculated
    def evaluate(self,inputs):
        return self.func(inputs)
        