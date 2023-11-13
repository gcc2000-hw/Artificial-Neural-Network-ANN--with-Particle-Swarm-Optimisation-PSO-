class adapter: 
    def __init__(self,func,param):
        self.func = func
        self.param = param
    def get_param(self):
        return self.param
    def set_param(self,param_new):
        self.param = param_new
    def evaluate(self,inputs):
        return self.func(inputs)
        