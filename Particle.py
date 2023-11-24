import numpy as np
class Particle:
    def __init__(self,adapter,optimP):
        self.adapter = adapter
        self.dim = len(adapter.get_param())
        self.position = np.random.randn(self.dim)
        self.velocity = np.random.randn(self.dim)
        self.pbestPos = self.position.copy()
        self.pFit = None
        self.fit = 0
        self.neighbours = []
        self.optimizationP = optimP
        self.lbestPos = self.position.copy()
        self.lFit = -np.inf if optimP == "max" else self.fit
        self.position_history = []
    #Set the neighbors
    def set_neighbours(self,neighbors):
        self.neighbours = neighbors

    # Update pbest position, pbest 
    def get_fit(self):
        self.adapter.set_param(self.position)
        self.fit = self.adapter.evaluate(self.position)
        if self.pFit is None:
            self.pFit = self.fit
        else:
            match self.optimizationP:
                case "min" if self.fit < self.pFit:
                    self.pFit = self.fit
                    self.pbestPos = self.position.copy()
                case "max" if self.fit > self.pFit:
                    self.pFit = self.fit
                    self.pbestPos = self.position.copy()


    #Update velocity
    def update_velocity(self,alpha,beta,gamma,delta,gbestPos):
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        r3 = np.random.rand(self.dim)
        inertia_comp = alpha * self.velocity
        cog_comp = beta * (self.pbestPos - self.position)
        social_comp = gamma * (self.lbestPos - self.position)
        global_comp = delta * (gbestPos - self.position)
        self.velocity = inertia_comp + (r1*cog_comp) + (r2*social_comp) + (r3*global_comp)
    #Move particle
    def update_position(self):
        self.position = self.position + self.velocity
        self.position_history.append(self.position.copy())
    
    def is_lbest(self):
        match self.optimizationP:
            case "min" if self.fit < self.lFit:
                self.lFit = self.fit
                self.lbestPos = self.position.copy()
            case "max" if self.fit > self.lFit:
                self.lFit = self.fit
                self.lbestPos = self.position.copy()

