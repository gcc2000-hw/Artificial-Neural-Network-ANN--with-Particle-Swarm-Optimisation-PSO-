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
                # for minimizaiton problem
                # if current fitness is better than the recorded fitness then update personal fitness
                case "min" if self.fit < self.pFit:
                    self.pFit = self.fit
                    self.pbestPos = self.position.copy()
                # for maximization problem
                # if current fitness is better than the recorded fitness then update personal fitness
                case "max" if self.fit > self.pFit:
                    self.pFit = self.fit
                    self.pbestPos = self.position.copy()

    #Update velocity
    def update_velocity(self,alpha,beta,gamma,delta,gbestPos):
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        r3 = np.random.rand(self.dim)
        #  alpha
        inertia_comp = alpha * self.velocity
        # beta
        cog_comp = beta * (self.pbestPos - self.position)
        # gamma
        social_comp = gamma * (self.lbestPos - self.position)
        # delta
        global_comp = delta * (gbestPos - self.position)
        #  update velocity based on weighted sum
        self.velocity = inertia_comp + (r1*cog_comp) + (r2*social_comp) + (r3*global_comp)
    
    #Move particle
    def update_position(self):
        # update position by adding velocity
        self.position = self.position + self.velocity
        # add position to history
        self.position_history.append(self.position.copy())
    
    # method to update the local best position and fitness
    def is_lbest(self):
        match self.optimizationP:
            # for minimizaiton problem
            # if current fitness is better than the recorded fitness then update local fitness
            case "min" if self.fit < self.lFit:
                self.lFit = self.fit
                self.lbestPos = self.position.copy()
            # for maximization problem
            # if current fitness is better than the recorded fitness then update local fitness
            case "max" if self.fit > self.lFit:
                self.lFit = self.fit
                self.lbestPos = self.position.copy()

