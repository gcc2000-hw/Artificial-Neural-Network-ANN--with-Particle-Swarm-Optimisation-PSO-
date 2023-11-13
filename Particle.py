import numpy as np
class Particle:
    def __init__(self,func,optimP = "min"):
        self.func = func
        self.dim = func.get_dimension()
        self.position = np.random.randn(self.dim)
        self.velocity = np.random.randn(self.dim)
        self.pbestPos = self.position.copy()
        self.pFit = None
        self.fit = 0
        self.neighbours = []
        self.optimizationP = optimP

    def get_neighbours(self,neighbors):
        self.neighbours = neighbors

    def get_fit(self):
        self.fit = self.func(self.position)
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



    def update_velocity(self,alpha,beta,gamma,delta,lbest_pos,gbest_pos):
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        r3 = np.random.rand(self.dim)
        inertia_comp = alpha * self.velocity
        cog_comp = beta * (self.pbestPos - self.position)
        social_comp = gamma * (lbest_pos - self.position)
        global_comp = delta * (gbest_pos - self.position)
        self.velocity = inertia_comp + (r1*cog_comp) + (r2*social_comp) + (r3*global_comp)

    def update_position(self):
        self.position = np.add(self.position,self.velocity)

    # calculate loss and update l_best, p_best and g_best
    def get_loss():
        pass