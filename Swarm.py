import numpy as np
from Particle import Particle
class Swarm:
    def __init__(self,adapter,alpha,beta,gamma,delta,population_size,iterations,neighbor_size,optimP = "min",neighbor_strat="r"):
        self.particles = [Particle(adapter, optimP) for i in range(population_size)]
        self.adapter = adapter
        self.dim = len(adapter.get_param())
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.strat = neighbor_strat
        self.pop_size = population_size
        self.iteration = iterations
        self.neighbor_size = neighbor_size
        self.gbestPos = np.random.randn(self.dim)
        self.gFit = np.inf if optimP == "min" else -np.inf
        self.optimizationP = optimP

    def update_neighborhood(self):
        if self.strat == "r":
            self.random_neighbors()
        elif self.strat == "d":
            self.distance_neighbors()
        elif self.strat == "h":
            self.hybrid_neighbors()
    def random_neighbors(self):
        for p in self.particles:
            neighbors =  np.random.choice([i for i in self.particles if p != i],self.neighbor_size,replace = False)
            p.set_neighbours(neighbors)
    def distance_neighbors(self):
        pass
    def hybrid_neighbors(self):
        pass
    def distance_for_hybrid(self,particle):
        pass
    def get_gbest(self):
        for p in self.particles:
            match self.optimizationP:
                case "min" if p.fit < self.gFit:
                    self.gFit = p.fit
                    self.gbestPos = p.position.copy()
                case "max" if p.fit > self.gFit:
                    self.gFit = p.fit
                    self.gbestPos = p.position.copy()
            
    def optimize(self):
        for i in range(self.iteration):
            self.update_neighborhood()
            for p in self.particles:
                p.get_fit()
                p.is_lbest()
                for n in p.neighbours:
                    match self.optimizationP:
                        case "min" if n.fit < p.lFit:
                            p.lfit = n.fit
                            p.lbestPos = n.position.copy()
                        case "max" if n.fit > p.lFit:
                            p.lfit = n.fit
                            p.lbestPos = n.position.copy()
            self.get_gbest()
            for p in self.particles:
                p.update_velocity(self.alpha,self.beta,self.gamma,self.delta,self.gbestPos)
                p.update_position()
        
        

