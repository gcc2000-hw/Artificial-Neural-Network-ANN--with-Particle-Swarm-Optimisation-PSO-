import numpy as np
from Activation import *
from Loss import *
from NetworkBuilder import ANNBuilder
import pandas as pd
from Activation import *
from GradientDescend import *
from Layer import *
from Swarm import Swarm
from Adapter import adapter

data=pd.read_csv("data_banknote_authentication.txt", delimiter=",")
UCI_auth_data = np.genfromtxt("data_banknote_authentication.txt", delimiter=",")
Ycol = data.columns[-1];
X = data.drop(Ycol,axis=1).values
Y = data[Ycol].values
def evaluate_ann(params, ann, X, Y, loss_function):
    ann.update_param(params)
    predictions = ann.forward(X.T)
    losses = loss_function.Evaluate(Y, predictions)
    mean_loss = np.mean(losses)  # Aggregate the losses into a single value
    return mean_loss

# PSO params
alpha = 0.5    
beta = 0.8     
gamma = 0.9   
delta = 0.4  
population_size = 30
iterations = 100
neighborhood_size = 5
optimP = "min"  # Minimization problem
# Shuffling & train/test split
shuffle_idx = np.arange(Y.shape[0])
shuffle_rng = np.random.RandomState(123)
shuffle_rng.shuffle(shuffle_idx)
X, Y = X[shuffle_idx], Y[shuffle_idx]
initial_input = np.size(X[0])
ann = ANNBuilder.build(3, np.array([2, 2, 5]), np.array([1, 2, 1]), initial_input)
initial_params = ann.get_param()
Adapter = adapter(lambda params: evaluate_ann(params,ann,X,Y,BinaryCrossEntropyLoss),initial_params)
swarm = Swarm(Adapter, alpha, beta, gamma, delta,population_size, iterations, neighborhood_size, optimP,"r")

# Run the optimization
swarm.optimize() 
# Print the best solution found
print(f"Best solution found: x = {swarm.gbestPos}, f(x) = {swarm.gFit}")
# ann= ANNBuilder.build(3,np.array([2,2,3]),np.array([1,2,1]), initial_input)
# loss, accuracy = mini_batch(ann, X, Y, 7, 0.0001, Mse, 196)
