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
from sklearn.model_selection import train_test_split 

data=pd.read_csv("data_banknote_authentication.txt", delimiter=",")
UCI_auth_data = np.genfromtxt("data_banknote_authentication.txt", delimiter=",")
Ycol = data.columns[-1];
X = data.drop(Ycol,axis=1).values
Y = data[Ycol].values
if(isinstance(Y[0],str)):
    unique_labels, Y_mapped = np.unique(Y,return_inverse=True)
    D = {label : i for i,label in enumerate(unique_labels)}
    Y = Y_mapped
def evaluate_ann(params, ann, X, Y, loss_function):
    ann.update_param(params)
    predictions = ann.forward(X.T)
    losses = loss_function.Evaluate(Y, predictions)
    mean_loss = np.mean(losses)  # Aggregate the losses into a single value
    return mean_loss

# PSO params
alpha = 0.2
beta = 1.0 
gamma = 2.0   
delta = 1.0
population_size = 100
iterations = 100
neighborhood_size = 9
optimP = "min"  # Minimization problem
# Shuffling & train/test split
shuffle_idx = np.arange(Y.shape[0])
shuffle_rng = np.random.RandomState(123)
shuffle_rng.shuffle(shuffle_idx)
X, Y = X[shuffle_idx], Y[shuffle_idx]
initial_input = np.size(X[0])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
ann = ANNBuilder.build(3, np.array([2, 2, 5]), np.array([1, 2, 1]), initial_input)
initial_params = ann.get_param()
Adapter = adapter(lambda params: evaluate_ann(params,ann,X_train,Y_train,BinaryCrossEntropyLoss),initial_params)
swarm = Swarm(Adapter, alpha, beta, gamma, delta,population_size, iterations, neighborhood_size, optimP,"h")

# Run the optimization
swarm.optimize() 

ann.update_param(swarm.gbestPos)

prediction = ann.forward(X_test.T)
predicted_labels = np.round(prediction)
accuracy = np.mean(predicted_labels == Y_test)
print(f"Accuracy on test set: {accuracy}")

# print(f"Best solution found: x = {swarm.gbestPos}, f(x) = {swarm.gFit}")
# ann= ANNBuilder.build(3,np.array([2,2,3]),np.array([1,2,1]), initial_input)
# loss, accuracy = mini_batch(ann, X, Y, 7, 0.0001, Mse, 196)
