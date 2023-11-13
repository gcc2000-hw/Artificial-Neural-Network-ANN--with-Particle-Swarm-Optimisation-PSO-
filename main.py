import numpy as np
from Activation import *
from Loss import *
from NetworkBuilder import ANNBuilder
import pandas as pd
from Activation import *
from GradientDescend import *
from Layer import *

data=pd.read_csv("data_banknote_authentication.txt", delimiter=",")
UCI_auth_data = np.genfromtxt("data_banknote_authentication.txt", delimiter=",")
Ycol = data.columns[-1];
X = data.drop(Ycol,axis=1).values
Y = data[Ycol].values

# Shuffling & train/test split
shuffle_idx = np.arange(Y.shape[0])
shuffle_rng = np.random.RandomState(123)
shuffle_rng.shuffle(shuffle_idx)
X, Y = X[shuffle_idx], Y[shuffle_idx]
initial_input = np.size(X[0])
ann= ANNBuilder.build(3,np.array([2,2,3]),np.array([1,2,1]), initial_input)
loss, accuracy = mini_batch(ann, X, Y, 7, 0.0001, Mse, 196)
