import numpy as np
from Activation import *
from Loss import *
from NetworkBuilder import ANNBuilder
import pandas as pd
from Activation import *
from GradientDescend import *


UCI_auth_data = np.genfromtxt("data_banknote_authentication.txt", delimiter=",")
X, Y = UCI_auth_data[:, :4], UCI_auth_data[:, 4]

Y = Y.astype(int)

# Shuffling & train/test split
shuffle_idx = np.arange(y.shape[0])
shuffle_rng = np.random.RandomState(123)
shuffle_rng.shuffle(shuffle_idx)
X, Y = X[shuffle_idx], Y[shuffle_idx]
print(X,Y)

print(X.shape, " " , Y.shape)


ann= ANNBuilder.build(3,np.array([2,3,3]),[Sigmoid(Activation),Sigmoid(Activation),Sigmoid(Activation)])
loss, accuracy = mini_batch(ann, UCI_auth_data, np.array([0,1]), 7, 0.0001, Mse, 196)