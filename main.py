import numpy as np
from Activation import *
from Loss import *
from NetworkBuilder import ANNBuilder
import pandas as pd
from Activation import *
from GradientDescend import *
from Layer import *


UCI_auth_data = np.genfromtxt("data_banknote_authentication.txt", delimiter=",")
X, Y = UCI_auth_data[:, :4], UCI_auth_data[:, 4]

Y = Y.astype(int)

# Shuffling & train/test split
shuffle_idx = np.arange(Y.shape[0])
shuffle_rng = np.random.RandomState(123)
shuffle_rng.shuffle(shuffle_idx)
X, Y = X[shuffle_idx], Y[shuffle_idx]

ann= ANNBuilder.build(3,np.array([2,2,3]),[1,2,1])
loss, accuracy = mini_batch(ann, X, Y, 7, 0.0001, Mse, 196)
