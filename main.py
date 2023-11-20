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
Ycol = data.columns[-1]
X = data.drop(Ycol,axis=1).values
Y = data[Ycol].values
if(isinstance(Y[0],str)):
    unique_labels, Y_mapped = np.unique(Y,return_inverse=True)
    D = {label : i for i,label in enumerate(unique_labels)}
    Y = Y_mapped

def data_type():
    pass
def evaluate_ann(params, ann, X, Y, loss_function):
    ann.update_param(params)
    predictions = ann.forward(X.T)
    losses = loss_function.Evaluate(Y, predictions)
    mean_loss = np.mean(losses)  # Aggregate the losses into a single value
    return mean_loss

def train_pso(ann, X_train, Y_train, population_size = 100, iterations = 100, alpha = 0.2, beta = 1.0, gamma = 2.0,  delta = 1.0, neighborhood_size = 9, optimP = "min"):
    # Shuffling & train/test split
    initial_input = np.size(X[0])
    # ann = ANNBuilder.build(3, np.array([1, 3, 3]), np.array([3, 1, 2]), initial_input)
    initial_params = ann.get_param()
    Adapter = adapter(lambda params: evaluate_ann(params,ann,X_train,Y_train,BinaryCrossEntropyLoss),initial_params)
    swarm = Swarm(Adapter, alpha, beta, gamma, delta,population_size, iterations, neighborhood_size, optimP,"h")

    # Run the optimization
    swarm.optimize() 
    ann.update_param(swarm.gbestPos)
    return ann

def test_pso(ann, X_test, Y_test):
    prediction = ann.forward(X_test.T)
    predicted_labels = np.round(prediction)
    accuracy = np.mean(predicted_labels == Y_test)
    print(f"Accuracy on test set: {accuracy}")
    return accuracy

def train_gradient_descent(ann, X_train, Y_train, method='mini_batch', epochs=7, rate=0.01, loss = BinaryCrossEntropyLoss,  batch_size=32):
    initial_input = np.size(X[0])
    # ann = ANNBuilder.build(3, [1, 3, 3], [3, 1, 2], initial_input)
    if (method == 'mini_batch'):
        loss, accuracy = mini_batch(ann, X_train, Y_train, epochs, rate, loss, batch_size)
    elif (method == 'dgd'):
        loss, accuracy = dgd(ann, X_train, Y_train, epochs, rate, loss)
    elif (method == 'sgd'):
        loss, accuracy = sgd(ann, X_train, Y_train, epochs, rate, loss)
    return loss, accuracy

def test_gradient_descent(ann, X_test, Y_test):
    # Evaluate the model
    predictions = ann.forward(X_test.T)
    predicted_labels = np.round(predictions)
    accuracy = np.mean(predicted_labels == Y_test)
    print(f"Accuracy on test set: {accuracy}")

    return accuracy

shuffle_idx = np.arange(Y.shape[0])
shuffle_rng = np.random.RandomState(123)
shuffle_rng.shuffle(shuffle_idx)
X, Y = X[shuffle_idx], Y[shuffle_idx]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# print(f"Best solution found: x = {swarm.gbestPos}, f(x) = {swarm.gFit}")
initial_input = np.size(X[0])
ann= ANNBuilder.build(3,np.array([2,2,3]),np.array([1,1,1]), initial_input)
# test_gradient_descent(ann, X_test, Y_test) 
# train_gradient_descent(X_train, Y_train, method='mini_batch', epochs=7, rate=0.01, loss = Mse,  batch_size=32)
# loss, accuracy = mini_batch(ann, X, Y, 7, 0.0001, Mse, 196)
# print(loss)
# print(accuracy)
# Train the network
loss, accuracy = train_gradient_descent(ann,X_train, Y_train, method='dgd', epochs=50, rate=0.0001, loss=Mse)
print("Training Loss:", loss)
print("Training Accuracy:", accuracy)
# Test the network
test_accuracy = test_gradient_descent(ann, X_test, Y_test)
print("Test Accuracy:", test_accuracy)
