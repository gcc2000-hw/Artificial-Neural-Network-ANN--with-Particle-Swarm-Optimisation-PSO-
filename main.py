import numpy as np
import matplotlib.pyplot as plt
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


def plot_accuracy(train_accuracy_list, val_accuracy_list):
    epochs = [i[0] for i in train_accuracy_list]
    accuracies = [i[1] for i in train_accuracy_list]
    val_accuracies = [i[1] for i in val_accuracy_list]

    plt.plot(epochs, accuracies, label='Training Accuracy', color='blue')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accurcy")
    plt.title("Training and Validation Accuracy over Epochs")
    plt.legend() 
    plt.show()

def plot_loss(loss_list, val_loss_list):
    epochs = [i[0] for i in loss_list]
    losses = [i[1] for i in loss_list]
    val_losses = [i[1] for i in val_loss_list]

    plt.plot(epochs, losses, label='Loss', color='red')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.show()

def train_pso(ann, X_train, Y_train, population_size = 100, iterations = 100, alpha = 0.2, beta = 1.0, gamma = 2.0,  delta = 1.0, neighborhood_size = 9, optimP = "min", informant_type = "r"):
    # Shuffling & train/test split
    # initial_input = np.size(X[0])
    # ann = ANNBuilder.build(3, np.array([1, 3, 3]), np.array([3, 1, 2]), initial_input)
    initial_params = ann.get_param()
    Adapter = adapter(lambda params: evaluate_ann(params,ann,X_train,Y_train,BinaryCrossEntropyLoss),initial_params)
    swarm = Swarm(Adapter, alpha, beta, gamma, delta,population_size, iterations, neighborhood_size, optimP, informant_type)

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

def train_gradient_descent(ann, X_train, Y_train, X_val, Y_val, method='mini_batch', epochs=7, rate=0.01, loss = BinaryCrossEntropyLoss,  batch_size=32):

    if (method == 'mini_batch'):
        loss, accuracy, accuracy_list, loss_list, val_accuracy_list, val_loss_list = mini_batch(ann,X_train, Y_train, X_val, Y_val, epochs, rate, loss, batch_size)
        plot_accuracy(accuracy_list, val_accuracy_list)
        plot_loss(loss_list, val_loss_list)
    elif (method == 'dgd'):
        loss, accuracy, accuracy_list, loss_list, val_accuracy_list, val_loss_list = dgd(ann, X_train, Y_train, X_val, Y_val, epochs, rate, loss)
        plot_accuracy(accuracy_list, val_accuracy_list)
        plot_loss(loss_list, val_loss_list)
    elif (method == 'sgd'):
        loss, accuracy, accuracy_list, loss_list, val_accuracy_list, val_loss_list = sgd(ann, X_train, Y_train, X_val, Y_val, epochs, rate, loss)
        plot_accuracy(accuracy_list, val_accuracy_list)
        plot_loss(loss_list, val_loss_list)

    return loss, accuracy, accuracy_list, loss_list, val_accuracy_list, val_loss_list


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

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# print(f"Best solution found: x = {swarm.gbestPos}, f(x) = {swarm.gFit}")
initial_input = np.size(X[0])
ann= ANNBuilder.build(3,np.array([2,2,1]),np.array([1,1,1]), initial_input)
# test_gradient_descent(ann, X_test, Y_test) 
# train_gradient_descent(X_train, Y_train, method='mini_batch', epochs=7, rate=0.01, loss = Mse,  batch_size=32)
# loss, accuracy = mini_batch(ann, X, Y, 7, 0.0001, Mse, 196)
# print(loss)
# print(accuracy)
# Train the network
loss, accuracy, accuracy_list, loss_list, val_accuracy_list, val_loss_list = train_gradient_descent(ann,X_train, Y_train, X_val, Y_val, method='mini_batch', epochs=50, rate=0.0001, loss=BinaryCrossEntropyLoss, batch_size=32)
print("Training Loss:", loss)
print("Training Accuracy:", accuracy)
# Test the network
test_accuracy = test_gradient_descent(ann, X_test, Y_test)
print("Test Accuracy:", test_accuracy)
