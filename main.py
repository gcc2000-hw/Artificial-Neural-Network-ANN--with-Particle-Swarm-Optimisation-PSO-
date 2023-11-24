import numpy as np
import matplotlib.pyplot as plt
import time
from Activation import *
from Loss import *
from NetworkBuilder import ANNBuilder
import pandas as pd
from Activation import *
from Layer import *
from Swarm import Swarm
from Adapter import adapter
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder

# from sklearn.preprocessing import OneHotEncoder 

# name, activation function, loss, last layer node if none let them choose
def d_type(Y):
    unique_labels = np.unique(Y, axis=0)
    if len(unique_labels) == 2:
        return ["binary_class", None, BinaryCrossEntropyLoss, 1]
    elif len(unique_labels) > 2 and isinstance(unique_labels[0], (np.ndarray)):
        print(len(unique_labels))
        return ["multi_class", Softmax, CrossEntropyLoss, len(unique_labels)]
    elif len(unique_labels) > 2 and isinstance(unique_labels[0], (float,np.float32, np.float64)):
        return ["logistic", Linear, Mse, 1]
    else:

        return [None, None, None, None]
def one_hot_encode(Y):
    label_encoder = LabelEncoder()
    Y_integer_encoded = label_encoder.fit_transform(Y)
    number_of_classes = len(np.unique(Y_integer_encoded))
    one_hot = np.zeros((len(Y_integer_encoded), number_of_classes))
    one_hot[np.arange(len(Y_integer_encoded)), Y_integer_encoded] = 1
    return one_hot
data=pd.read_csv("iris.data", delimiter=",")
UCI_auth_data = np.genfromtxt("data_banknote_authentication.txt", delimiter=",")
Ycol = data.columns[-1]
X = data.drop(Ycol,axis=1).values
Y = data[Ycol].values
if(isinstance(Y[0],str)):
    Y = one_hot_encode(Y)
    unique_labels = np.unique(Y, axis=0)
    print(type(unique_labels[0]))

data_type = d_type(Y)
    # Y = one_hot_encode(Y)
def evaluate_ann(params, ann, X, Y, loss_function):
    ann.update_param(params)
    predictions = ann.forward(X.T)
    losses = loss_function.Evaluate(Y, predictions)
    mean_loss = np.mean(losses)  # Aggregate the losses into a single value
    return mean_loss

def plot_accuracy(train_accuracy_list, val_accuracy_list):
    
    epochs = [i[0] for i in train_accuracy_list]
    accuracies = [i[1] for i in train_accuracy_list]
    val_accuracies = [i[1] for i in val_accuracy_list]

    fig, ax = plt.subplots()
    ax.plot(epochs, accuracies, label='Training Accuracy', color='blue')
    ax.plot(epochs, val_accuracies, label='Validation Accuracy', color='green')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accurcy")
    ax.set_title("Training and Validation Accuracy over Epochs")
    ax.legend() 
    return fig

def plot_loss(loss_list, val_loss_list):

    epochs = [i[0] for i in loss_list]
    losses = [i[1] for i in loss_list]
    val_losses = [i[1] for i in val_loss_list]

    fig, ax = plt.subplots()
    ax.plot(epochs, losses, label='Loss', color='red')
    ax.plot(epochs, val_losses, label='Validation Loss', color='orange')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss over Epochs")
    ax.legend()

def train_pso(loss_fn,ann,initial_params, X_train, Y_train, population_size = 100, iterations = 100, alpha = 0.2, beta = 1.0, gamma = 2.0,  delta = 1.0, neighborhood_size = 8, optimP = "min", informant_type = "h" ):
    start_time = time.time()
    Adapter = adapter(lambda params: evaluate_ann(params,ann,X_train,Y_train,loss_fn),initial_params)
    swarm = Swarm(Adapter, alpha, beta, gamma, delta,population_size, iterations, neighborhood_size, optimP, informant_type)

    # Run the optimization
    swarm.optimize() 
    ann.update_param(swarm.gbestPos)
    end_time = time.time()
    execution_time = end_time - start_time 
    print(f"PSO executed in: {execution_time} seconds")
    return ann, swarm

def test_pso(ann, X_test, Y_test):
    prediction = ann.forward(X_test.T)
    predicted_labels = np.round(prediction)
    accuracy = np.mean(predicted_labels == Y_test)
    print(f"Accuracy on test set: {accuracy}")
    return accuracy
def test_pso_multi(ann, X_test, Y_test):
    prediction = ann.forward(X_test)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)
    print("Predicted probabilities shape:", prediction.shape)
    if prediction.shape != Y_test.shape:
        prediction = prediction.T
    predicted_labels = np.argmax(prediction, axis=1)
    print(prediction)
    labels = np.argmax(Y_test, axis=1)
    accuracy = np.mean(predicted_labels == labels)
    print(f"Accuracy on test set: {accuracy}")
    return accuracy


def train_gradient_descent(ann, X_train, Y_train, X_val, Y_val, method='mini_batch', epochs=7, rate=0.01, loss = BinaryCrossEntropyLoss,  batch_size=32):

    if (method == 'mini_batch'):
        loss, accuracy, accuracy_list, loss_list, val_accuracy_list, val_loss_list = mini_batch(ann,X_train, Y_train, X_val, Y_val, epochs, rate, loss, batch_size)
        plt1 = plot_accuracy(accuracy_list, val_accuracy_list)
        plt2 = plot_loss(loss_list, val_loss_list)
    elif (method == 'dgd'):
        loss, accuracy, accuracy_list, loss_list, val_accuracy_list, val_loss_list = dgd(ann, X_train, Y_train, X_val, Y_val, epochs, rate, loss)
        plt1 = plot_accuracy(accuracy_list, val_accuracy_list)
        plt2 = plot_loss(loss_list, val_loss_list)
    elif (method == 'sgd'):
        loss, accuracy, accuracy_list, loss_list, val_accuracy_list, val_loss_list = sgd(ann, X_train, Y_train, X_val, Y_val, epochs, rate, loss)
        plt1 = plot_accuracy(accuracy_list, val_accuracy_list)
        plt2 = plot_loss(loss_list, val_loss_list)

    return loss, accuracy, plt1, plt2


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
X_sample, Y_sample = X_train[:10], Y_train[:10]
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# print(f"Best solution found: x = {swarm.gbestPos}, f(x) = {swarm.gFit}")
initial_input = np.size(X[0])
ann= ANNBuilder.build(4,np.array([2,4,3,3]),np.array([1,3,1,4]), initial_input)
initial_params = ann.get_param()

# test_gradient_descent(ann, X_test, Y_test) 
# train_gradient_descent(X_train, Y_train, method='mini_batch', epochs=7, rate=0.01, loss = Mse,  batch_size=32)
# loss, accuracy = mini_batch(ann, X, Y, 7, 0.0001, Mse, 196)
# print(loss)
# print(accuracy)
# Train the network
ann, swarm = train_pso(data_type[2],ann,initial_params, X_train, Y_train)
swarm.plot_convergence()
swarm.plot_particle_movement()

test_pso_multi(ann, X_test, Y_test)





# loss, accuracy = train_gradient_descent(ann,X_train, Y_train, X_val, Y_val, method='mini_batch', epochs=50, rate=0.0001, loss=BinaryCrossEntropyLoss, batch_size=32)
# print("Training Loss:", loss)
# print("Training Accuracy:", accuracy)
# Test the network
# test_accuracy = test_gradient_descent(ann, X_test, Y_test)
# print("Test Accuracy:", test_accuracy)
