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

#This function checks the class column and determines the type of data and creates a list defining the fixed attributes for the output layer needed to run
#the dataset correctly
def d_type(Y):
    unique_labels = np.unique(Y, axis=0)
    #If there are only 2 classes, it is a binary classification
    if len(unique_labels) == 2:
        #Since we can have any activation safely, it is none
        #Since we need Binary Cross Entropy as the loss, we fix it
        #Since we need one output Node specifically for binary class, without a softmax, we set the number of nodes in the output layer as 1
        return ["binary_class", None, BinaryCrossEntropyLoss, 1]
    #similarly, if we have more than 2 classes, we check if its a one hot encoded numpy array indicating multi-class
    #If there are more than 2 classes, and the classes are float, its a regression 
    elif len(unique_labels) > 2 and isinstance(unique_labels[0], (np.ndarray)):
        print(len(unique_labels))
        return ["multi_class", Softmax, CrossEntropyLoss, len(unique_labels)]
    elif len(unique_labels) > 2 and isinstance(unique_labels[0], (float,np.float32, np.float64)):
        return ["logistic", Linear, Mse, 1]
    else:
        return [None, None, None, None]
#Function to one hot encode the classes   
def one_hot_encode(Y):
    label_encoder = LabelEncoder()
    Y_integer_encoded = label_encoder.fit_transform(Y)
    number_of_classes = len(np.unique(Y_integer_encoded))
    one_hot = np.zeros((len(Y_integer_encoded), number_of_classes))
    one_hot[np.arange(len(Y_integer_encoded)), Y_integer_encoded] = 1
    return one_hot
#Function to plot the classification scatterplot
def classification_scatter(f1,f2,p):
    fig, ax = plt.subplots()
    Scatter = ax.scatter(f1,f2,c=p,cmap='viridis')
    legend1 = ax.legend(*Scatter.legend_elements(),
                        loc="upper right", title="Classes")
    ax.add_artist(legend1)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Classification Scatter Plot')
    return fig
#function to that evaluates the fitness for an ANN (We this to the adapter)
def evaluate_ann(params, ann, X, Y, loss_function):
    ann.update_param(params)
    predictions = ann.forward(X.T)
    losses = loss_function.Evaluate(Y, predictions)
    mean_loss = np.mean(losses)  
    return mean_loss
#Function optimize the ANN using PSO
def train_pso(loss_fn,ann,initial_params, X_train, Y_train, population_size = 100, iterations = 100, alpha = 0.2, beta = 1.0, gamma = 2.0,  delta = 1.0, neighborhood_size = 8, optimP = "min", informant_type = "h" ):
    start_time = time.time()
    Adapter = adapter(lambda params: evaluate_ann(params,ann,X_train,Y_train,loss_fn),initial_params)
    swarm = Swarm(Adapter, alpha, beta, gamma, delta,population_size, iterations, neighborhood_size, optimP, informant_type)
    swarm.optimize() 
    ann.update_param(swarm.gbestPos)
    end_time = time.time()
    execution_time = end_time - start_time 
    print(f"PSO executed in: {execution_time} seconds")
    return ann, swarm
#Function to test the optimized ANN with the validation data (BINARY CLASS)
def test_pso(ann, X_test, Y_test):
    f1 = X_test[:,0]
    f2 = X_test[:,1]
    prediction = ann.forward(X_test.T)
    predicted_labels = np.round(prediction)
    plt = classification_scatter(f1,f2,predicted_labels)
    accuracy = np.mean(predicted_labels == Y_test)
    print(f"Accuracy on test set: {accuracy}")
    return accuracy,plt
#Function to test the optimized ANN with the validation data (MULTI CLASS)
def test_pso_multi(ann, X_test, Y_test):
    f1 = X_test[:,0]
    f2 = X_test[:,1]
    prediction = ann.forward(X_test)
    if prediction.shape != Y_test.shape:
        prediction = prediction.T
    predicted_labels = np.argmax(prediction, axis=1)
    plt = classification_scatter(f1,f2,predicted_labels)
    labels = np.argmax(Y_test, axis=1)
    accuracy = np.mean(predicted_labels == labels)
    print(f"Accuracy on test set: {accuracy}")
    return accuracy,plt

data=pd.read_csv("iris.data", delimiter=",")
Ycol = data.columns[-1]
X = data.drop(Ycol,axis=1).values
Y = data[Ycol].values
if(isinstance(Y[0],str)):
    Y = one_hot_encode(Y)
    unique_labels = np.unique(Y, axis=0)
    print(type(unique_labels[0]))

data_type = d_type(Y)

shuffle_idx = np.arange(Y.shape[0])
shuffle_rng = np.random.RandomState(123)
shuffle_rng.shuffle(shuffle_idx)
X, Y = X[shuffle_idx], Y[shuffle_idx]

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_sample, Y_sample = X_train[:10], Y_train[:10]
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

initial_input = np.size(X[0])
ann= ANNBuilder.build(4,np.array([2,4,3,3]),np.array([1,3,1,4]), initial_input)
initial_params = ann.get_param()
ann, swarm = train_pso(data_type[2],ann,initial_params, X_train, Y_train)
swarm.plot_convergence()
swarm.plot_particle_movement()

test_pso_multi(ann, X_test, Y_test)

