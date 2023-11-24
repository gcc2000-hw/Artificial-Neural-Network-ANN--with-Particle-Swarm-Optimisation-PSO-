import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from Loss import *
from NetworkBuilder import ANNBuilder
from main import train_pso, test_pso, test_pso_multi, train_gradient_descent, d_type, one_hot_encode

import matplotlib.pyplot as plt
import random

st.set_option('deprecation.showPyplotGlobalUse', False)
# dictionary to map activation functions to numbers
activation_map_ui = {
    "Sigmoid": 1,
    "ReLU": 2,
    "TanH": 3
}
# dictionary to map the informant types to its identifiers
informant_type_map= {
    "Random" : "r",
    "Distance": "d",
    "Hybrid" : "h"
}
# Generate a raondom color in hex
def generate_random_color():
    # concatenating six randomly chosen characters with # to generate a colour
    return "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])

# Function to draw neural network
# Code from https://gist.github.com/craffel/2d727968c3aaebd10359
def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    # to store the number of layers in the network
    n_layers = len(layer_sizes)
    # To get the vertical spacing between the layers
    v_spacing = (top - bottom)/float(max(layer_sizes))
    # to get the horizontal spacing
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # generates a list of random colours (one colour per layer)
    layer_colors = [generate_random_color() for _ in range(n_layers)]
    # Nodes
    # iterates over each node and layer 
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            # creates a circle 
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color=layer_colors[n], ec='k', zorder=4)
            # adds the circle to the matplotlib axes
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        # vertical position of top node in current layer
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        # verticle position for the top node in the next layer
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        # Loop through each node in current layer
        for m in range(layer_size_a):
            # loop through each node in next layer
            for o in range(layer_size_b):
                # create a line from a node in the current layer to a node in the next layer
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                # adds the line to the matplotlib axes
                ax.add_artist(line)

# Function to plot the ANN
def plot_neural_network(layer_sizes):
    # fig with specific size
    fig = plt.figure(figsize=(12, 12))
    # get current axes
    ax = fig.gca()
    ax.axis('off')
    # call the function to draw the neural network
    draw_neural_net(ax, .1, .9, .1, .9, layer_sizes)
    return fig

# Function to load data
def load_data(uploaded_file):
    # check if user uploaded a file
    if uploaded_file is not None:
        # read file into dataframe
        return pd.read_csv(uploaded_file,delimiter=",")
    else:
        # load default dataset
        return pd.read_csv("data_banknote_authentication.txt", delimiter=",")
    
# Function to preprocess the data
def preprocess_data(data, train_size, selected_columns):
    # get last column from data
    Ycol = data.columns[-1]
    # get values
    Y = data[Ycol].values
    # extract features from data based on user selected columns
    X = data[selected_columns].values

    # check if target values are strings
    if(isinstance(Y[0],str)):
        # apply one hot encode
        Y = one_hot_encode(Y)
   
    # get problem type, activation, loss and number of output nodes
    problem_type, activation_fn, loss_fn, output_nodes = d_type(Y)
    # an array of indices for shuffling the data
    shuffle_idx = np.arange(Y.shape[0])
    # random seed for reproducablity 
    shuffle_rng = np.random.RandomState(123)
    # shuffle the data
    shuffle_rng.shuffle(shuffle_idx)
    # apply shuffle to features and target column
    X, Y = X[shuffle_idx], Y[shuffle_idx]
    # split data into training and temp
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size= (1 - train_size), random_state=42)
    # split temp into validation and test
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    # store as session state
    st.session_state["problem_type"] = problem_type
    st.session_state["activation_fn"] = activation_fn
    st.session_state["loss_fn"] = loss_fn
    st.session_state["output_nodes"] = output_nodes

    # return the split data
    return X_train, X_val, X_test, Y_train, Y_val, Y_test
    
# Function to train the model
def train_model():
    # reteive parameters from session state
    ann = st.session_state.get('ann', None)
    initial_params = st.session_state.get('initial_params', None)
    problem_type = st.session_state.get("problem_type", None)
    loss_fn = st.session_state.get('loss_fn',None)

    # check if ann is initialized 
    if not ann:
        st.warning("ANN not initialized. Please build the ANN first.")
        return None
    
    # train ann using pso
    ann, swarm = train_pso(loss_fn,ann, initial_params, X_train, Y_train,population_size, iterations, alpha, beta, gamma, delta, neighborhood_size, optimization_problem, informant_type)
    # plot the graphs
    plt1 = swarm.plot_convergence()
    plt2 = swarm.plot_particle_movement()
    # check if the problem type is multi class
    if problem_type == "multi_class":
        # test the ann using multi clas test
        accuracy,classification_plot= test_pso_multi(ann, X_test, Y_test)
    else:
        # test the ann on test set
        accuracy, classification_plot= test_pso(ann, X_test, Y_test)
    # display the graphs
    st.pyplot(plt1)
    st.pyplot(plt2)
    st.pyplot(classification_plot)
    return accuracy

# UI
# set the title
st.title("ANN with PSO and Gradient Descent")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a CSV or TXT file", type=["csv", "txt", "data"])
data = load_data(uploaded_file)

# a check box to show and hide the dataset
if st.checkbox("Show Data"):
    st.subheader("Dataset")
    st.dataframe(data)
    # check if data is loaded
if data is not None:
    # extract all features except last one
    all_columns = data.columns[:-1].tolist()
    # allow users to select features for training
    selected_columns = st.multiselect("Select columns for training", all_columns, default=all_columns)
else:
    # if no data, return empty list
    selected_columns = []

#Train-Test split
st.subheader("Train-Test Split")
# slider so user can choose the size of training data
train_size = st.slider("Training data size", min_value=0.1, max_value=0.9, value=0.8, step=0.1)
# preprocessing data inti train, validation and test sets
X_train, X_val, X_test, Y_train, Y_val, Y_test = preprocess_data(data, train_size, selected_columns)

#Initialize ANN
# create a subheader
st.subheader("Build your ANN")
# create a numeric input to enter the number of layers
num_layers = st.number_input("Number of Layers", min_value=1, max_value=10, value=3)
# initialize lists
layer_nodes = []
list_activation = []
# loop through each layer except the last layer to get user input for nodes and activation functions
for i in range(num_layers-1):
    # numeric input for number of nodes in a layer
    nodes = st.number_input(f"Nodes in layer {i+1}", min_value=1, max_value=100, value=3, key=f"layer_{i+1}")
    # append the number of nodes to the list
    layer_nodes.append(nodes)
    # dropdown to choose activation function
    activation_fn_name = st.selectbox(f"Activation function for layer {i+1}", list(activation_map_ui.keys()), key=f"activation_{i+1}")
    # get numeric code for the activation function
    activation_fn = activation_map_ui[activation_fn_name] 
    # add the activation function to list
    list_activation.append(activation_fn)

# get output nodes and problem type from session state
output_nodes = st.session_state.get("output_nodes", None)
problem_type = st.session_state.get("problem_type", None)

# check if problem type is multi class
if problem_type == "multi_class":
    # for multi-class, set softmax and fix the number of output nodes
    last_layer_nodes = st.number_input("Nodes in the last layer", min_value=1, max_value=100, value=output_nodes, key="last_layer", disabled=True)
    # append number of nodes in last layer
    layer_nodes.append(last_layer_nodes)
    # append softmax to the list
    activation_fn_last_layer = st.selectbox("Activation function for the last layer", ["Softmax"], disabled=True)
    # append numeric code for softmax
    list_activation.append(4)
else:
    if output_nodes is None:
        # default output node to 1
        output_nodes = 1 
        last_layer_nodes = st.number_input("Nodes in the last layer", min_value=1, max_value=100, value=output_nodes, key="last_layer")
    else:
        # if output node is specified use that and disable it so user cannot change it
        last_layer_nodes = st.number_input("Nodes in the last layer", min_value=1, max_value=100, value=output_nodes, key="last_layer", disabled=True)
        # append the number of nodes in the last layer
        layer_nodes.append(last_layer_nodes)

    # retrive default activation function if specified
    default_activation_fn = st.session_state.get("activation_fn", Sigmoid)
    # find the key for the default activation function
    default_activation_key = next((key for key, value in activation_map_ui.items() if value == default_activation_fn), "Sigmoid")
    activation_fn_last_layer = st.selectbox("Activation function for the last layer", list(activation_map_ui.keys()), index=list(activation_map_ui.keys()).index(default_activation_key), key="last_activation", disabled=default_activation_fn is not None)
    # append the code for the selected activation function
    list_activation.append(activation_map_ui[activation_fn_last_layer])

#Initialize ANN button
if st.button("Build ANN"):
    # determine input size for yhe first layer
    initial_input = np.size(X_train[0])
    # build the ann using parameters specified by the user
    ann = ANNBuilder.build(num_layers,np.array(layer_nodes),np.array(list_activation), initial_input)
    # get ann from session state
    st.session_state['ann'] = ann
    # get initial parameters of ann
    initial_params = ann.get_param()
    st.session_state['initial_params'] = initial_params
    st.write("ANN Initialized")
    fig = plot_neural_network(layer_nodes)
    st.pyplot(fig)

st.markdown("---")

# PSO Hyperparameters selection
st.subheader("PSO Hyperparameters")
population_size = st.number_input("Population Size", min_value=10, max_value=1000, value=100)
iterations = st.number_input("Iterations", min_value=10, max_value=1000, value=100)
alpha = st.number_input("Alpha", min_value=0.0, max_value=1000.0, value=0.2)
beta = st.number_input("Beta", min_value=0.0, max_value=2.0, value=1.0)
gamma = st.number_input("Gamma", min_value=0.0, max_value=2.0, value=2.0,)
delta = st.number_input("Delta", min_value=0.0, max_value=2.0, value=1.0)
neighborhood_size = st.number_input("Neighbourhood Size", min_value=2, max_value=200, value= 20)
optimization_problem = st.selectbox("Optimization Problem", ["min", "max"])
informant_type = st.selectbox("Informant Type", list(informant_type_map.keys()))

# Train button
if st.button("Train Model"):
    result = train_model()
    st.write("Accuracy: ",result) 
