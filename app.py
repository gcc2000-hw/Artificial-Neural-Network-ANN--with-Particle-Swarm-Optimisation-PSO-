import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from Loss import *
from NetworkBuilder import ANNBuilder
from main import train_pso, test_pso, train_gradient_descent

import matplotlib.pyplot as plt
import random
# import networkx as nx

activation_map_ui = {
    "Sigmoid": 1,
    "ReLU": 2,
    "TanH": 3
}

informant_type_map= {
    "Random" : "r",
    "Distance": "d",
    "Hybrid" : "h"
}

# Generate a raondom color in hex
def generate_random_color():
    return "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])

# Function to draw neural network
# Code from https://gist.github.com/craffel/2d727968c3aaebd10359
def draw_neural_net(ax, left, right, bottom, top, layer_sizes): 
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)

    layer_colors = [generate_random_color() for _ in range(n_layers)]

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color=layer_colors[n], ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)

# Function to plot the ANN
def plot_neural_network(layer_sizes):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')
    draw_neural_net(ax, .1, .9, .1, .9, layer_sizes)
    return fig

# Function to load data
def load_data(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            return pd.read_csv(uploaded_file, delimiter=',')
        elif uploaded_file.name.endswith('.data'):
            return pd.read_csv(uploaded_file, delimiter=',')
    else:
        # Load default dataset
        return pd.read_csv("data_banknote_authentication.txt")
    
# Function to preprocess the data
def preprocess_data(data, train_size):
    Ycol = data.columns[-1]
    X = data.drop(Ycol,axis=1).values
    Y = data[Ycol].values
    if(isinstance(Y[0],str)):
        unique_labels, Y_mapped = np.unique(Y,return_inverse=True)
        D = {label : i for i,label in enumerate(unique_labels)}
        Y = Y_mapped
    shuffle_idx = np.arange(Y.shape[0])
    shuffle_rng = np.random.RandomState(123)
    shuffle_rng.shuffle(shuffle_idx)
    X, Y = X[shuffle_idx], Y[shuffle_idx]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= (1 - train_size), random_state=42)

    return X_train, X_test, Y_train, Y_test
    

# Function to train the model
def train_model(method):
    ann = st.session_state.get('ann', None)
    if not ann:
        st.warning("ANN not initialized. Please build the ANN first.")
        return None

    if method == "PSO":
        ann = train_pso(ann, X_train, Y_train, population_size, iterations, alpha, beta, gamma, delta, neighborhood_size, optimization_problem, informant_type)
        accuracy = test_pso(ann, X_test, Y_test)
        return accuracy 
    
    elif method == "Gradient Descent":
        loss, accuracy = train_gradient_descent(ann, X_train, Y_train, gd_method, epochs, learning_rate, BinaryCrossEntropyLoss, batch_size)
        print("HERE:::" , loss)
        result = f"Training Loss: {float(loss[0])}, Accuracy: {accuracy}"
        return result
    


# UI
st.title("ANN with PSO and Gradient Descent")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a CSV or TXT file", type=["csv", "txt", "data"])
data = load_data(uploaded_file)


if st.checkbox("Show Data"):
    st.subheader("Dataset")
    st.dataframe(data)

#Train-Test split
st.subheader("Train-Test Split")
train_size = st.slider("Training data size", min_value=0.1, max_value=0.9, value=0.8, step=0.1)
X_train, X_test, Y_train, Y_test = preprocess_data(data, train_size)

#Initialize ANN
st.subheader("Build your ANN")
num_layers = st.number_input("Number of Layers", min_value=1, max_value=10, value=3)
layer_nodes = []
list_activation = []
for i in range(num_layers):
    nodes = st.number_input(f"Nodes in layer {i+1}", min_value=1, max_value=100, value=3, key=f"layer_{i+1}")
    layer_nodes.append(nodes)
    activation_fn_name = st.selectbox(f"Activation function for layer {i+1}", list(activation_map_ui.keys()), key=f"activation_{i+1}")
    activation_fn = activation_map_ui[activation_fn_name] 
    list_activation.append(activation_fn)
  

#Initialize ANN button
if st.button("Build ANN"):
    # X_train, X_test, Y_train, Y_test = preprocess_data(data, train_size=train_size)
    initial_input = np.size(X_train[0])
    ann = ANNBuilder.build(num_layers,np.array(layer_nodes),np.array(list_activation), initial_input)
    st.session_state['ann'] = ann
    st.write("ANN Initialized")
    fig = plot_neural_network(layer_nodes)
    st.pyplot(fig)

st.markdown("---")

# Method selection
method = st.selectbox("Choose the training method:",
                      ["PSO", "Gradient Descent"])

if method == "PSO":
    # Particle Swarm Hyperparameters
    st.subheader("PSO Hyperparameters")
    population_size = st.number_input("Population Size", min_value=10, max_value=1000, value=100)
    iterations = st.number_input("Iterations", min_value=10, max_value=1000, value=100)
    alpha = st.number_input("Alpha", min_value=0.0, max_value=1000.0, value=0.2)
    beta = st.number_input("Beta", min_value=0.0, max_value=2.0, value=1.0)
    gamma = st.number_input("Gamma", min_value=0.0, max_value=2.0, value=2.0,)
    delta = st.number_input("Delta", min_value=0.0, max_value=2.0, value=1.0)
    neighborhood_size = st.number_input("Neighbourhood Size", min_value=5, max_value=20, value= 9)
    optimization_problem = st.selectbox("Optimization Problem", ["min", "max"])
    informant_type = st.selectbox("Informant Type", list(informant_type_map.keys()))
elif method == "Gradient Descent":
    # Gradient Descent Hyperparameters
    st.subheader("Gradient Descent Hyperparameters")
    gd_method = st.selectbox("GD Method", ["mini_batch", "dgd", "sgd"])
    epochs = st.number_input("Epochs", min_value=1, max_value = 100, value = 100)
    learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.01)
    if gd_method == "mini_batch":
        batch_size = st.number_input('Batch Size', min_value=1, max_value=100, value=32)

# Train button

if st.button("Train Model"):
    result = train_model(method)
    st.write("Fitness: ",result)  # Display results
