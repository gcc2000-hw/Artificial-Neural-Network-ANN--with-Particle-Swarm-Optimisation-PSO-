import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from Loss import *
from NetworkBuilder import ANNBuilder
from main import train_pso, test_pso, train_gradient_descent

# import matplotlib.pyplot as plt
# import networkx as nx

activation_map_ui = {
    "Sigmoid": 1,
    "ReLU": 2,
    "TanH": 3
}

# Function to load data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
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
        ann = train_pso(ann, X_train, Y_train, population_size, iterations, alpha, beta, gamma, delta, neighborhood_size, optimization_problem)
        accuracy = test_pso(ann, X_test, Y_test)
        return accuracy 
    
    elif method == "Gradient Descent":
        loss, accuracy = train_gradient_descent(ann, X_train, Y_train, gd_method, epochs, learning_rate, BinaryCrossEntropyLoss, batch_size)
        result = f"Training Loss: {loss}, Accuracy: {accuracy}"
        return result
    


# UI
st.title("ANN with PSO and Gradient Descent")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a CSV file or use default dataset", type="csv")
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
