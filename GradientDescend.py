import numpy as np


def getTrue(classes, x):
    return classes[x]

def evaluate_ann(params, ann, X, Y, loss_function):
    ann.update_param(params)
    predictions = ann.forward(X.T)
    losses = loss_function.Evaluate(Y, predictions)
    mean_loss = np.mean(losses) 

    predicted_labels = np.round(predictions)
    correct_predictions = np.sum(predicted_labels == Y)
    accuracy = correct_predictions / len(Y) 

    return mean_loss, accuracy

def createBatches(data, classes, batch_size):
    batches = []
    for i in range(0, len(data), batch_size):
        # batch data is the data included within the batch size
        batch_data = data[i:i+batch_size]
        batch_classes = classes[i:i+batch_size]
        batches.append((batch_data, batch_classes))
    return batches

def base_gd(ann, data, classes, rate, loss):
    L = 0
    loss_gradient = 0
    correct_predictions = 0
    for index in range(len(data)):
        i = data[index]
        # gets output after forward propogation
        t = classes[index]
        y = ann.forward(i)
        print("y =" , y)
        L+= np.mean(loss.Evaluate(y, t))

        prediction = np.round(y)
        if prediction == t:
            correct_predictions += 1

        # backpropagation to be changed to PSO
        loss_gradient = loss.Derivate(y, t)
        ann.backward(loss_gradient, rate)

    avg_loss = L/len(data)
    accuracy = correct_predictions / len(data)

    return avg_loss, accuracy

def gd(ann, X_train, Y_train, X_val, Y_val, epochs, rate, loss, batch_size):
    L = 0
    accuracy_list = []
    loss_list = []
    val_loss_list = []
    val_accuracy_list = []
    batches = createBatches(X_train, Y_train, batch_size)
    # for each epoch
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_correct_predictions = 0
        total_samples = 0
        # in each epoch go through every batch in batches
        for batch_data, batch_classes in batches:
            # get the loss and accuracy for the current batch
            batch_loss, batch_accuracy = base_gd(ann, batch_data, batch_classes, rate, loss)
            # add that to the epoch loss and accuracy
            epoch_loss += batch_loss
            epoch_correct_predictions += batch_accuracy * len(batch_data)
            total_samples += len(batch_data)
        # getting average loss and accuracy for the epoch
        epoch_loss_avg = epoch_loss / len(batches)
        epoch_accuracy = epoch_correct_predictions / total_samples

        accuracy_list.append([epoch, epoch_accuracy])
        loss_list.append([epoch, epoch_loss_avg])

        val_loss, val_accuracy = evaluate_ann(ann.get_param(), ann, X_val, Y_val, loss)
        val_accuracy_list.append([epoch, val_accuracy])
        val_loss_list.append([epoch, val_loss])
        print(f"Epoch {epoch}: Train Acc = {epoch_accuracy}, Val Acc = {val_accuracy}")

        

    # calculating average loss and accuracy over all the epochs
    avg_loss =sum([i[1] for i in loss_list]) / epochs
    avg_accuracy = sum([i[1] for i in accuracy_list]) / epochs

    return avg_loss , avg_accuracy, accuracy_list, loss_list, val_accuracy_list, val_loss_list

# HEREEE
# y = [[0.5586343 ]
#  [0.53935224]
#  [0.65263914]]
# idk its there for every loop this is hhe last one
def mini_batch(ann, X_train, Y_train, X_val, Y_val, epochs, rate, loss, batch_size):
    L, accuracy, accuracy_list, loss_list, val_accuracy_list, val_loss_list = gd(ann, X_train, Y_train, X_val, Y_val, epochs, rate, loss, batch_size)
    return L, accuracy, accuracy_list, loss_list, val_accuracy_list, val_loss_list

# Decentralized Gradient Descent
def dgd(ann, data, X_train, Y_train, X_val, Y_val, epochs, rate, loss):
    L, accuracy, accuracy_list, loss_list, val_accuracy_list, val_loss_list = gd(ann, X_train, Y_train, X_val, Y_val, epochs, rate, loss, data.size)
    return L, accuracy, accuracy_list, loss_list, val_accuracy_list, val_loss_list

# Stochastic Gradient Descent
def sgd(ann, X_train, Y_train, X_val, Y_val, epochs, rate, loss):
    L, accuracy, accuracy_list, loss_list, val_accuracy_list, val_loss_list = gd(ann, X_train, Y_train, X_val, Y_val, epochs, rate, loss, 1)
    return L, accuracy, accuracy_list, loss_list, val_accuracy_list, val_loss_list