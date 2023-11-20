import numpy as np
def getTrue(classes, x):
    return classes[x]

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
    accuracy = 0
    threshold = 0.5
    c =0
    for index in range(len(data)):
        print(c,"helo")
        c+=1
        i = data[index]
        # gets output after forward propogation
        t = classes[index]
        y = ann.forward(i)
        L+= loss.Evaluate(y, t)
        print("L::::" ,L)
        prediction = 1 if np.argmax(y)> threshold else 0
        accuracy += 1 if prediction == t else 0
        print("pred",prediction)
        print("T",t)
        print(prediction == t)
        print("----------------------------------")
        print(accuracy)

    # backpropagation to be changed to PSO
    loss_gradient = loss.Derivate(y, t)
    ann.backward(loss_gradient, rate)

    accuracy /= len(data)
    avg_loss = L/len(data)
    
    return avg_loss, accuracy

def gd(ann, data, classes, epochs, rate, loss, batch_size):
    L = 0
    accuracy = 0
    batches = createBatches(data, classes, batch_size)
    # for each epoch
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        # in each epoch go through every batch in batches
        for batch in batches:
            # get the loss and accuracy for the current batch
            batch_loss, batch_accuracy = base_gd(ann, batch[0], batch[1], rate, loss)
            # add that to the epoch loss and accuracy
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
        # getting average loss and accuracy for the epoch
        L += epoch_loss/len(batches)
        accuracy += epoch_accuracy/len(batches)

    # calculating average loss and accuracy over all the epochs
    avg_loss = L/epochs
    avg_accuracy = accuracy/epochs

    return avg_loss , avg_accuracy

def mini_batch(ann, data, classes, epochs, rate, loss, batch_size):
    L, accuracy = gd(ann, data, classes, epochs, rate, loss, batch_size)
    return L, accuracy

# Decentralized Gradient Descent
def dgd(ann, data, classes, epochs, rate, loss):
    L, accuracy = gd(ann, data, classes, epochs, rate, loss, data.size)
    return L, accuracy

# Stochastic Gradient Descent
def sgd(ann, data, classes, epochs, rate, loss):
    L, accuracy = gd(ann, data, classes, epochs, rate, loss, 1)
    return L, accuracy