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
    c =0
    for index in range(len(data)):
        print(c,"helo")
        c+=1
        i = data[index]
        # gets output after forward propogation
        t = classes[index]
        y = ann.forward(i)
        L+= loss.Evaluate(y, t)
        accuracy += 1 if np.array_equal(y, t) else 0
    
    accuracy /= len(data)

    # backpropagation to be changed to PSO
    return L, accuracy

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
            batch_loss, batch_accurancy = base_gd(ann, batch[0], batch[1], rate, loss)
            # add that to the epoch loss and accuracy
            epoch_loss += batch_loss
            epoch_accuracy += batch_accurancy
        # getting average loss and accuracy for the epoch
        L += epoch_loss/len(batches)
        accuracy += epoch_accuracy/len(batches)

    # calculating average loss and accuracy over all the epochs
    L/= epochs
    accuracy/=epochs

    return L, accuracy

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