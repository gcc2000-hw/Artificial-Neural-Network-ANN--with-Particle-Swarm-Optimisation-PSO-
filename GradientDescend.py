from Loss import Loss

def getTrue(classes, x):
    return classes[x]

def base_gd(ann, data, classes, rate, loss):
    L = 0
    dL = 0
    accuracy = 0
    for i in data:
        # gets output after forward propogation
        y = ann.forward(i)
        t = getTrue(classes, i)
        L+= Loss.evaluate(y, t)
        dL += Loss.Derivate(y,t)
        accuracy += 1 if y == t else 0
    
    L /= len(data)
    dL /= len(data)
    accuracy /= len(data)

    # backpropagation to be changed to PSO
    ann.backpropagate(dL, rate)
    return L, accuracy


   