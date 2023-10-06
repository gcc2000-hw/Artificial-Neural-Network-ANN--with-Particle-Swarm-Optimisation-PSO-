def base_gd(ann, data, classes, rate, loss):
    for i in data:
        # gets output after forward propogation
        y = ann.forward(i)
        t = getTrue(classes, i)
   