import random
import numpy as np
import matplotlib.pyplot as plt
from mlp.engine import Value
from mlp.nn import Neuron, Layer, MLP
import os
import mfcc

np.random.seed(1337)
random.seed(1337)

# initialize a model 
model = MLP(3, [256, 64, 16, 6]) # 2-layer neural network
print(model)
print("number of parameters", len(model.parameters()))

# loss function
def loss(X,y,batch_size=None):
    
    # inline DataLoader :)
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    inputs = [list(map(Value, xrow)) for xrow in Xb]
    
    # forward the model to get scores
    scores = list(map(model, inputs))
    
    # svm "max-margin" loss
    losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum((p*p for p in model.parameters()))
    total_loss = data_loss + reg_loss
    
    # also get accuracy
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)

# total_loss, acc = loss()
# print(total_loss, acc)
def data_prep():
    directory = 'OdessaRecordings'
    # List all files in the directory
    files_in_directory = os.listdir(directory)
    sample_rate = 16000
    features_arr = []
    y=[]
    filenames = ["odessa","play_music"]
    for filename in filenames:
        for i in range(29):
            features = mfcc.calculations_for_onerecording(False, directory=directory,filename=f"{filename}_{i+1}",sample_rate=sample_rate)
            features_arr.append(features)
            if filename == "odessa":
                y.append([1,0,0,0,0,0])
            else:
                y.append([0,1,0,0,0,0])

    return features_arr,y

X,y = data_prep()
# optimization
for k in range(100):
    
    # forward
    total_loss, acc = loss(X,y,8)
    
    # backward
    model.zero_grad()
    total_loss.backward()
    
    # update (sgd)
    learning_rate = 1.0 - 0.9*k/100
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    
    if k % 1 == 0:
        print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")


