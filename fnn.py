"""

This program builds a two-layer neural network for the Iris dataset.
The first layer is a relu layer with 10 units, and the second one is 
a softmax layer. The network structure is specified in the "train" function.

The parameters are learned using SGD.  The forward propagation and backward 
propagation are carried out in the "compute_neural_net_loss" function.

"""


import numpy as np
import os, sys
import math

# Data sets
IRIS_TRAINING = os.getcwd() + "/data/iris_training.csv"
IRIS_TEST = os.getcwd() + "/data/iris_test.csv"

def get_data():
    # Load datasets.
    train_data = np.genfromtxt(IRIS_TRAINING, skip_header=1, 
        dtype=float, delimiter=',') 
    test_data = np.genfromtxt(IRIS_TEST, skip_header=1, 
        dtype=float, delimiter=',') 
    train_x = train_data[:, :4]
    train_y = train_data[:, 4].astype(np.int64)
    test_x = test_data[:, :4]
    test_y = test_data[:, 4].astype(np.int64)

    return train_x, train_y, test_x, test_y

def compute_neural_net_loss(params, X, y, reg=0.0):
    """
    Neural network loss function.
    Inputs:
    - params: dictionary of parameters, including "W1", "b1", "W2", "b2"
    - X: N x D array of training data. Each row is a D-dimensional point.
    - y: 1-d array of shape (N, ) for the training labels.

    Returns:
    - loss: the softmax loss with regularization
    - grads: dictionary of gradients for the parameters in params
    """
    # Unpack variables from the params dictionary
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    N, D = X.shape

    loss = 0.0
    grads = {}

    # forward propagation
    relu = lambda x : x * (x > 0)
    z1 = X.dot(W1) + b1
    u1 = np.vectorize(relu)(z1)
    z2 = u1.dot(W2) + b2
    u2 = np.vectorize(math.exp)(z2)
    NLL = - (np.vectorize(math.log)((np.array([u2[i][y[i]] / u2[i].sum() for i in range(N)])))).sum()
    loss = NLL / N + 0.5 * reg * ((W1 ** 2).sum() + (W2 ** 2).sum())

    # backward propagation
    d_relu = lambda x: 1 * (x >= 0)
    delta2 = np.zeros(z2.shape)
    for i in range(delta2.shape[0]):
        for k in range(delta2.shape[1]):
            delta2[i][k] = u2[i][k] / u2[i].sum() - (y[i] == k)

    dW2 = np.zeros(W2.shape)
    for i in range(N):
        dW2 += (u1[i].reshape(-1, 1)).dot(delta2[i].reshape(1, -1))
    dW2 = dW2 / N + reg * W2

    db2 = np.zeros(len(b2))
    for i in range(N):
        db2 += delta2[i]
    db2 = db2 / N

    delta1 = np.zeros(z1.shape)
    for i in range(delta1.shape[0]):
        for j in range(delta1.shape[1]):
            delta1[i][j] = d_relu(z1[i][j]) * (delta2[i].dot(W2[j].T))

    dW1 = np.zeros(W1.shape)
    for i in range(N):
        dW1 += (X[i].reshape(-1, 1)).dot(delta1[i].reshape(1, -1))
    dW1 = dW1 / N + reg * W1

    db1 = np.zeros(len(b1))
    for i in range(N):
        db1 += delta1[i]
    db1 = db1 / N

    grads['W1']=dW1
    grads['W2']=dW2
    grads['b1']=db1
    grads['b2']=db2
    
    return loss, grads

def predict(params, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - params: dictionary of parameters, including "W1", "b1", "W2", "b2"
    - X: N x D array of training data. Each row is a D-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    # Unpack variables from the params dictionary
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']

    y_pred = np.zeros(X.shape[1])
   
    relu = lambda x: x * (x > 0)
    z1 = np.dot(X,W1)+b1
    u1 = relu(z1)
    z2 = np.dot(u1,W2)+b2
    y_pred = np.argmax(z2, axis=1)
    
    return y_pred

def acc(ylabel, y_pred):
    return np.mean(ylabel == y_pred)

def sgd_update(params, grads, learning_rate):
    """
    Perform sgd update for parameters in params.
    """
    for key in params:
        params[key] += -learning_rate * grads[key]


def train(X, y, Xtest, ytest, learning_rate=1e-3, reg=1e-5, epochs=100, batch_size=20):
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
    num_iters_per_epoch = int(math.floor(1.0*num_train/batch_size))

    params = {}
    std = 0.001
    params['W1'] = std * np.random.randn(dim, 10)
    params['b1'] = np.zeros(10)
    params['W2'] = std * np.random.randn(10, num_classes)
    params['b2'] = np.zeros(num_classes)

    for epoch in range(max_epochs):
        perm_idx = np.random.permutation(num_train)
        # perform mini-batch SGD update
        for it in range(num_iters_per_epoch):
            idx = perm_idx[it*batch_size:(it+1)*batch_size]
            batch_x = X[idx]
            batch_y = y[idx]
            
            # evaluate loss and gradient
            loss, grads = compute_neural_net_loss(params, batch_x, batch_y, reg)

            # update parameters
            sgd_update(params, grads, learning_rate)
            
        # evaluate and print every 10 steps
        if epoch % 10 == 0:
            train_acc = acc(y, predict(params, X))
            test_acc = acc(ytest, predict(params, Xtest))
            print('Epoch %4d: loss = %.2f, train_acc = %.4f, test_acc = %.4f' \
                % (epoch, loss, train_acc, test_acc))
    
    return params


max_epochs = 200
batch_size = 20
learning_rate = 0.1
reg = 0.001

# get training and testing data
train_x, train_y, test_x, test_y = get_data()
params = train(train_x, train_y, test_x, test_y, learning_rate, reg, max_epochs, batch_size)

# Classify two new flower samples.
def new_samples():
    return np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
new_x = new_samples()
predictions = predict(params, new_x)

print("New Samples, Class Predictions:    {}\n".format(predictions))
