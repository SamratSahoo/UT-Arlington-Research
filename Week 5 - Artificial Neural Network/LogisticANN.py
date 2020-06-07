import sys

import numpy as np
import sklearn
import torch
from sklearn import model_selection
from torch import nn
import torchvision
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, xTrain, yTrain, xTest, yTest, iterations=1000, learningRate=0.1, verbose=False, epsilon=1e-5):

        # Training + Testing Data
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.xTest = xTest
        self.yTest = yTest

        # Gradient Descent Stuff
        self.iterations = iterations
        self.learningRate = learningRate

        # Miscellaneous
        self.verbose = verbose
        self.epsilon = epsilon

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initializeZeros(self, dim):
        shape = (dim, 1)
        weights = np.zeros(shape)
        bias = 0
        return weights, bias

    def loss(self, h):
        print(self.yTrain.shape, h.shape)
        return (-1 * self.yTrain * np.log(h + self.epsilon) - (1 - self.yTrain) * np.log(1 - h + self.epsilon)).mean()

    def propagate(self, weights, bias):
        # Forward Propagation
        activation = self.sigmoid(np.dot(weights.T, self.xTrain) + bias)
        loss = self.loss(activation)

        # Back Propagation
        dw = (1 / self.xTrain.shape[0]) * np.dot(self.xTrain, (activation - self.yTrain).T)
        db = np.sum(activation - self.yTrain).mean()

        return dw, db, loss

    def gradientDescent(self, weights, bias):
        costs = []
        for i in range(self.iterations):
            dw, db, loss = self.propagate(weights, bias)
            weights -= self.learningRate * dw
            bias -= self.learningRate * db

            if i % 100 == 0:
                costs.append(loss)

            if self.verbose and i % 100 == 0:
                print("Cost is {} for iteration {}".format(costs[i - 1], i))

        return weights, bias, dw, db, costs

    def predict(self, weights, bias, X):
        print(X)
        predictions = np.zeros(1, X[1])
        weights = weights.reshape(X.shape[0], 1)

        activation = self.sigmoid((np.dot(weights.T, X) + bias))
        for i in range(activation.shape[1]):
            predictions[0, i] = 1 if activation[0, i] > 0.5 else 0

        return predictions


if __name__ == '__main__':
    # Initialize Settings + Dataset
    np.set_printoptions(threshold=sys.maxsize)
    dataPath = "Data/"
    # mnist = fetch_openml('mnist_784', data_home=dataPath)
    mnist = sklearn.datasets.load_digits()  # Use for Debugging!
    xTrain = mnist.data
    yTrain = (mnist.target != 0)
    xTrain, xTest, yTrain, yTest = model_selection.train_test_split(xTrain, yTrain, train_size=0.65, test_size=0.35,
                                                                    random_state=101)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork(xTrain, yTrain, xTest, yTest)

    weights, bias = model.initializeZeros(xTrain.shape[0])
    weights, bias, dw, db, costs = model.gradientDescent(weights, bias)
    preds = model.predict(weights, bias, xTest)
