import os
import sys

import sklearn
from sklearn import model_selection
from sklearn.datasets import fetch_openml
import numpy as np


class LogisticRegression:

    def __init__(self, learningRate=0.01, iterations=10000, fitIntercept=True, verbose=False, epsilon=1e-5):
        self.learningRate = learningRate
        self.iterations = iterations
        self.fitIntercept = fitIntercept
        self.verbose = verbose
        self.epsilon = epsilon

    def sigmoid(self, z):
        # Sigmoid Function
        return 1 / (1 + np.exp(-z))

    def loss(self, y, h):
        if self.verbose:
            # np.set_printoptions(threshold=sys.maxsize)
            # print("y: " + str(y))
            # print("h: " + str(h))
            pass
        # Loss Function with modified epsilon
        return (-y * np.log(h + self.epsilon) - (1 - y + self.epsilon) * np.log(1 - h + self.epsilon)).mean()

    def addIntercept(self, x):
        intercept = np.ones((x.shape[0], 1))
        return np.concatenate((intercept, x), axis=1)

    def fit(self, x, y):
        if self.fitIntercept:
            x = self.addIntercept(x)

        # Weight initialization
        self.theta = np.zeros(x.shape[1])

        # Apply Gradient Descent
        for i in range(self.iterations):
            z = np.dot(x, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(x.T, (h - y)) / y.size
            self.theta -= self.learningRate * gradient

            z = np.dot(x, self.theta)
            h = self.sigmoid(z)
            loss = self.loss(h, y)

            if self.verbose and i % 100 == 0:
                print("Loss: " + str(loss) + '\n')
                print("Theta: " + str(self.theta[4]))

    def predictProbability(self, x):
        m, n = np.shape(x)
        x = np.c_[np.ones((m, 1)), x]
        self.theta = np.zeros((n + 1, 1))
        return self.sigmoid(np.dot(x, self.theta))

    def predict(self, x, threshold=0.5):
        # Predict which category the data belongs
        return self.predictProbability(x) >= threshold


if __name__ == '__main__':
    try:
        # Initialize Settings + Dataset
        np.set_printoptions(threshold=sys.maxsize)
        # dataPath = "Data/"
        # mnist = fetch_openml('mnist_784', data_home=dataPath)
        mnist = sklearn.datasets.load_digits() # Use for Debugging!

        # Split Data
        model = LogisticRegression(verbose=True, iterations=1000)
        xTrain = mnist.data
        yTrain = (mnist.target != 0)
        xTrain, xTest, yTrain, yTest = model_selection.train_test_split(xTrain, yTrain, train_size=0.65, test_size=0.35,
                                                                        random_state=101)
        # Fit and Predict
        model.fit(xTrain, yTrain)
        preds = model.predict(xTest)
        print(preds.mean())  # Prints 1.0 on MNIST

    except Exception as e:
        exceptionType, exceptionObject, exceptionThrowback = sys.exc_info()
        fileName = os.path.split(exceptionThrowback.tb_frame.f_code.co_filename)[1]
        print(exceptionType, fileName, exceptionThrowback.tb_lineno)
        print(e)
