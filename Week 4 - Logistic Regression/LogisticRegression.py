import sys
from sklearn import datasets
import numpy as np


class LogisticRegression:

    def __init__(self, learningRate=0.01, iterations=100000, fitIntercept=True, verbose=False, epsilon=1e-5):
        self.learningRate = learningRate
        self.iterations = iterations
        self.fitIntercept = fitIntercept
        self.verbose = verbose
        self.epsilon = epsilon

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, y, h):
        if self.verbose:
            np.set_printoptions(threshold=sys.maxsize)
            print("y: " + str(y))
            print("h: " + str(h))

        return (-y * np.log(h + self.epsilon) - (1 - y + self.epsilon) * np.log(1 - h + self.epsilon)).mean()

    def addIntercept(self, x):
        intercept = np.ones((x.shape[0], 1))
        return np.concatenate((intercept, x), axis=1)

    def fit(self, x, y):
        if self.fitIntercept:
            x = self.addIntercept(x)

        # Weight initialization
        theta = np.zeros(x.shape[1])

        # Apply Gradient Descent
        for i in range(self.iterations):
            z = np.dot(x, theta)
            h = self.sigmoid(z)
            gradient = np.dot(x.T, (h - y)) / y.size
            theta -= self.learningRate * gradient

            z = np.dot(x, theta)
            h = self.sigmoid(z)
            loss = self.loss(h, y)

            if self.verbose and i % 10000 == 0:
                print(f'loss: {loss} \t')

    def predictProbability(self, x):
        theta = np.zeros(x.shape[1])
        return self.sigmoid(np.dot(x, theta))

    def predict(self, x, threshold=0.5):
        return self.predictProbability(x) >= threshold


if __name__ == '__main__':
    try:
        np.set_printoptions(threshold=sys.maxsize)
        mnist = datasets.load_digits()
        model = LogisticRegression()
        X = mnist.data[:, :2]
        y = (mnist.target != 0)
        print(y)
        model.fit(X, y)
        preds = model.predict(X)
        print(preds.mean())
    except Exception as e:
        print(e)
