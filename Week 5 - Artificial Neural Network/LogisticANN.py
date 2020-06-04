import sys

import numpy as np
import sklearn
import torch
from torch import nn
import torchvision
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, iterations, learningRate):
        self.iterations = iterations
        self.learningRate = learningRate


if __name__ == '__main__':
    # Initialize Settings + Dataset
    np.set_printoptions(threshold=sys.maxsize)
    dataPath = "Data/"
    # mnist = fetch_openml('mnist_784', data_home=dataPath)
    mnist = sklearn.datasets.load_digits()  # Use for Debugging!
    print(nn.Sigmoid())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
