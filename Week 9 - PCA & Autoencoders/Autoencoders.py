import os

import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


class Autoencoder(nn.Module):
    def __init__(self, epochs=100, batchSize=128, learningRate=1e-3):
        super(Autoencoder, self).__init__()
        # Encoder Network
        self.encoder = nn.Sequential(nn.Linear(784, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, 12),
                                     nn.ReLU(True),
                                     nn.Linear(12, 3))
        # Decoder Network
        self.decoder = nn.Sequential(nn.Linear(3, 12),
                                     nn.ReLU(True),
                                     nn.Linear(12, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 784),
                                     nn.Tanh())

        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate

        # Data + Data Loaders
        self.imageTransforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.data = MNIST('./Data', transform=self.imageTransforms)
        self.dataLoader = torch.utils.data.DataLoader(dataset=self.data,
                                                      batch_size=self.batchSize,
                                                      shuffle=True)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def trainModel(self):
        for epoch in range(self.epochs):
            for data in self.dataLoader:
                image, _ = data
                image = image.view(image.size(0), -1)
                image = Variable(image)
                # Predictions
                output = self(image)
                # Calculate Loss
                loss = self.criterion(output, image)
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, self.epochs, loss.data))

    def testImage(self, indexSample):
        sample, _ = self.data[indexSample]
        sample = sample.view(sample.size(0), -1)
        sample = Variable(sample)
        return self(sample)


if __name__ == '__main__':
    # Initialize Autoencoder
    model = Autoencoder()
    # Load Model
    model.load_state_dict(torch.load('Model/Autoencoder.pth'))
    # Choose an image
    tensor = model.testImage(7777)
    # Reshape
    tensor = torch.reshape(tensor, (28, 28))
    # toImage function
    toImage = torchvision.transforms.ToPILImage()
    # Convert to image
    image = toImage(tensor)
    # Save image
    image.save('After.png')
