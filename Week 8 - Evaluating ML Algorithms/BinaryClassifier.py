import numpy as np
import torch
import torchvision
from sklearn import metrics
from torch import nn
from torchvision.transforms import transforms
import matplotlib.pyplot as plt


class BinaryClassifier(nn.Module):
    def __init__(self, trainData, testData, batchSize=100, epochs=10, inputSize=784, outputSize=2, learningRate=0.001):
        super(BinaryClassifier, self).__init__()
        # Device to use
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

        # Training/Testing Data + Loaders
        self.trainData = trainData
        self.testData = testData
        self.batchSize = batchSize
        self.trainLoader = torch.utils.data.DataLoader(dataset=self.trainData,
                                                       batch_size=self.batchSize,
                                                       shuffle=True)

        self.testLoader = torch.utils.data.DataLoader(dataset=self.testData,
                                                      batch_size=self.batchSize,
                                                      shuffle=False)
        # Training/Model Info
        self.epochs = epochs
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.learningRate = learningRate

        # Linear Transformation
        self.linear = torch.nn.Linear(self.inputSize, self.outputSize)

        # Cross Entropy Loss
        self.criterion = nn.CrossEntropyLoss()
        # Stochastic Gradient Descent Optimization
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learningRate)

        # Model Evaluations
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.fMeasure = None
        self.FPR = None

    def forward(self, x):
        # Simple Ax + b
        outputs = self.linear(x)
        return outputs

    def trainModel(self):
        # Get total length of loader
        totalStep = len(self.trainLoader)
        # Iterate through specified epochs
        for epoch in range(self.epochs):
            for i, (images, labels) in enumerate(self.trainLoader):
                # Resize + Move to device
                images = images.reshape(-1, 28 * 28).to(self.device)

                # Move tensors to the configured device
                labels = labels.to(self.device)

                # Compute Loss
                outputs = self(images)
                loss = self.criterion(outputs, labels)

                # Backpropagation and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, self.epochs, i + 1, totalStep, loss.item()))

    def testModel(self):
        with torch.no_grad():
            # Values for calculations
            correct = 0
            total = 0
            truePositive = 0
            falseNegative = 0
            trueNegative = 0
            falsePositive = 0
            beta = 0.5
            self.predictions = []
            # Iterate through images + Labels
            for images, labels in self.testLoader:
                # Resize + Move to device
                images = images.reshape(-1, 28 * 28).to(self.device)
                labels = labels.to(self.device)

                # Predict
                outputs = self(images)

                # Take maximum probabilities of Data
                _, predicted = torch.max(outputs.data, 1)
                self.predictions = self.predictions + predicted.tolist()
                # Correct/Incorrect
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Turn labels + predicted into list
                labels = labels.tolist()
                predicted = predicted.tolist()
                for index in range(len(labels)):
                    if labels[index] == 0 and predicted[index] == 0:
                        # True Positive
                        truePositive += 1

                    if labels[index] == 1 and predicted[index] == 1:
                        # True Negative
                        trueNegative += 1

                    if labels[index] == 0 and predicted[index] == 1:
                        # False Positive
                        falsePositive += 1

                    if labels[index] == 1 and predicted[index] == 0:
                        # False Negative
                        falseNegative += 1

            # Calculate model evaluations
            self.accuracy = round(100 * correct / total, 2)
            self.precision = round(truePositive / (truePositive + falsePositive) * 100, 2)
            self.recall = round((truePositive / (truePositive + falseNegative)) * 100, 2)  # Sensitivity
            self.fMeasure = round(1 / ((beta * (1 / self.precision)) + ((1 - beta) * (1 / self.recall))), 2)
            self.FPR = round(falsePositive / (falsePositive + trueNegative), 2) * 100

    def printStats(self):
        # Print Stats of Model
        print('Accuracy of the network on the 10000 test images: {} %'.format(self.accuracy))
        print('Precision of the network on the 10000 test images: {} %'.format(self.precision))
        print('Recall of the network on the 10000 test images: {} %'.format(self.recall))
        print('F-Measure of the network on the 10000 test images: {} %'.format(self.fMeasure))
        print('FPR of the network on the 10000 test images: {} %'.format(self.FPR))

    def drawROC(self):
        # Data
        targetData = self.testData.targets
        predictedData = torch.FloatTensor(self.predictions)

        # Get points to plot
        fpr, tpr, thresholds = metrics.roc_curve(targetData, predictedData)
        print(metrics.auc(fpr, tpr))

        # Plot & Save
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig('ROC.png')
        plt.close()

    def drawPR(self):
        # Data
        targetData = self.testData.targets
        predictedData = torch.FloatTensor(self.predictions)

        # Get points to plot
        precision, recall, _ = metrics.precision_recall_curve(predictedData, targetData)

        # Plot & Save
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig('PR.png')
        plt.close()


if __name__ == '__main__':
    # MNIST Data
    trainData = torchvision.datasets.MNIST(root='Data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

    testData = torchvision.datasets.MNIST(root='Data/',
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=True)

    ''' Use to Train + Save Model if necessary
    
    trainData.targets = trainData.targets != 0
    testData.targets = testData.targets != 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNetwork(trainData=trainData, testData=testData).to(device)
    model.trainModel()
    torch.save(model.state_dict(), 'Models/binary.pt') '''

    # For Binary Classification
    trainData.targets = trainData.targets != 0
    testData.targets = testData.targets != 0

    # Load Model and Test
    model = BinaryClassifier(trainData=trainData, testData=testData)
    model.load_state_dict(torch.load('Models/binary.pt'))
    model.testModel()
    model.printStats()
    model.drawPR()
    model.drawROC()
