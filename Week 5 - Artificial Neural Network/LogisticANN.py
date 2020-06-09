import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class NeuralNetwork(nn.Module):

    def __init__(self, trainData, testData, inputSize=784, hiddenSize=500, classes=10, batchSize=100,
                 learningRate=0.001, epochs=10):
        super(NeuralNetwork, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.classes = classes
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.epochs = epochs
        self.fc1 = nn.Linear(self.inputSize, self.hiddenSize)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hiddenSize, self.classes)
        self.trainData = trainData
        self.testData = testData
        self.trainLoader = torch.utils.data.DataLoader(dataset=self.trainData,
                                                       batch_size=self.batchSize,
                                                       shuffle=True)

        self.testLoader = torch.utils.data.DataLoader(dataset=self.testData,
                                                      batch_size=self.batchSize,
                                                      shuffle=False)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def trainModel(self):
        totalStep = len(model.trainLoader)
        for epoch in range(self.epochs):
            for i, (images, labels) in enumerate(self.trainLoader):
                # Move tensors to the configured device
                images = images.reshape(-1, 28 * 28).to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = model(images)
                loss = self.criterion(outputs, labels)

                # Backpropagation and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, model.epochs, i + 1, totalStep, loss.item()))

    def testModel(self):
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.testLoader:
                images = images.reshape(-1, 28 * 28).to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNetwork(trainData=trainData, testData=testData).to(device)
    model.trainModel()
    torch.save(model.state_dict(), 'Models/model.pt') '''

    # Load Model and Test
    model = NeuralNetwork(trainData=trainData, testData=testData)
    model.load_state_dict(torch.load('Models/model.pt'))
    model.testModel()
