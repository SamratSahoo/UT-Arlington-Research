import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class NeuralNetwork(nn.Module):

    # NN Constructor
    def __init__(self, trainData, testData, inputSize=784, hiddenSize=500, classes=10, batchSize=100,
                 learningRate=0.001, epochs=10):
        super(NeuralNetwork, self).__init__()

        # Use GPU if possible
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Layer Sizes
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.classes = classes

        # Optimization Parameters
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.epochs = epochs

        # Fully Connected Layers + Rectified Linear Unit
        self.fc1 = nn.Linear(self.inputSize, self.hiddenSize)  # Input Layer --> Hidden Layer
        self.relu = nn.ReLU()  # ReLU to Process Data
        self.fc2 = nn.Linear(self.hiddenSize, self.classes)  # Hidden Layer ---> Output Layer

        # Data + Data Loaders
        self.trainData = trainData
        self.testData = testData
        self.trainLoader = torch.utils.data.DataLoader(dataset=self.trainData,
                                                       batch_size=self.batchSize,
                                                       shuffle=True)

        self.testLoader = torch.utils.data.DataLoader(dataset=self.testData,
                                                      batch_size=self.batchSize,
                                                      shuffle=False)
        # Optimizer - Stochastic Gradient Descent
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learningRate)

        # Cross Entropy Loss
        self.criterion = nn.CrossEntropyLoss()

        self.epsilon = 1e-5

    # Forward Propagation
    def forward(self, x):
        # Run through first FCL
        out = self.fc1(x)
        # Process through ReLU
        out = self.relu(out)
        # Run Through second FCL
        out = self.fc2(out)
        return out

    # Train Model
    def trainModel(self):
        totalStep = len(self.trainLoader)
        # Training Loop
        for epoch in range(self.epochs):
            # Gather Images + Labels
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
            correct = 0
            total = 0
            # Iterate through images + Labels
            for images, labels in self.testLoader:
                # Resize + Move to device
                images = images.reshape(-1, 28 * 28).to(self.device)
                labels = labels.to(self.device)

                # Predict
                outputs = self(images)

                # Take maximum probabilities of Data
                _, predicted = torch.max(outputs.data, 1)

                # Correct/Incorrect
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
    torch.save(model.state_dict(), 'Models/binary.pt') '''

    # Load Model and Test
    model = NeuralNetwork(trainData=trainData, testData=testData)
    model.load_state_dict(torch.load('Models/binary.pt'))
    model.testModel()
