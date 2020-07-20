import numpy as np
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt


class PCANumpy:

    def __init__(self, data):
        self.data = data
        self.center = self.getCenter()
        self.covarianceMatrix = self.getCovarianceMatrix()
        self.eigenvector = self.getEigenvector()

    # Center the matrix
    def getCenter(self):
        center = self.data - np.mean(self.data, axis=0)
        return center

    # Get Covariance Matrix (Sum of squares)
    def getCovarianceMatrix(self):
        return np.cov(np.transpose(self.center))

    # Calculate Eigenvector
    def getEigenvector(self):
        regular, normalized = np.linalg.eig(self.covarianceMatrix)
        return normalized

    # Project the data onto PC Space
    def projectPCA(self):
        return np.dot(self.data, self.eigenvector)[:, :2]


if __name__ == '__main__':
    range = np.random.RandomState(1)
    # Load Data
    data = np.dot(range.rand(2, 2), range.randn(2, 200)).T
    plt.scatter(data[:, 0], data[:, 1])
    # Show Data
    plt.axis('equal')
    plt.savefig('PCAData.png')


    # Numpy PCA
    pca = PCANumpy(data=data)
    # Print projected PCA
    print('Numpy PCA: ')
    print(pca.projectPCA())
    print('\n============\n')

    # Scikitlearn PCA
    pcaSklearn = IncrementalPCA(n_components=2, batch_size=10)
    newSklearnPCA = pcaSklearn.fit_transform(data)
    print('Scikit-learn PCA: ')
    print(newSklearnPCA)

    # Distance between matrices
    print('\n============\n')
    print("Distance Between Matrices:")
    print(np.linalg.norm(pca.projectPCA() - newSklearnPCA))