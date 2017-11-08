from ActiveLayer import ActiveLayer
import numpy as np


# 'x' stores the training data as a matrix
#   one training example should be stored in a column
#
# 'y' stores the training labels
#   labels are either expected as one-hot,
#   where the labels for one example is in a column
#   or as a column vector with the labels starting with 0,
#   which will get converted to a one-hot array
#   ADD INITIALIZING PARAMETER 'yIsOneHot=False" THEN


class NeuralNet:
    def __init__(self, learningRate, layerInformation, x, y, yIsOneHot=True):
        self.x = x
        self.m = self.x.shape[1]
        self.y = self.oneHotY(y, yIsOneHot)
        self.learningRate = learningRate
        self.activeLayers = self.initializeLayers(layerInformation)
        self.currentCost = np.float32(0)


    def train(self, iterations):
        for i in range(iterations):
            self.forwardpropagation()
            self.backwardpropagation()
            print("Cost after ", i, " iterations: ", self.currentCost)


    def forwardpropagation(self):
        self.activeLayers[0].activate(self.x)

        for i in range(1, len(self.activeLayers)):
            self.activeLayers[i].activate(self.activeLayers[i - 1].output)


    def backwardpropagation(self):
        self.calculateErrors()
        self.updateWeights()


    def calculateErrors(self):
        self.activeLayers[-1].error = np.subtract(self.activeLayers[-1].output, self.y)
        self.currentCost = np.sum(self.activeLayers[-1].error) / self.m

        for i in range(len(self.activeLayers) - 2, -1, -1):
            layer = self.activeLayers[i]
            w = self.activeLayers[i + 1].weights

            wTimesE = np.dot(w.T, self.activeLayers[i + 1].error)
            layer.error = wTimesE * layer.activationFunction.derivative(layer.z)


    def updateWeights(self):
        firstActLayer = self.activeLayers[0]
        gradient = np.dot(firstActLayer.error, self.x.T)
        firstActLayer.weights -= self.learningRate * gradient / self.m

        for i in range(1, len(self.activeLayers)):
            layer = self.activeLayers[i]
            gradient = np.dot(layer.error, self.activeLayers[i - 1].output.T)
            layer.weights -= self.learningRate * gradient / self.m

        #  TODO: update BIAS as well


    def initializeLayers(self, layerInformation):
        l = len(layerInformation)
        if (l == 0 or l % 2 == 1):
            return None

        activeLayers = []
        activeLayers.append(ActiveLayer(layerInformation[1],
                               len(self.x),
                               layerInformation[0]))

        for i in range(2, len(layerInformation), 2):
            activeLayers.append(ActiveLayer(layerInformation[i + 1],
                                activeLayers[-1].size,
                                layerInformation[i]))

        return activeLayers


    def oneHotY(self, y, yIsOneHot):
        if (yIsOneHot):
            return y
        else:
            size = np.size(y)
            oneHotY = np.zeros([np.max(y) + 1, size])
            oneHotY[y, np.arange(size)] = 1
            return oneHotY


    def printLayer(self, layerIndex):
        layer = self.activeLayers[layerIndex]
        print("activationFunctionName: ", layer.activationFunctionName)
        print("layer size: ", layer.size)
        print("weights: ", layer.weights)
        print("weightsshape: ", np.shape(layer.weights))
        print("output: ", layer.output)
        print("----------------------")


x = np.array([[0,1,2],
              [2,3,4]])

y = np.array([[0,2,1]])
nn = NeuralNet(0, ["sigmoid", 10, "sigmoid", 3], x, y, yIsOneHot=False)
nn.forwardpropagation()
nn.backwardpropagation()

# print("y: ", nn.y)
# print("m: ", nn.m)
# print("predictions", nn.activeLayers[-1].output)