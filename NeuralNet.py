from ActiveLayer import ActiveLayer, activationFunctions
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
    def __init__(self, learningRate, layerInformation, x, y, x_test=None, y_test=None, yIsOneHot=True):
        self.x = x
        self.x_test = x_test
        self.m = self.x.shape[1]
        self.y = self.oneHotY(y, yIsOneHot)
        self.y_test = self.vectoriceY(y_test, yIsOneHot)
        self.yVectorized = self.vectoriceY(y, yIsOneHot)
        self.learningRate = learningRate
        self.activeLayers = self.initializeLayers(layerInformation)
        self.currentCost = None


    def train(self, iterations):
        for i in range(iterations):
            self.forwardpropagation()
            self.currentCost = self.costFunction()
            self.backwardpropagation()

            accuracy, results = self.getAccuracy()
            print("Cost after ", i, " iterations: ", self.currentCost, " | Accuracy training set: ", accuracy)
            print("Accuracy training set: ", accuracy)
            # accuracy, results = self.test()
            # print("Accuracy testing set: ", accuracy)
            # print()

            self.updateWeightsAndBiases()



            # optional
            # print("guckguck")
            # self.gradientChecking()


    def forwardpropagation(self):
        self.activeLayers[0].activate(self.x)

        for i in range(1, len(self.activeLayers)):
            self.activeLayers[i].activate(self.activeLayers[i - 1].output)


    # TODO: clear up backprop
    # TODO: add input layer to self.activeLayers (rename to self.layers) to simplify, only one for loop needed
    def backwardpropagation(self):
        # calculate gradients for output layer
        outputL = self.activeLayers[-1]
        outputL.dz = np.subtract(outputL.output, self.y)
        if (len(self.activeLayers) == 1):
            outputL.dw = outputL.dz.dot(self.x.T) / self.m
        else:
            outputL.dw = outputL.dz.dot(self.activeLayers[-2].output.T) / self.m
            self.activeLayers[-2].da = outputL.weights.T.dot(outputL.dz)

        # TODO: Gradients from backprop are twice as large as gradients from numeric gradient calculation
        outputL.db = np.sum(outputL.dz, axis=1, keepdims=True) / self.m

        for i in range(len(self.activeLayers) - 2, -1, -1):
            currentL = self.activeLayers[i]
            currentL.dz = currentL.da * currentL.activationFunction.derivative(currentL.z)
            currentL.db = currentL.dz.sum(axis=1, keepdims=True) / self.m

            if (i == 0):
                currentL.dw = currentL.dz.dot(self.x.T) / self.m
                break

            currentL.dw = currentL.dz.dot(self.activeLayers[i - 1].output.T) / self.m
            self.activeLayers[i - 1].da = currentL.weights.T.dot(currentL.dz)


    def updateWeightsAndBiases(self):
        for i, layer in enumerate(self.activeLayers):
            layer.weights -= self.learningRate * layer.dw
            layer.bias -= self.learningRate * layer.db


    def costFunction(self):
        # cross-entropy cost (good cost function for logistic regression)
        a = self.activeLayers[-1].output
        return np.sum(np.sum(-(np.multiply(self.y, np.log(a)) + np.multiply((1 - self.y), np.log(1 - a))))) / self.m


    def gradientChecking(self):
        numericGradientsW = []
        numericGradientsB = []

        e = np.float32(1e-4)

        for i, layer in enumerate(self.activeLayers):
            numericGradientsW.append(np.zeros(layer.weights.shape))
            numericGradientsB.append(np.zeros(layer.bias.shape))

            # calculate numericGradientsW for one layer
            for rowI in range(layer.weights.shape[0]):
                for colI in range(layer.weights.shape[1]):
                    temp = layer.weights[rowI, colI]

                    layer.weights[rowI, colI] += e
                    self.forwardpropagation()
                    loss1 = self.costFunction()

                    layer.weights[rowI, colI] = temp

                    layer.weights[rowI, colI] -= e
                    self.forwardpropagation()
                    loss2 = self.costFunction()

                    layer.weights[rowI, colI] = temp

                    gradient = (loss1 - loss2) / (2*e)
                    numericGradientsW[i][rowI, colI] = gradient

            # calculate numericGradientsB for one layer
            for rowI in range(layer.size):
                temp = layer.bias[rowI]

                layer.bias[rowI] += e
                self.forwardpropagation()
                loss1 = self.costFunction()

                layer.bias[rowI] = temp

                layer.bias[rowI] -= e
                self.forwardpropagation()
                loss2 = self.costFunction()

                layer.bias[rowI] = temp

                gradient = (loss1 - loss2) / (2*e)
                numericGradientsB[i][rowI] = gradient

        print("GRADIENT CHECKING")
        print("NUMERICAL | BACKPROP")
        for i, layer in enumerate(self.activeLayers):
            print("\nLAYER ", i)
            print("WEIGHTS")
            for rowI in range(np.minimum(5, self.activeLayers[i].size)):
                print(numericGradientsW[i][rowI, 1], " | ", layer.dw[rowI, 1])

            print("\nBIASES")
            for rowI in range(np.minimum(5, self.activeLayers[i].size)):
                print(numericGradientsB[i][rowI, 0], " | ", layer.db[rowI, 0])


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


    def vectoriceY(self, y, yIsOneHot):
        if (yIsOneHot and y is not None):
            return np.argmax(y, axis=0)
        else:
            return y


    def getAccuracy(self, yVectorized=None):
        # get accuracy in percent
        # also returns a bool vector of which predictions were right/wrong
        if yVectorized is None:
            yVectorized = self.yVectorized

        maxIndexes = self.activeLayers[-1].output.argmax(axis=0)
        results = np.equal(maxIndexes, yVectorized)
        rightPredictionsCount = np.count_nonzero(results)

        return (rightPredictionsCount / yVectorized.size) * 100, results


    def test(self):
        if self.x_test is None or self.y_test is None:
            print("No test data available")
            return None, None

        x = self.x
        self.x = self.x_test
        self.forwardpropagation()
        self.x = x
        return self.getAccuracy(yVectorized=self.y_test)

    def classifyExample(self, example):
        x = self.x
        self.x = example
        self.forwardpropagation()
        label = self.activeLayers[-1].output.argmax()
        self.x = x
        print(self.activeLayers[-1].output)
        return label


    def getCost(self):
        return self.currentCost


    def saveWeightsAndBiases(self):
        for i, layer in enumerate(self.activeLayers):
            fileName = "weightsLayer{0}.npy".format(i)
            np.save(fileName, layer.weights)

            fileName = "biasesLayer{0}.npy".format(i)
            np.save(fileName, layer.bias)


    def loadWeightsAndBiases(self):
        for i, layer in enumerate(self.activeLayers):
            fileName = "weightsLayer{0}.npy".format(i)
            layer.weights = np.load(fileName)

            fileName = "biasesLayer{0}.npy".format(i)
            layer.bias = np.load(fileName)


    def printLayer(self, layerIndex):
        layer = self.activeLayers[layerIndex]
        print("activationFunctionName: ", layer.activationFunctionName)
        print("layer size: ", layer.size)
        print("weights: ", layer.weights)
        print("weightsshape: ", np.shape(layer.weights))
        print("output: ", layer.output)
        print("----------------------")
