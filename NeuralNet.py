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
    def __init__(self, learningRate, layerInformation, x, y, yIsOneHot=True):
        self.x = x
        self.m = self.x.shape[1]
        self.y = self.oneHotY(y, yIsOneHot)
        self.yVectorized = self.vectoriceY(y, yIsOneHot)
        self.learningRate = learningRate
        self.activeLayers = self.initializeLayers(layerInformation)
        self.currentCost = np.float32(0)


    def train(self, iterations):
        for i in range(iterations):
            self.forwardpropagation()
            self.backwardpropagation()
            #print("Cost after ", i, " iterations: ", self.currentCost)


    def forwardpropagation(self):
        self.activeLayers[0].activate(self.x)

        for i in range(1, len(self.activeLayers)):
            self.activeLayers[i].activate(self.activeLayers[i - 1].output)


    def backwardpropagation(self):
        self.calculateErrors()
        gradientsW, gradientsB = self.calculateGradients()
        # optional
        self.gradientChecking(gradientsW, gradientsB)


        self.updateWeightsAndBiases(gradientsW, gradientsB)
        # TODO: implement gradient checking boy


    def calculateErrors(self):
        self.activeLayers[-1].error = np.subtract(self.y, self.activeLayers[-1].output)

        for i in range(len(self.activeLayers) - 2, -1, -1):
            layer = self.activeLayers[i]
            w = self.activeLayers[i + 1].weights

            wTimesE = np.dot(w.T, self.activeLayers[i + 1].error)
            layer.error = wTimesE * layer.activationFunction.derivative(layer.z)


    def calculateGradients(self):
        gradientsW = []
        gradientsB = []
        firstActLayer = self.activeLayers[0]
        gradientsW.append(np.dot(firstActLayer.error, self.x.T) / self.m)
        gradientsB.append(np.sum(firstActLayer.error, axis=1, keepdims=True) / self.m)

        for i in range(1, len(self.activeLayers)):
            layer = self.activeLayers[i]
            gradientsW.append(np.dot(layer.error, self.activeLayers[i - 1].output.T) / self.m)
            # TODO: is bias gradient calculation implemented correctly?
            gradientsB.append(np.sum(layer.error, axis=1, keepdims=True) / self.m)

        return gradientsW, gradientsB


    def updateWeightsAndBiases(self, gradientsW, gradientsB):
        for i, layer in enumerate(self.activeLayers):
            layer.weights -= self.learningRate * gradientsW[i]
            layer.bias -= self.learningRate * gradientsB[i]

    def costFunction(self):
        # cross-entropy cost (good cost function for logistic regression)
        a = self.activeLayers[-1].output
        return np.sum(np.sum(-(np.multiply(self.y, np.log(a)) + np.multiply((1 - self.y), np.log(1 - a))))) / self.m



    def gradientChecking(self, backpropGradientsW, backpropGradientsB):
        numericGradientsW = []
        numericGradientsB = []

        e = np.float32(1e-4)

        for layI, layer in enumerate(self.activeLayers):
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
                    numericGradientsW[layI][rowI, colI] = gradient

            # calculate numericGradientsB for one layer
            for i in range(layer.size):
                temp = layer.bias[i]

                layer.bias[i] += e
                self.forwardpropagation()
                loss1 = self.costFunction()

                layer.bias[i] = temp

                layer.bias[i] -= e
                self.forwardpropagation()
                loss2 = self.costFunction()

                layer.bias[i] = temp

                gradient = (loss1 - loss2) / (2*e)
                numericGradientsB[layI][i] = gradient

        print("GRADIENT CHECKING")
        print("NUMERICAL | BACKPROP")
        for layI in range(len(self.activeLayers)):
            print("\nLAYER ", layI)
            print("WEIGHTS")
            for rowI in range(np.minimum(5, self.activeLayers[layI].size)):
                print(numericGradientsW[layI][rowI, 0], " | ", backpropGradientsW[layI][rowI, 0])

            print("\nBIASES")
            for rowI in range(np.minimum(5, self.activeLayers[layI].size)):
                print(numericGradientsB[layI][rowI, 0], " | ", backpropGradientsB[layI][rowI, 0])



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
        if (yIsOneHot):
            return np.argmax(y, axis=0)
        else:
            return y


    def getAccuracy(self):
        # get accuracy in percent
        # also returns a bool vector of which predictions were right/wrong
        maxIndexes = self.activeLayers[-1].output.argmax(axis=0)
        results = np.equal(maxIndexes, self.yVectorized)
        rightPredictionsCount = np.count_nonzero(results)

        return (rightPredictionsCount / self.m) * 100, results

    def getCost(self):
        return self.currentCost


    def printLayer(self, layerIndex):
        layer = self.activeLayers[layerIndex]
        print("activationFunctionName: ", layer.activationFunctionName)
        print("layer size: ", layer.size)
        print("weights: ", layer.weights)
        print("weightsshape: ", np.shape(layer.weights))
        print("output: ", layer.output)
        print("----------------------")



