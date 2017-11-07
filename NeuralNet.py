from ActiveLayer import ActiveLayer
import numpy as np


class NeuralNet:
    def __init__(self, x, y, learningRate, layerInformation):
        self.x = x
        self.y = y # TODO: convert y to one hot array
        self.learningRate = learningRate
        self.activeLayers = self.initializeLayers(layerInformation)
        print(len(self.activeLayers))


    def forwardpropagation(self):
        self.activeLayers[0].activate(self.x)

        for i in range(1, len(self.activeLayers)):
            self.activeLayers[i].activate(self.activeLayers[i - 1].output)


    def backwardpropagation(self):
        print("not implemented")


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

    def printLayer(self, layerIndex):
        layer = self.activeLayers[layerIndex]
        print("activationFunctionName: ", layer.activationFunctionName)
        print("layer size: ", layer.size)
        print("weights: ", layer.weights)
        print("weightsshape: ", np.shape(layer.weights))
        print("output: ", layer.output)
        print("----------------------")

