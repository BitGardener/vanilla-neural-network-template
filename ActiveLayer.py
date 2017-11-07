import numpy as np
from ActivationFunction import activationFunctions


class ActiveLayer:
    def __init__(self, size, inputSize, activationFunctionName):
        self.size = size
        self.inputSize = inputSize
        self.activationFunctionName = activationFunctionName
        self.activationFunction = activationFunctions[activationFunctionName]

        self.bias = np.float32(0) # bias can be initialized with 0, TEST IF TRUE
        self.weights = self.initializeWeights()
        self.output = None
        self.gradientAccumulator = None


    def initializeWeights(self):
        # TODO: initialize in different ways depending on activation function
        # return np.random.rand(self.size, self.inputSize)
        return np.ones((self.size, self.inputSize))



    def activate(self, x):
        z = np.dot(self.weights, x)
        self.output = self.activationFunction.function(z)

