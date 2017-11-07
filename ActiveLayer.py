import numpy as np
from ActivationFunction import activationFunctions

class ActiveLayer:
    def __init__(self, size, xLength, activationFunctionName):
        self.size = size
        self.xLength = xLength
        self.activationFunctionName = activationFunctionName
        self.activationFunction = activationFunctions[activationFunctionName]

        self.bias = np.float32(0) # bias can be initialized with 0, TEST IF TRUE
        self.weights = self.initializeWeights()
        self.output = None
        self.gradientAccumulator = None


    def initializeWeights(self):
        # TODO initialize in different ways depending on activation function
        return np.random.rand(self.size, self.xLength)


    def activate(self, x):
        z = np.dot(self.weights, x)
        self.output = self.activationFunction.function(z)

