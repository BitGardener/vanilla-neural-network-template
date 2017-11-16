import numpy as np
from ActivationFunction import activationFunctions


class ActiveLayer:
    def __init__(self, size, inputSize, activationFunctionName):
        self.size = size
        self.inputSize = inputSize
        self.activationFunctionName = activationFunctionName
        self.activationFunction = activationFunctions[activationFunctionName]

        self.bias = self.initializeBiases() #np.zeros([self.size, 1])  # TODO: bias can be initialized with 0, TEST IF TRUE
        self.weights = self.initializeWeights()
        self.z = 0
        self.output = None

        # d stands for derivative
        self.dz = None
        self.dw = None # gradients for the weights
        self.db = None # gradients for the biases
        self.da = None


    # TODO: initialize in different ways depending on activation function
    def initializeWeights(self):
        return np.random.randn(self.size, self.inputSize) * 0.01


    def initializeBiases(self):
        return np.random.randn(self.size, 1) * 0.01


    def activate(self, x):
        self.z = np.add(np.dot(self.weights, x), self.bias)
        self.output = self.activationFunction.function(self.z)
