import numpy as np


class ActivationFunction:
    def __init__(self, name, function, derivative):
        self.name = name
        self.function = function
        self.derivative = derivative


# dictionary of all available activation functions objects
# keys are there names
activationFunctions = {}

# ================================================
# RELU
def relu(z):
    return np.maximum(0, z)

def derivedRelu(z):
    return np.greater(z, 0) * 1

activationFunctions["relu"] = ActivationFunction("relu", relu, derivedRelu)


# =================================================
# SIGMOID
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def derivedSigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

activationFunctions["sigmoid"] = ActivationFunction("sigmoid", sigmoid, derivedSigmoid)
