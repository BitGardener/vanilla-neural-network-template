import numpy as np
import matplotlib.pyplot as plt
from NeuralNet import NeuralNet
# from data import x_train, y_train, x_test, y_test
#
# from sklearn import preprocessing
#
# x_train = preprocessing.scale(x_train)

x_test = np.array([[0, 0],
              [0.2, 0],
              [0.5, 1],
              [0, 0.5]])

y_test = np.array([0, 1])



nn = NeuralNet(0.01, ["sigmoid", 5, "sigmoid", 2], x_test, y_test, yIsOneHot=False)
nn.forwardpropagation()
nn.backwardpropagation()


# for i in range(1, 11):
#     print("training ...")
#     nn.train(iterations)
#     accuracy, results = nn.getAccuracy()
#     print("Accuracy after ", i * iterations, " iterations: ", accuracy)
#     print("Cost after ", i * iterations, " iterations: ", nn.getCost())
