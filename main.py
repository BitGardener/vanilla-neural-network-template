import numpy as np
import matplotlib.pyplot as plt
from NeuralNet import NeuralNet
from data import x_train, y_train, x_test, y_test


# nn = NeuralNet(0.03, ["sigmoid", 16, "sigmoid", 16, "sigmoid", 10], x_test, y_test, yIsOneHot=False)
# nn.train(1000)

sample = x_train[:, 1]
sample = sample.reshape([28,28])

print(sample)
imgplot = plt.imshow(sample)
plt.gray()
plt.show()