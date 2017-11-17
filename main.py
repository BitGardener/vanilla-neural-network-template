import numpy as np
import matplotlib.pyplot as plt
from NeuralNet import NeuralNet
from data import x_train, y_train, x_test, y_test
from sklearn import preprocessing
from PIL import Image

def loadDrawing():
    img = Image.open(r"C:\Users\bviehhauser\PycharmProjects\vanilla-neural-network-template\drawing_example.png")
    x = np.array(img)
    x = x[:, :, 0]
    return x
def analyseDrawingExamples(nn, x_for_scaling):
    while True:
        print("Analyzing ...")
        x = loadDrawing()
        plt.imshow(x)
        x = x.reshape(x.size, 1)
        x_for_scaling = np.append(x_for_scaling, x, axis=1)
        x = preprocessing.scale(x_for_scaling, axis=1)[:, -1].reshape(28*28, 1)
        label = nn.classifyExample(x)
        title = "Ich glaube das ist eine {0}!".format(label)
        plt.title(title, fontsize=20)
        plt.gray()
        plt.show()


x_for_scaling = x_train
x_train = preprocessing.scale(x_train, axis=1)
x_test = preprocessing.scale(x_test, axis=1)


nn = NeuralNet(0.3, ["relu", 256, "relu", 128, "sigmoid", 10], x_train, y_train, x_test, y_test, yIsOneHot=False)
nn.train(100)