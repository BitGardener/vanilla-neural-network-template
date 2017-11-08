# unzips and converts data for neural net
import gzip
import os.path as p
import numpy as np


# startIndex sets how many values should be skipped
# in the raw byte stream
# ( skipps metadata )
def unzipData(filename, startIndex):
    with gzip.open(filename) as file:
        data = file.read()

        return list(data[startIndex:])


dir = p.dirname(__file__)
x_train = unzipData(p.join(dir, ".\\train-images-idx3-ubyte.gz"), 16)
y_train = unzipData(p.join(dir, ".\\train-labels-idx1-ubyte.gz"), 8)
x_test  = unzipData(p.join(dir, ".\\t10k-images-idx3-ubyte.gz"), 16)
y_test  = unzipData(p.join(dir, ".\\t10k-labels-idx1-ubyte.gz"), 8)

x_train = np.array(x_train).reshape((60000, 28 * 28))
x_train = x_train.T
y_train = np.array(y_train).reshape((1, 60000))
x_test  = np.array(x_test).reshape((10000, 28 * 28))
x_test  = x_test.T
y_test  = np.array(y_test).reshape((1, 10000))
