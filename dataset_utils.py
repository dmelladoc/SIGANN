import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from Cifar10 import *
import gzip
from sklearn.utils import shuffle

class Dataset(object):
    """
    Abstract Class para manejar los datasets
    """
    def __init__(self, dataset, datapath="/tmp/dataset"):
        assert dataset in ["MNIST", "CIFAR10", "CIFAR100", "EMNIST"]
        self.name = dataset
        self.n_classes = None
        self.X_shape = None
        self.Xtrain = None
        self.Ytrain = None
        self.Xtest = None
        self.Ytest = None

    def split_section(self, X, Y, clases):
        """
        Entrega un segmento de las clases a utilizar
        """
        idx = np.concatenate([np.argwhere(Y == c) for c in clases], 0).squeeze()
        return X[idx], Y[idx]

    def load_segment_of_data(self, clases, kind="train"):
        return self.split_section(self.Xtrain, self.Ytrain, clases) if kind == "train" else self.split_section(self.Xtest, self.Ytest, clases)
    
    def load_percent_of_data(self, clases, percent=1.0, kind="train"):
        X, Y = self.load_segment_of_data(clases, kind)
        if percent >= 1.0:
            return X, Y
        else:
            n_data = int(len(X)*percent)
            X, Y = shuffle(X, Y, n_samples=n_data)
            return X, Y


class MNIST(Dataset):
    def __init__(self, datapath="/tmp/mnist/"):
        super(MNIST, self).__init__("MNIST", datapath)
        mnist = input_data.read_data_sets(datapath, one_hot=False)
        self.Xtrain = mnist.train.images.reshape(-1, 28, 28, 1)
        self.Ytrain = mnist.train.labels
        self.Xtest = mnist.test.images.reshape(-1, 28, 28, 1)
        self.Ytest = mnist.test.labels
        self.X_shape = self.Xtrain.shape[1:]
        self.n_classes = 10

class EMNIST(Dataset):
    def __init__(self, datapath="EMNIST/"):
        super(EMNIST, self).__init__("EMNIST", datapath)
        self.Xtrain = self._read_images(datapath + "emnist-balanced-train-images-idx3-ubyte.gz")
        self.Xtest = self._read_images(datapath + "emnist-balanced-test-images-idx3-ubyte.gz")
        self.Ytrain = self._read_labels(datapath + "emnist-balanced-train-labels-idx1-ubyte.gz")
        self.Ytest = self._read_labels(datapath + "emnist-balanced-test-labels-idx1-ubyte.gz")
        self.X_shape = self.Xtrain.shape[1:]
        self.n_classes = 47

    def _read32(self, bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)[0]

    def _read_labels(self, label_file):
        with gzip.open(label_file,'rb') as flbl:
            magic = self._read32(flbl)
            assert magic == 2049
            num = self._read32(flbl)
            buffer = flbl.read(num)
            return np.frombuffer(buffer,dtype=np.uint8)

    def _read_images(self,img_file):
        with gzip.open(img_file,'rb') as fimg:
            magic = self._read32(fimg)
            assert magic == 2051
            num = self._read32(fimg)
            row = self._read32(fimg)
            col = self._read32(fimg)
            buffer = fimg.read(num*row*col)
            images = np.frombuffer(buffer,dtype=np.uint8).reshape(num,row,col,1).swapaxes(1,2)
            return (images/255).astype(np.float32)


class CIFAR10(Dataset):
    def __init__(self, datapath="/tmp/cifar10/"):
        super(CIFAR10, self).__init__("CIFAR10", datapath)
        (self.Xtrain, self.Ytrain), (self.Xtest, self.Ytest) = load_data(datapath)
        self.X_shape = self.Xtrain.shape[1:]
        self.n_classes = 10
