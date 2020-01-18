from keras.datasets import cifar10
from keras.utils import np_utils


class CIFAR10Dataset:
    def __init__(self):
        self.name = "CIFAR10"
        self.image_size = 32
        self.num_channels = 3
        self.num_classes = 10

    def get_test_dataset(self):
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        X_test = X_test.reshape(X_test.shape[0], self.image_size, self.image_size, self.num_channels)
        X_test = X_test.astype('float32')
        X_test /= 255
        Y_test = np_utils.to_categorical(y_test, self.num_classes)
        del X_train, y_train
        return X_test, Y_test

    def get_val_dataset(self):
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        val_size = 5000
        X_val = X_train[:val_size]
        X_val = X_val.reshape(X_val.shape[0], self.image_size, self.image_size, self.num_channels)
        X_val = X_val.astype('float32') / 255
        y_val = y_train[:val_size]
        Y_val = np_utils.to_categorical(y_val, self.num_classes)
        del X_train, y_train, X_test, y_test

        return X_val, Y_val
