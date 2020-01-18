from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda
from keras.layers import MaxPooling2D, Conv2D
import os


def MNIST_carlini(use_softmax=True, rel_path='./'):
    # Define neural architecture
    model = Sequential()
    model.add(Lambda(lambda x: x - 0.5, input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32 * 2, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32 * 2, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(10))
    if use_softmax:
        model.add(Activation('softmax'))

    # Load pre-trained weights
    model.load_weights(os.path.join('%smodels/weights' % rel_path, "MNIST_carlini.keras_weights.h5"))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
    model.name = 'MNIST_carlini'
    return model
