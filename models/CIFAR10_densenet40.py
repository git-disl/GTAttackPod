from keras.layers import Input, Dense, Activation, Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.merge import concatenate
from keras.regularizers import l2
from keras.models import Model
import keras.backend as K
import os


def CIFAR10_densenet40(use_softmax=True, rel_path='./'):
    nb_classes = 10
    depth = 40
    nb_dense_block = 3
    growth_rate = 12
    nb_filter = 16
    activation = "softmax"

    # Determine proper input shape
    inputs = Input(shape=(32, 32, 3))

    x = __create_dense_net(nb_classes, inputs, use_softmax, depth, nb_dense_block,
                           growth_rate, nb_filter, -1, False, 0.0, 1E-4, activation)

    # Create model.
    model = Model(inputs, x, name='CIFAR10_densenet40')
    model.load_weights(os.path.join('%smodels/weights' % rel_path, "CIFAR10_densenet.keras_weights.h5"))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
    return model


def __transition_block(ip, nb_filter, compression=1.0, weight_decay=1E-4):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(ip)
    x = Activation('relu')(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer='he_uniform', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x


def __conv_block(ip, nb_filter, bottleneck=False, weight_decay=1E-4):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(ip)
    x = Activation('relu')(x)
    if bottleneck:
        inter_channel = nb_filter * 4  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua
        x = Conv2D(inter_channel, (1, 1), kernel_initializer='he_uniform', padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                               beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    return x


def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, weight_decay=1E-4,
                  grow_nb_filters=True, return_concat_list=False):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x_list = [x]
    for i in range(nb_layers):
        cb = __conv_block(x, growth_rate, bottleneck, weight_decay)
        x_list.append(cb)
        x = concatenate([x, cb], axis=concat_axis)
        if grow_nb_filters:
            nb_filter += growth_rate
    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter


def __create_dense_net(nb_classes, img_input, use_softmax, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1,
                       nb_layers_per_block=-1, bottleneck=False, reduction=0.0, weight_decay=1E-4,
                       activation='softmax'):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    assert (depth - 4) % 3 == 0, 'Depth must be 3 N + 4'
    if reduction != 0.0:
        assert 0.0 < reduction <= 1.0, 'reduction value must lie between 0.0 and 1.0'

    # layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == (nb_dense_block + 1), 'If list, nb_layer is used as provided. ' \
                                                       'Note that list size must be (nb_dense_block + 1)'
        final_nb_layer = nb_layers[-1]
        nb_layers = nb_layers[:-1]
    else:
        if nb_layers_per_block == -1:
            count = int((depth - 4) / 3)
            nb_layers = [count for _ in range(nb_dense_block)]
            final_nb_layer = count
        else:
            final_nb_layer = nb_layers_per_block
            nb_layers = [nb_layers_per_block] * nb_dense_block

    if bottleneck:
        nb_layers = [int(layer // 2) for layer in nb_layers]

    # compute initial nb_filter if -1, else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate

    # compute compression factor
    compression = 1.0 - reduction

    # Initial convolution
    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_uniform', padding='same', name='initial_conv2D',
               use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, bottleneck=bottleneck,
                                     weight_decay=weight_decay)
        # add transition_block
        x = __transition_block(x, nb_filter, compression=compression, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    # The last dense_block does not have a transition_block
    x, nb_filter = __dense_block(x, final_nb_layer, nb_filter, growth_rate, bottleneck=bottleneck,
                                 weight_decay=weight_decay)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(nb_classes, kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)
    if use_softmax:
        x = Activation(activation)(x)
    return x
