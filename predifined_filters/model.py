from keras.layers import *
from keras.models import *
from keras.optimizers import *

from create_kernels import create_edge_kernels, create_angle_kernels
from soccer.utils import create_edge_kernels_for_soccer

non_trainable_layers = []
layers_of_size = {}
layers_of_size_corners = {}


def create_edge_layer(shape, edge, kernel, inputs):
    layer_name = 'conv_edge' + str(shape) + "_" + str(edge)
    layer = Conv2D(1, (shape, shape), activation='relu', padding='same', trainable=False, use_bias=False, name=layer_name)(
        inputs)
    non_trainable_layers.append((layer_name, shape, kernel))
    return layer


def create_angle_layer(shape, edge, angles_degree, kernel, inputs):
    layer_name = 'conv_angle' + str(shape) + "_" + str(edge) + "_" + str(angles_degree)
    layer = Conv2D(1, shape, activation='relu', padding='same', trainable=False, use_bias=False, name=layer_name)(
        inputs)
    non_trainable_layers.append((layer_name, shape, kernel))
    return layer


def net(pretrained_weights=None, input_size=(256, 256, 1)):
    # kernelEdgeBank = create_edge_kernels()
    # kernelAngleBank = create_angle_kernels(kernelEdgeBank)
    kernelEdgeBank = create_edge_kernels_for_soccer()


    inputs = Input(input_size)

    for key in kernelEdgeBank:
        layers = []
        for edge, kernel in kernelEdgeBank[key]:
            layers.append(create_edge_layer(key, edge, kernel, inputs))
        layers_of_size[key] = layers

    # layers for angles will be uncommented later
    # for key in kernelAngleBank:
    #     layers = []
    #     for edge, angles_degree, kernel in kernelAngleBank[key]:
    #         layers.append(create_angle_layer(key, edge, angles_degree, kernel, inputs))
    #     layers_of_size_corners[key] = layers

    # test_edge, test_kernel = kernelEdgeBank[SIZES[0]][0]

    # inputs = Input(input_size)

    # test_kernel = tf.convert_to_tensor(test_kernel, dtype=tf.float32)
    # test_kernel = tf.expand_dims(test_kernel, 0)
    # test_kernel = tf.expand_dims(test_kernel, 0)
    # conv1 = tf.nn.relu(tf.nn.covn2d(data, test_kernel, strides=[1, 1, 1, 1], padding='SAME'))

    # conv1 = Conv2D(1, 3, activation='relu', padding='same', trainable=False, use_bias=False, name='conv1')(inputs)

    # conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    # conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    # conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    # conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    # conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    # drop4 = Dropout(0.5)(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    #
    # conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    # conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    # drop5 = Dropout(0.5)(conv5)
    #
    # up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    # merge6 = concatenate([drop4,up6], axis = 3)
    # conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    # conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    #
    # up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    # merge7 = concatenate([conv3,up7], axis = 3)
    # conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    # conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    #
    # up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    # merge8 = concatenate([conv2,up8], axis = 3)
    # conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    # conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    #
    # up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    # merge9 = concatenate([conv1,up9], axis = 3)
    # conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    # conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    # conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    # conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    # temporary decision to set all weights to non trainable layers
    total_temp = []
    for key in kernelEdgeBank:
        # total_temp = total_temp + layers_of_size[key] + layers_of_size_corners[key]
        total_temp = total_temp + layers_of_size[key]
    conv_edge_output = concatenate(total_temp, axis=3, name="concat")

    # conv_output = tf.keras.layers.Concatenate(axis=1)([layers_of_size[3][0], layers_of_size[3][1]])
    # conv_output = tf.keras.layers.Concatenate(axis=1)(layers_of_size[3])

    model = Model(input=inputs, output=conv_edge_output)

    # model.get_layer("conv1").set_weights([np.reshape(test_kernel, (3, 3, 1, 1))])
    for layer_name, shape, kernel in non_trainable_layers:
        model.get_layer(layer_name).set_weights([np.reshape(kernel, (shape, shape, 1, 1))])

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
