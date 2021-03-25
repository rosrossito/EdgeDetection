import math

from keras.layers import *
from keras.models import *
from keras.optimizers import *

from rect_lookup.custom_relu import Custom_relu
from soccer.utils import create_edge_kernels_for_soccer_with_parameters

non_trainable_layers = []


def rectnet(input_size):
    inputs = Input(input_size)
    pool1 = AveragePooling2D(pool_size=(2, 2))(inputs)
    act1 = Activation(restricted_step)(pool1)
    pool2 = AveragePooling2D(pool_size=(2, 2))(act1)
    act2 = Activation(restricted_step)(pool2)
    pool3 = AveragePooling2D(pool_size=(2, 2))(act2)
    act3 = Activation(restricted_step)(pool3)
    pool4 = AveragePooling2D(pool_size=(2, 2))(act3)
    act4 = Activation(binary_step)(pool4)
    # up1 = UpSampling2D(size=(2, 2))(act3)
    # up2 = UpSampling2D(size=(2, 2))(up1)
    # up3 = UpSampling2D(size=(2, 2))(up2)

    # output_layer = Conv2D(1, (3, 3), activation='relu', padding='same', trainable=False, use_bias=False, name="conv_1")(
    #     inputs)

    layers_of_size, minimal_kernel_size = create_edge_layers(pool4.shape[1:3], act4)
    total_temp = []
    for key in layers_of_size:
        total_temp = total_temp + layers_of_size[key]
    conv_edge_output = concatenate(total_temp, axis=3, name="concat")
    # final_pool = MaxPooling2D(pool_size=(pool4.shape[1:3][0], pool4.shape[1:3][1]))(conv_edge_output)

    model = Model(input=inputs, output=conv_edge_output)

    for layer_name, shape, kernel in non_trainable_layers:
        model.get_layer(layer_name).set_weights([np.reshape(kernel, (shape, shape, 1, 1))])

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model, act4, minimal_kernel_size


def create_edge_layers(input_size, act4):
    layers_of_size = {}
    kernelEdgeBank, minimal_kernel_size = create_edge_kernels_for_soccer_with_parameters(input_size[0], input_size[1])
    for key in kernelEdgeBank:
        # revert this condition later
        # if key == 27:
        layers = []
        for edge, kernel in kernelEdgeBank[key]:
            # revert this condition later
            # if edge == 63 or edge == 64:
            layers.append(create_edge_layer(key, edge, kernel, act4))
        layers_of_size[key] = layers
    return layers_of_size, minimal_kernel_size


def create_edge_layer(shape, edge, kernel, inputs):
    pad = int(math.floor(shape/2))
    layer_name = 'conv_edge' + str(shape) + "_" + str(edge)
    padded_inputs = ZeroPadding2D(padding=((pad, pad), (pad, pad)))(inputs)
    layer = Conv2D(1, (shape, shape), activation=Custom_relu(shape=shape), padding='valid', trainable=False, use_bias=False,
                   name=layer_name)(padded_inputs)
    non_trainable_layers.append((layer_name, shape, kernel))
    return layer


# def get_kernel():
#     return np.ones((3, 3), dtype=int)/9

def restricted_step(x):
    # return x
    return K.switch(K.less(x, 0), K.minimum(x, -1), K.maximum(x, 1))


def binary_step(x):
    # return x
    return K.switch(K.less(x, 0), K.maximum(x, 0), K.maximum(x, 1))


def custom_relu(x):
    return K.relu(K.abs(x))

# def binary_step(x):
#     if x<0:
#         return 0
#     else:
#         return 1
