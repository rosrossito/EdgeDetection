from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras
from tensorflow.keras import layers


def create_model():
    # for the moment number of filters is constant (not changed).
    #Todo:
    #1. Define regions with lines/edges (ROI where kernels will be applied). Can be standard algorithm.
    #2. Another kind of convolution. If to convolve just ROI, not all image, it will be not usual grid,
    # convolution can be not strictly vertical, diagonally or horisontal
    #3. The number of convolutional filters in the first layer should have possibility
    # dynamically be changed (increased). As a result all further layers  should be changed proportionally.
    # Theoretical basis: If new features are appeared (old ones are not activated) new feature is necessary
    # so we keep all small possible lines/edges/circumferences on the recognition basement. Obviously they will be used
    # in different images
    #4. Indepth convolution instead of pooling - add pros and cons. The number of kernels is taken approximately less
    # but maybe should be adjusted dynamically

    inputs = keras.Input((28, 28, 1))
    # decrease resolution/precision
    pool1 = layers.AveragePooling2D(pool_size=(2, 2))(inputs)

    # convolution (create new feature in the space)
    # Same padding - to preserve dimensionality and convolve bigger with smaller features
    # (another option - Valid padding means do not any zero to the input)
    # Choose 4 size kernels
    conv1 = layers.Conv2D(filters=120, kernel_size=(4, 4), padding='Same', activation='relu',
                          input_shape=(14, 14, 1))(pool1)
    # gather similar feature through the layers, decrease feature space.
    # by this, edges that are similar should be treated as the same
    conv2 = layers.Conv2D(filters=80, kernel_size=(1, 1), padding='Same', activation='relu',
                          input_shape=(14, 14, 120))(conv1)
    # convolution (create new feature in the space). One feature cover 8*8 area (2 excels of 4 pixels with stride 4)
    conv3 = layers.Conv2D(filters=1200, kernel_size=(2, 2), strides=(4, 4), padding='Same', activation='relu',
                          input_shape=(14, 14, 80))(conv2)
    # gather similar feature through the layers, decrease feature space.
    # by this, edges that are similar should be treated as the same
    # used for generalization instead of pooling layer
    conv4 = layers.Conv2D(filters=800, kernel_size=(1, 1), padding='Same', activation='relu',
                              input_shape=(14, 14, 1200))(conv3)
    # convolution (create new feature in the space). One feature cover 16*16 area (2 excels of 8 pixels with stride 8)
    conv5 = layers.Conv2D(filters=12000, kernel_size=(2, 2), strides=(8, 8), padding='Same', activation='relu',
                          input_shape=(14, 14, 800))(conv4)
    # gather similar feature through the layers, decrease feature space.
    # by this, edges that are similar should be treated as the same
    conv6 = layers.Conv2D(filters=8000, kernel_size=(1, 1), padding='Same', activation='relu',
                          input_shape=(14, 14, 12000))(conv5)

    flatten = layers.Flatten()(conv6)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    #last layer. Here 10 digits
    dense2 = layers.Dense(10, activation="softmax")(dense1)

    model = keras.Model(inputs, dense2)

    # Define the optimizer
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # Compile the model
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # summarize the model
    model.summary()

    return model