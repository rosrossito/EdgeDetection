from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop


def create_model():
    # 1 iteration
    # loss: 0.0032 - accuracy: 0.9951 - val_loss: 0.0176 - val_accuracy: 0.9814
    # Trainable params: 123,082,882


    # 2 iteration (dilated convolution) (eve_model.h5)
    # loss: 0.0039 - accuracy: 0.9954 - val_loss: 0.0170 - val_accuracy: 0.9860
    # Trainable params: 29,924,730

    # 3 iteration (tanh) (eve_model_tanh.h5)
    # loss: 0.0215 - accuracy: 0.9644 - val_loss: 0.0295 - val_accuracy: 0.9564
    # Trainable params: 28,711,090


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
    # pool1 = layers.AveragePooling2D(pool_size=(2, 2))(inputs)

    # convolution (create new feature in the space)
    # Same padding - to preserve dimensionality and convolve bigger with smaller features
    # (another option - Valid padding means do not any zero to the input)
    conv1 = layers.Conv2D(filters=50, kernel_size=(3, 3), padding='valid', activation='tanh',
                          input_shape=(28, 28, 1))(inputs)
    # gather similar feature through the layers, decrease feature space.
    # by this, edges that are similar should be treated as the same
    conv2 = layers.Conv2D(filters=30, kernel_size=(1, 1), padding='Same', activation='tanh',
                          input_shape=(26, 26, 50))(conv1)
    # convolution (create new feature in the space). One feature cover 5*5 area (2 excels of 3 pixels with dilation 2)
    conv3 = layers.Conv2D(filters=1200, kernel_size=(2, 2), strides=(1, 1), padding='valid',
                                   dilation_rate=(2, 2), activation='tanh', input_shape=(26, 26, 30))(conv2)
    # gather similar feature through the layers, decrease feature space.
    # by this, edges that are similar should be treated as the same
    # used for generalization instead of pooling layer
    conv4 = layers.Conv2D(filters=600, kernel_size=(1, 1), padding='Same', activation='tanh',
                              input_shape=(24, 24, 1200))(conv3)
    # convolution (create new feature in the space). One feature cover 9*9 area (2 excels of 5 pixels with dilation 4)
    conv5 = layers.Conv2D(filters=3000, kernel_size=(2, 2), strides=(1, 1), padding='valid',
                                   dilation_rate=(4, 4), activation='tanh', input_shape=(24, 24, 600))(conv4)
    # gather similar feature through the layers, decrease feature space.
    # by this, edges that are similar should be treated as the same
    conv6 = layers.Conv2D(filters=1500, kernel_size=(1, 1), padding='valid', activation='tanh',
                          input_shape=(20, 20, 3000))(conv5)
    # convolution (create new feature in the space). One feature cover 16*16 area (2 excels of 9 pixels with dilation 7)
    conv7 = layers.Conv2D(filters=2000, kernel_size=(2, 2), strides=(1, 1), padding='valid',
                                   dilation_rate=(7, 7), activation='tanh', input_shape=(20, 20, 1500))(conv6)
    conv8 = layers.Conv2D(filters=1000, kernel_size=(1, 1), padding='Same', activation='tanh',
                          input_shape=(13, 13, 2000))(conv7)
    # convolution (create new feature in the space). One feature cover 27*27 area (2 excels of 16 pixels with dilation 11)
    conv9 = layers.Conv2D(filters=500, kernel_size=(2, 2), strides=(1, 1), padding='valid',
                                   dilation_rate=(11, 11), activation='tanh', input_shape=(13, 13, 1000))(conv8)
    conv10 = layers.Conv2D(filters=250, kernel_size=(1, 1), padding='Same', activation='tanh',
                          input_shape=(2, 2, 500))(conv9)

    # convolution (create new feature in the space). One feature cover 21*21 area (2 excels of 8 pixels with dilation 5)
    # conv11 = layers.SeparableConv2D(filters=100, kernel_size=(2, 2), strides=(1, 1), padding='valid',
    #                                dilation_rate=(6, 6), activation='relu', input_shape=(8, 8, 500))(conv10)
    # conv12 = layers.Conv2D(filters=50, kernel_size=(1, 1), padding='Same', activation='relu',
    #                       input_shape=(2, 2, 100))(conv11)
    # conv13 = layers.Conv2D(filters=100, kernel_size=(2, 2), padding='valid', activation='relu',
    #                       input_shape=(2, 2, 500))(conv10)
    flatten = layers.Flatten()(conv10)
    # dense1 = layers.Dense(50, activation="relu")(flatten)
    #last layer. Here 10 digits
    dense2 = layers.Dense(10, activation="softmax")(flatten)

    model = keras.Model(inputs, dense2)

    # Define the optimizer
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # Compile the model
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # summarize the model
    model.summary()

    return model