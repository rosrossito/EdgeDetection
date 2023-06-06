from keras.optimizers import RMSprop
from tensorflow import keras
from tensorflow.keras import layers


def create_model():

    # for the moment number of filters is constant (not changed).

    inputs = keras.Input((28, 28, 1))
    # decrease resolution/precision
    pool1 = layers.AveragePooling2D(pool_size=(2,2))(inputs)

    conv1 = layers.Conv2D(filters=120, kernel_size=(4, 4), padding='Same', activation='relu',
                          input_shape=(14, 14, 1))(pool1)

    conv2 = layers.Conv2D(filters=80, kernel_size=(1, 1), padding='Same', activation='relu',
                          input_shape=(14, 14, 120))(conv1)

    conv3 = layers.Conv2D(filters=1200, kernel_size=(2, 2), strides=(4,4), padding='Same', activation='relu',
                          input_shape=(14, 14, 80))(conv2)

    conv4 = layers.Conv2D(filters=800, kernel_size=(1, 1), padding='Same', activation='relu',
                          input_shape=(14, 14, 1200))(conv3)

    conv5 = layers.Conv2D(filters=12000, kernel_size=(2, 2), strides=(8,8), padding='Same', activation='relu',
                          input_shape=(14, 14, 800))(conv4)

    conv6 = layers.Conv2D(filters=8000, kernel_size=(1, 1), padding='Same', activation='relu',
                          input_shape=(14, 14, 12000))(conv5)

    flatten = layers.Flatten()(conv6)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense2 = layers.Dense(10, activation="softmax")(dense1)

    model = keras.Model(inputs, dense2)

    # Define the optimizer
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # Compile the model
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model