from keras.optimizers import RMSprop
from tensorflow import keras
from tensorflow.keras import layers


def create_model():
    inputs = keras.Input((14, 14, 372))
    conv1 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu',
                          input_shape=(14, 14, 372))(inputs)
    conv2 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu',
                          input_shape=(14, 14, 128))(
        conv1)
    conv3 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='Same', activation='relu',
                          input_shape=(14, 14, 128))(
        conv2)
    conv4 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='Same', activation='relu',
                          input_shape=(14, 14, 256))(
        conv3)
    flatten = layers.Flatten()(conv4)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense2 = layers.Dense(10, activation="softmax")(dense1)

    model = keras.Model(inputs, dense2)

    # Define the optimizer
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # Compile the model
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model
