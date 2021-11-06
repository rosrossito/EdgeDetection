# from keras import Sequential
from keras.models import *
from keras.layers import Dense, Dropout, Flatten, Conv2D, Input


def initialize_model():
    inputs = Input(14, 14, 372)
    conv1 = Conv2D(filters=256, kernel_size=(3, 3), padding='Same', activation='relu', input_shape=(14, 14, 372))(inputs)
    conv2 = Conv2D(filters=256, kernel_size=(3, 3), padding='Same', activation='relu', input_shape=(14, 14, 372))(
        conv1)

    model = Model(input=inputs, output=conv2)

    # model = Sequential()
    #
    # model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='Same', activation='relu', input_shape=(14, 14, 372)))
    # model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='Same', activation='relu'))
    # model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu'))
    # model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu'))
    # model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='Same', activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(256, activation="relu"))
    # # model.add(Dropout(0.5))
    # model.add(Dense(10, activation="softmax"))
    return model
