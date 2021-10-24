from eve_v1.model.model import initialize_model


def train_model(x_train, y_train):

    model = initialize_model()

    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    epochs = 1 # Turn epochs to 20 if needed
    batch_size = 64
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    return model





    # https://www.kaggle.com/ruslanomelchenko/introduction-to-cnn-keras-0-997-top-6/edit
    # https://www.kaggle.com/ruslanomelchenko/digits-recognition-cnn/edit