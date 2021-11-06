# from keras.optimizers import RMSprop

from eve_v1.model.model import initialize_model


def train_model(x_train, y_train, x_val, y_val):

    # Define the optimizer
    # optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model = initialize_model()
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    epochs = 1 # Turn epochs to 20 if needed
    batch_size = 64

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                 validation_data = (x_val, y_val), verbose = 2)

    return model, history





    # https://www.kaggle.com/ruslanomelchenko/introduction-to-cnn-keras-0-997-top-6/edit
    # https://www.kaggle.com/ruslanomelchenko/digits-recognition-cnn/edit