import numpy as np
import pandas as pd


def train_model(model, X_train, X_val, Y_train, Y_val, path):
    epochs = 30
    model.compile(
        optimizer="Adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        X_train, Y_train, epochs=epochs,
        validation_data=(X_val, Y_val),
    )
    model.save(path)

    return history


def predict(X, model):
    # predict results
    results = model.predict(X)

    # select the index with the maximum probability
    results = np.argmax(results, axis=1)
    results = pd.Series(results, name="Label")
    results.to_csv("prediction.csv", index=False)

    return results

    # https://www.kaggle.com/ruslanomelchenko/introduction-to-cnn-keras-0-997-top-6/edit
    # https://www.kaggle.com/ruslanomelchenko/digits-recognition-cnn/edit
    # https://www.kaggle.com/ruslanomelchenko/cnn-without-pooling-layers/edit/run/85736021
