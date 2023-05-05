import keras
import numpy as np

from eve_v1.model.model import create_model


def train_model(model, X_train_preprocessed, Y_train_preprocessed, X_val_preprocessed, Y_val_preprocessed):

    is_load_model = True

    if is_load_model:
        #Load trained model
        model = keras.models.load_model("../input/testmodel5/eve_model4.h5")
        loss, acc = model.evaluate(X_val_preprocessed, Y_val_preprocessed)
        print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

    epochs = 30

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        X_train_preprocessed, Y_train_preprocessed, epochs=epochs,
        validation_data=(X_val_preprocessed, Y_val_preprocessed),
    )
    model.save("./eve_model.h5")

    return history

def predict(X_preprocessed):
    model = keras.models.load_model("../input/testmodel5/eve_model4.h5")

    # predict results
    results = model.predict(X_preprocessed)

    # select the indix with the maximum probability
    results = np.argmax(results, axis=1)

    # results = pd.Series(results,name="Label")

    # 1 batch
    # results.to_csv("prediction1.csv", index=False)
    # 2 batch
    # results.to_csv("prediction2.csv", index=False)
    # 3 batch
    results.to_csv("prediction3.csv", index=False)
    print(results.shape)

    return results

    # https://www.kaggle.com/ruslanomelchenko/introduction-to-cnn-keras-0-997-top-6/edit
    # https://www.kaggle.com/ruslanomelchenko/digits-recognition-cnn/edit