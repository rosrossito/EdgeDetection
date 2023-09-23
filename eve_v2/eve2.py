import argparse

import keras
import pandas as pd

from eve_v2.model.model import create_model
from eve_v2.model.model_runner import train_model, predict
from utils.mnist_util import load_mnist_dataset
from utils.vizualize_service import viz_filter

MODEL_PATH = "./eve2_model.h5"


def get_train_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-mode", required=True, help="mode")
    return vars(ap.parse_args())


def execute(mode):
    # Load 37800 train images and 4200 val images
    X_train, X_val, Y_train, Y_val, test = load_mnist_dataset()

    if mode == "train":
        model = create_model()
        history = train_model(model, X_train, X_val, Y_train, Y_val, MODEL_PATH)

    elif mode == "visualize":
        model = keras.models.load_model(MODEL_PATH)
        viz_filter(model)

    elif mode == "predict":
        model = keras.models.load_model(MODEL_PATH)
        results = predict(test, model)

    elif mode == "submit":
        prediction = pd.read_csv("prediction.csv")
        results = pd.Series(prediction, name="Label")
        submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)
        submission.to_csv("cnn_mnist_datagen.csv", index=False)


def main(args):
    execute(args["mode"])


if __name__ == '__main__':
    main(get_train_args())
