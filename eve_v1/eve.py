import keras
import numpy as np
import torch
import pandas as pd

from eve_v1.filters.feature_builder import build_feature
from eve_v1.filters.first_level.elementary_feature_map import get_elementary_feature_map
from eve_v1.filters.second_level.second_level_feature_map import get_second_level_feature_map
from eve_v1.filters.third_level.third_level_feature_map import get_third_level_feature_map
from eve_v1.load_mnist import load_mnist_dataset, get_MNIST_train_example
# from eve_v1.model.model_runner import train_model
from eve_v1.model.model import create_model
from eve_v1.model.model_runner import train_model, predict
from eve_v1.utils import load_Image, save_image
from eve_v1.visualization.vizualizer import get_pixel_value_pic

mode = "debug"

def preprocess_features(X_set):
    gray_img_tensor = torch.from_numpy(X_set).unsqueeze(1)
    preprocessed_features = build_feature(gray_img_tensor)
    X_preprocessed = np.swapaxes(preprocessed_features, 1, 2)
    X_preprocessed = np.swapaxes(X_preprocessed, 2, 3)
    del preprocessed_features
    return X_preprocessed

if mode == "debug":
    # load pic
    gray_img = get_MNIST_train_example()
    save_image(gray_img * 255.0, 'train_images/mnist_example')
    gray_img = load_Image('train_images/mnist_example_8.jpg')
    gray_img = gray_img / 255
    get_pixel_value_pic(gray_img)

    gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1)

    build_feature(gray_img_tensor)


    # elementary_feature_map = get_elementary_feature_map(gray_img_tensor)  # shape 12x14x14
    # second_level_feature_with_generalization, second_level_feature_map, second_level_manually_created_features = \
    #     get_second_level_feature_map(elementary_feature_map)
    # third_level_feature_with_generalization, third_level_feature_map, third_level_manually_created_features = \
    #     get_third_level_feature_map(second_level_feature_map, second_level_manually_created_features)





    # 1 layer - shape 12x14x14
    # 2 layer 92(60 + 32(4*8) generalized)x14x14
    # 3 layer 268(208 + 60(14*2+16*2) generalized)x14x14
    # Total features 372(12+92+268)x14x14

    # Run cycle and extract needed features for input
    #Todo -- First 10 features of third_level_feature_map + 105-115 features are lines
    #Todo --elementary level produces 6 (not 10) features - how to put them to input?


elif mode == "train":

    # Load 37800 train images and 4200 val images
    X_train, X_val, Y_train, Y_val, test = load_mnist_dataset()

    # Get input tensor
    # 1-st batch
    # X_set =  np.concatenate((X_train[0:9450, :, :, 0], X_val[0:1050, :, :, 0]), axis=0)
    # 2-nd batch
    # X_set =  np.concatenate((X_train[9450:18900, :, :, 0], X_val[1050:2100, :, :, 0]), axis=0)
    # 3-rd batch
    # X_set =  np.concatenate((X_train[18900:28350, :, :, 0], X_val[2100:3150, :, :, 0]), axis=0)
    # 4-th batch
    X_set = np.concatenate((X_train[28350:37800, :, :, 0], X_val[3150:4200, :, :, 0]), axis=0)
    X_preprocessed = preprocess_features(X_set)

    X_train_preprocessed = X_preprocessed[0:9450]
    X_val_preprocessed = X_preprocessed[9450:10500]

    # 1-st batch
    # Y_train_preprocessed = Y_train[0:9450]
    # Y_val_preprocessed = Y_val[0:1050]
    # del X_preprocessed

    # 2-nd batch
    # Y_train_preprocessed = Y_train[9450:18900]
    # Y_val_preprocessed = Y_val[1050:2100]
    # del X_preprocessed

    # 3-rd batch
    # Y_train_preprocessed = Y_train[18900:28350]
    # Y_val_preprocessed = Y_val[2100:3150]
    # del X_preprocessed

    # 4-th batch
    Y_train_preprocessed = Y_train[28350:37800]
    Y_val_preprocessed = Y_val[3150:4200]
    del X_preprocessed

    print(X_train_preprocessed.shape)
    print(X_val_preprocessed.shape)
    print(Y_train_preprocessed.shape)
    print(Y_val_preprocessed.shape)

    # Images (height = 14px, width = 14px , canal = 372)
    # 1 layer - shape 12x14x14
    # 2 layer 92(60 + 32(4*8) generalized)x14x14
    # 3 layer 268(208 + 60(14*2+16*2) generalized)x14x14
    # Total features 372(12+92+268)x14x14

    model = create_model()

    #Train model
    history = train_model(model, X_train_preprocessed, Y_train_preprocessed, X_val_preprocessed, Y_val_preprocessed)

elif mode == "predict":

    # Load 37800 train images and 4200 val images
    X_train, X_val, Y_train, Y_val, test = load_mnist_dataset()

    # Test set
    # 1 batch
    # X_set = test[0:10500, :, :, 0]
    # 2 batch
    # X_set = test[10500:21000, :, :, 0]
    # 3 batch
    X_set = test[21000:28000, :, :, 0]
    X_preprocessed = preprocess_features(X_set)

    results = predict(X_preprocessed)

elif mode == "submission":

    prediction1 = pd.read_csv("../input/results/prediction1.csv")
    prediction2 = pd.read_csv("../input/results/prediction2.csv")
    prediction3 = pd.read_csv("../input/results/prediction3.csv")
    print(prediction1.shape)
    print(prediction2.shape)
    print(prediction3.shape)
    print(prediction3.head(3))
    prediction = np.concatenate((prediction1, prediction2, prediction3), axis=0)
    print(prediction.shape)
    prediction = np.ravel(prediction[:, 0])
    print(prediction.shape)

    results = pd.Series(prediction, name="Label")

    submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)

    submission.to_csv("cnn_mnist_datagen.csv", index=False)

