import numpy as np
import torch

from eve_v1.filters.feature_builder import build_feature
from eve_v1.load_mnist import load_mnist_dataset
# from eve_v1.model.model_runner import train_model
from eve_v1.model.model_runner import train_model
from eve_v1.utils import load_Image

# load pic
# gray_img = get_MNIST_train_example()
# save_image(gray_img * 255.0, 'train_images/mnist_example')
# gray_img = load_Image('train_images/mnist_example_8.jpg')
# gray_img = gray_img / 255
# get_pixel_value_pic(gray_img)

X_train, X_val, Y_train, Y_val, test = load_mnist_dataset()

X_train_preprocessed = []
counter=1

# Todo rework to pass all images
for x in X_train:
    gray_img_tensor = torch.from_numpy(x[:, :, 0]).unsqueeze(0).unsqueeze(1)
    preprocessed_features = build_feature(gray_img_tensor)
    X_train_preprocessed.append(preprocessed_features)
    print(counter)
    counter=counter+1

# Reshape image in 3 dimensions (height = 14px, width = 14px , canal = 372)
# 1 layer - shape 12x14x14
# 2 layer 92(60 + 32(4*8) generalized)x14x14
# 3 layer 268(208 + 60(14*2+16*2) generalized)x14x14
# Total features 372(12+92+268)x14x14
X_train_preprocessed = np.array(X_train_preprocessed).values.reshape(-1, 14, 14, 372)

model, history = train_model(X_train_preprocessed, Y_train, X_val, Y_val)
