import cv2

import torch

from eve_v1.filters.first_level.elementary_feature_map import get_elementary_feature_map
from eve_v1.filters.second_level.second_level_feature_map import get_second_level_feature_map
from eve_v1.filters.third_level.third_level_feature_map import get_third_level_feature_map
from eve_v1.load_mnist import get_MNIST_train_example
# from eve_v1.model.model_runner import train_model
from eve_v1.utils import save_image, load_Image
from eve_v1.visualization.vizualizer import get_pixel_value_pic

# load pic
# gray_img = get_MNIST_train_example()
# save_image(gray_img * 255.0, 'train_images/mnist_example')
gray_img = load_Image('train_images/mnist_example_8.jpg')
gray_img = gray_img / 255
# get_pixel_value_pic(gray_img)
gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1)

elementary_feature_map = get_elementary_feature_map(gray_img_tensor) #shape 12x14x14
second_level_feature_with_generalization, second_level_feature_map, second_level_manually_created_features = get_second_level_feature_map(
    elementary_feature_map)
third__level_feature_with_generalization, third_level_feature_map = get_third_level_feature_map(
    second_level_feature_map, second_level_manually_created_features)

#Todo: get x_train (my representation) and y_train
#1 layer - shape 12x14x14
#2 layer 92(60 + 32(4*8) generalized)x14x14
#3 layer 268(208 + 60(14*2+16*2) generalized)x14x14
#Total features 372(12+92+268)x14x14


# model = train_model(x_train, y_train)





# Todo Refine visualisation (add angles)
# maybe to merge similar features for the output (direct and reversed). They used in different places to build higher features
# but represent the same
# Todo: now we have generalization by angle, need to add generalization by direction (map of directions?) and
#  spatial generalization (the same angle (or similar if some are absent - pyramid) in different layers) and
#  maybe layer generalization
# + affine transformation - incline
# Todo Back to notes description and create CNN
