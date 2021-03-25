import cv2

import torch

from eve_v1.filters.elementary.first_level_features_model import First_level_net
from eve_v1.filters.elementary.first_level_filters import create_first_level_filters
from eve_v1.visualization.vizualizer import viz_layer, get_pixel_value_layer


def get_elementary_feature_map(gray_img_tensor):
    icons = [cv2.imread("././resources/180_h.png")]

    filters = create_first_level_filters()

    # instantiate the model and set the weights
    weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
    model = First_level_net(weight, len(filters))

    # print out the layer in the network
    print(model)

    # mult_x, activated_layer, pooled_layer = model.forward(gray_img_tensor.float())
    mult_x, pooled_layer = model.forward(gray_img_tensor.float())
    # viz_layer(mult_x)
    # viz_layer(pooled_layer)
    # get_pixel_value_layer(mult_x)
    # get_pixel_value_layer(pooled_layer, icons)
    return pooled_layer

