import cv2
import torch
import torch.nn.functional as F

from eve_v1.filters.conversion.feature_map_conversion import get_binary_feature_map
from eve_v1.filters.first_level.first_level_features_model import First_level_net
from eve_v1.filters.first_level.first_level_filters import create_first_level_filters
from eve_v1.visualization.vizualizer import get_pixel_value_layer_with_icon, \
    get_total_picture, viz_layer, get_converted_picture_first_layer

THRESHOLD_FIRST_LAYER = 0.01

def get_elementary_feature_map(gray_img_tensor):
    icons = [cv2.imread("././resources/180_h.png"),
             cv2.imread("././resources/180_1d.png"),
             cv2.imread("././resources/180_v.png"),
             cv2.imread("././resources/180_2d.png"),
             cv2.imread("././resources/180_1d.png"),
             cv2.imread("././resources/180_2d.png"),
             cv2.imread("././resources/180_h.png"),
             cv2.imread("././resources/180_1d.png"),
             cv2.imread("././resources/180_v.png"),
             cv2.imread("././resources/180_2d.png"),
             cv2.imread("././resources/180_1d.png"),
             cv2.imread("././resources/180_2d.png")
             ]

    filters = create_first_level_filters()

    # instantiate the model and set the weights
    weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
    model = First_level_net(weight, len(filters))

    # print out the layer in the network
    print(model)

    gray_img_tensor = F.pad(gray_img_tensor.float(), (0, 2, 0, 2))
    mult_x, pooled_layer = model.forward(gray_img_tensor)
    # viz_layer(mult_x)
    # viz_layer(pooled_layer)
    # get_pixel_value_layer_with_icon(mult_x, icons, len(mult_x.detach().numpy()[0]))
    # get_total_picture(mult_x)
    # get_total_picture(pooled_layer)
    # get_pixel_value_layer_with_icon(pooled_layer, icons, len(pooled_layer.detach().numpy()[0]))

    binary_pooled_layer = get_binary_feature_map(pooled_layer.detach().numpy()[0], THRESHOLD_FIRST_LAYER)
    # viz_layer(torch.from_numpy(binary_pooled_layer).unsqueeze(0).float())
    # get_pixel_value_layer_with_icon(torch.from_numpy(binary_pooled_layer).unsqueeze(0).float(), icons, len(pooled_layer.detach().numpy()[0]))
    get_total_picture(torch.from_numpy(binary_pooled_layer).unsqueeze(0).float())

    get_converted_picture_first_layer(binary_pooled_layer)

    return binary_pooled_layer

