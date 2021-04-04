import cv2
import torch
import torch.nn.functional as F

from eve_v1.filters.elementary.first_level_features_model import First_level_net
from eve_v1.filters.elementary.first_level_filters import create_first_level_filters
from eve_v1.visualization.vizualizer import get_pixel_value_layer_with_icon, \
    get_total_picture


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
    # get_pixel_value_layer(mult_x)
    # get_total_picture(pooled_layer)
    # get_pixel_value_layer_with_icon(pooled_layer, icons, len(pooled_layer.detach().numpy()[0]))
    return pooled_layer

