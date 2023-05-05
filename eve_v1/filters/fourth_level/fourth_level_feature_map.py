import cv2
import torch
import torch.nn.functional as F

from eve_v1.filters.conversion.feature_map_conversion import get_thresholds, \
    get_binary_feature_map_with_different_thresholds
from eve_v1.filters.generalization.generalization_service import generalize_angles, generalize_orientation, \
    concat_features, generalize_space
from eve_v1.filters.third_level.third_level_feature_model import Third_level_net
from eve_v1.filters.third_level.third_level_filters import create_third_level_filters
from eve_v1.visualization.vizualizer import get_total_picture, get_converted_picture


def get_fourth_level_feature_map(features_arr, third_level_manually_created_features):
    icons = []

    line_filters_number, angle_filters_number, filters, manually_created_features = create_fourth_level_filters(
        len(features_arr[1]))

    # instantiate the model and set the weights
    weight = torch.from_numpy(filters).type(torch.FloatTensor)
    model = Third_level_net(weight, len(filters), len(features_arr))

    # print out the layer in the network
    # print(model)

    # Prepare correct input - to have real convolution we need both elements are activated. Unfortunately torch
    # does not allow to do that. So, we need to simplify model - to make all element binary
    # That's why we have to do:
    # 1. For first layer: all elements of input array to second layer are equal to 0 if they are less than 0,1
    # and they are equal to 0.5 otherwise (get_binary_feature_map).
    # 2. Convolution starting from second layer: Taking into account that values in kernels are equal one or zero,
    # if 2 pixels in convolution layer are activated we have 1 (0.5*1 + 0.5*1), if only 1 pixel is activated
    # we have 0.5.
    # 3. For other layers: all elements of input array to other layers are equal to 0 if they less than 1
    # and they are equal to 0.5 otherwise (get_binary_feature_map).

    tensor = F.pad(torch.from_numpy(features_arr).float(), (0, 2, 0, 2))
    conv_layer = model.forward(tensor)

    binary_conv_layer = get_binary_feature_map_with_different_thresholds(conv_layer.detach().numpy(),
                                                                         get_thresholds(filters))

    # To generalize turning of features we need to sum up same features with different angles
    # Second parameter is quantity of filters for lines, third parameter is quantity of different rotation for every feature

    generalized_orientation_feature = generalize_orientation(binary_conv_layer, line_filters_number, angle_filters_number)
    generalized_kind_feature = generalize_angles(binary_conv_layer, line_filters_number, angle_filters_number)
    generalized_space_orientation_feature = generalize_space(generalized_orientation_feature)
    generalized_space_kind_feature = generalize_space(generalized_kind_feature)
    generalized_binary_conv_layer = concat_features(generalized_orientation_feature,
                                                    generalized_kind_feature,
                                                    generalized_space_orientation_feature,
                                                    generalized_space_kind_feature,
                                                    binary_conv_layer)

    # get_total_picture(torch.from_numpy(binary_conv_layer).unsqueeze(0).float())
    # get_pixel_value_layer_with_icon(binary_conv_layer_with_generalization_feature_tensor, icons,
    #                                 26)
    # len(generalized_binary_conv_layer))
    # get_converted_picture(binary_conv_layer, manually_created_features, second_level_manually_created_features)
    return generalized_binary_conv_layer, binary_conv_layer, manually_created_features
