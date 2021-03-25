import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class First_level_net(nn.Module):

    def __init__(self, weight, n_filters):
        super(First_level_net, self).__init__()
        # initializes the weights of the convolutional layer to be the weights of the 4 defined filters
        k_height, k_width = weight.shape[2:]
        # # assumes there are 4 grayscale filters
        # self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)
        # assumes there are n_filters grayscale filters
        self.conv = nn.Conv2d(1, n_filters, kernel_size=(k_height, k_width), bias=False)
        # initializes the weights of the convolutional layer
        self.conv.weight = torch.nn.Parameter(weight)
        # define a pooling layer
        # self.pool = nn.MaxPool2d(2, 2)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        # calculates the output of a convolutional layer
        # pre- and post-activation

        conv_x = self.conv(x)

        activated_x = F.relu(conv_x)

        mult_x = self.mult(activated_x)

        # applies pooling layer
        pooled_x = self.pool(mult_x)

        return mult_x, pooled_x

    def mult(self, conv_x):
        features = conv_x.detach().numpy()[0]
        feature_V_1 = np.absolute(features[0])
        feature_opposite_V_1 = np.absolute(features[1])
        feature_H_1 = np.absolute(features[2])
        feature_opposite_H_1 = np.absolute(features[3])

        feature_first_diag, feature_second_diag, feature_vertical = self.get_vertical_diagonal_feature(feature_V_1)
        feature_first_diag_down, feature_second_diag_down, feature_horizontal = self.get_horizontal_feature(feature_H_1)
        feature_opposite_first_diag, feature_opposite_second_diag, feature_opposite_vertical = self.get_vertical_diagonal_feature(
            feature_opposite_V_1)
        feature_opposite_first_diag_down, feature_opposite_second_diag_down, feature_opposite_horizontal = self.get_horizontal_feature(
            feature_opposite_H_1)

        # build tensor back with new calculated np arrays:
        # add dimension to np arrays, convert to tensor and add gradient
        new_features = np.asarray([feature_horizontal, feature_first_diag, feature_vertical, feature_second_diag,
                                   feature_first_diag_down, feature_second_diag_down,
                                   feature_opposite_horizontal, feature_opposite_first_diag, feature_opposite_vertical,
                                   feature_opposite_second_diag, feature_opposite_first_diag_down,
                                   feature_opposite_second_diag_down])
        new_features = new_features[np.newaxis, :, :]
        with torch.enable_grad():
            new_features_tensor = torch.from_numpy(new_features)

        return new_features_tensor

    def get_vertical_diagonal_feature(self, feature_V_1):
        # for feature in conv_x.detach().numpy()[0]:
        #     horizontal_count = horizontal_count + 1
        #     # first two feature are horizontal
        #     if horizontal_count < 2:
        feature_V_2 = np.delete(feature_V_1, 0, 0)
        feature_V_2 = np.insert(feature_V_2, -1, 0, axis=0)
        feature_vertical = feature_V_2 * feature_V_1

        feature_D_1 = np.delete(feature_V_2, 0, 1)
        feature_D_1 = np.insert(feature_D_1, -1, 0, axis=1)
        feature_first_diag = feature_D_1 * feature_V_1

        feature_D_2 = np.delete(feature_V_2, -1, 1)
        feature_D_2 = np.insert(feature_D_2, 0, 0, axis=1)
        feature_second_diag = feature_D_2 * feature_V_1

        return feature_first_diag, feature_second_diag, feature_vertical

    def get_horizontal_feature(self, feature_H_1):
        feature_H_2 = np.delete(feature_H_1, 0, axis=1)
        feature_H_2 = np.insert(feature_H_2, -1, 0, axis=1)
        feature_horizontal = feature_H_2 * feature_H_1

        feature_D_1 = np.delete(feature_H_2, 0, axis=0)
        feature_D_1 = np.insert(feature_D_1, -1, 0, axis=0)
        feature_first_diag_down = feature_D_1 * feature_H_1

        feature_D_2 = np.delete(feature_H_2, -1, 0)
        feature_D_2 = np.insert(feature_D_2, 0, 0, axis=0)
        feature_second_diag_down = feature_D_2 * feature_H_1

        return feature_first_diag_down, feature_second_diag_down, feature_horizontal
