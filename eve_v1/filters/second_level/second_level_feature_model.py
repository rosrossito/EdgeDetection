import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Second_level_net(nn.Module):

    def __init__(self, weight, n_filters, input_channels):
        super(Second_level_net, self).__init__()
        # initializes the weights of the convolutional layer to be the weights of the 4 defined filters
        k_depth, k_height, k_width = weight.shape[1:]
        self.conv = nn.Conv2d(input_channels, n_filters, kernel_size=(k_depth, k_height, k_width), bias=False)
        # initializes the weights of the convolutional layer
        self.conv.weight = torch.nn.Parameter(weight)
        # define a pooling layer
        # self.pool = nn.MaxPool2d(2, 2)
        # self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        # calculates the output of a convolutional layer

        conv_x = self.conv(x)

        # activated_x = F.relu(conv_x)
        return conv_x
