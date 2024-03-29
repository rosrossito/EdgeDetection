from typing import List

import numpy as np

from eve_v1.filters.feature_generator.generator import get_kernel

LINE_FILTERS_NUMBER = 8
ANGLE_FILTERS_NUMBER = 16


def create_third_level_filters(kernel_depth):
    kernel_size = 3
    filters = []

    manually_created_features = get_manually_created_feature(kernel_depth)

    for manually_created_feature in manually_created_features:
        filters.append(get_kernel(kernel_size, kernel_depth, manually_created_feature))
    return LINE_FILTERS_NUMBER, ANGLE_FILTERS_NUMBER, np.array(filters), manually_created_features


def get_manually_created_feature(kernel_depth):
    manually_created_feature: List[List[List[int]]] = []

    reversed_step = int(kernel_depth / 2)
    features_types = [0, reversed_step]

    # Input: 60 features (without generalized) - line by 6 rotations and 3 angles by 8 rotations + reversed
    # Output: 208 features (104 features - line by 8 rotations and 6 angles by 16 rotations + reversed)

    for feature in features_types:
        reversed_feature = reversed_step - feature

        # lines (180 degree)
        # parameters: kernel number, column, raw
        manually_created_feature.append([[feature + 1, 1, 1], [feature + 1, 2, 1], [feature + 1, 3, 1]])
        manually_created_feature.append(
            [[feature + 1, 1, 1], [reversed_feature + 11, 2, 1], [feature + 7, 3, 1]])
        manually_created_feature.append([[feature + 2, 1, 1], [feature + 2, 2, 2], [feature + 2, 3, 3]])
        manually_created_feature.append([[feature + 3, 1, 1], [feature + 8, 1, 2], [reversed_feature + 12, 1, 3]])
        manually_created_feature.append([[feature + 3, 1, 1], [feature + 3, 1, 2], [feature + 3, 1, 3]])
        manually_created_feature.append([[feature + 3, 1, 2], [reversed_feature + 13, 1, 2], [feature + 9, 1, 3]])
        manually_created_feature.append([[feature + 4, 3, 1], [feature + 4, 2, 2], [feature + 4, 1, 3]])
        manually_created_feature.append(
            [[reversed_feature + 14, 1, 1], [feature + 10, 2, 1], [reversed_feature + 1, 3, 1]])

        # lines (157.5 degree)
        manually_created_feature.append([[reversed_feature + 11, 1, 1], [feature + 1, 3, 2]])
        manually_created_feature.append([[reversed_feature + 5, 1, 1], [reversed_feature + 11, 3, 2]])
        manually_created_feature.append([[reversed_feature + 12, 1, 1], [feature + 2, 2, 3]])
        manually_created_feature.append([[feature + 3, 1, 1], [reversed_feature + 12, 1, 3]])

        manually_created_feature.append([[reversed_feature + 13, 1, 1], [feature + 3, 1, 3]])
        manually_created_feature.append([[feature + 4, 2, 1], [reversed_feature + 13, 1, 3]])
        manually_created_feature.append([[feature + 6, 1, 2], [reversed_feature + 14, 3, 1]])
        manually_created_feature.append([[reversed_feature + 14, 1, 1], [reversed_feature + 1, 3, 1]])

        manually_created_feature.append([[feature + 5, 3, 2], [reversed_feature + 7, 1, 1]])
        manually_created_feature.append([[reversed_feature + 1, 1, 1], [reversed_feature + 7, 3, 1]])
        manually_created_feature.append([[reversed_feature + 2, 1, 1], [reversed_feature + 13, 3, 3]])
        manually_created_feature.append([[reversed_feature + 8, 1, 1], [reversed_feature + 3, 2, 3]])

        manually_created_feature.append([[reversed_feature + 3, 2, 1], [reversed_feature + 9, 1, 3]])
        manually_created_feature.append([[reversed_feature + 9, 3, 1], [reversed_feature + 4, 1, 3]])
        manually_created_feature.append([[reversed_feature + 5, 3, 1], [reversed_feature + 10, 1, 3]])
        manually_created_feature.append([[feature + 1, 1, 2], [reversed_feature + 10, 3, 1]])

        # lines (135 degree)
        manually_created_feature.append([[reversed_feature + 5, 1, 1], [feature + 11, 2, 2], [feature + 1, 3, 3]])
        manually_created_feature.append([[feature + 8, 1, 1], [feature + 2, 1, 2], [feature + 7, 2, 3]])
        manually_created_feature.append([[feature + 3, 1, 1], [feature + 8, 1, 2], [feature + 2, 1, 3]])
        manually_created_feature.append([[feature + 9, 1, 1], [feature + 3, 1, 2], [feature + 8, 1, 3]])
        manually_created_feature.append([[feature + 4, 1, 1], [feature + 9, 1, 2], [feature + 3, 1, 3]])
        manually_created_feature.append([[feature + 10, 2, 1], [feature + 4, 1, 2], [feature + 9, 1, 3]])
        manually_created_feature.append([[feature + 6, 1, 1], [feature + 10, 2, 1], [reversed_feature + 1, 3, 1]])
        manually_created_feature.append([[feature + 10, 1, 1], [reversed_feature + 1, 2, 1], [feature + 11, 3, 1]])
        manually_created_feature.append([[reversed_feature + 1, 1, 1], [feature + 11, 2, 1], [feature + 5, 3, 1]])
        manually_created_feature.append([[feature + 11, 1, 1], [feature + 12, 3, 2]])
        manually_created_feature.append(
            [[reversed_feature + 2, 1, 1], [feature + 12, 2, 2], [reversed_feature + 3, 3, 3]])
        manually_created_feature.append([[feature + 12, 1, 1], [reversed_feature + 3, 2, 2], [feature + 13, 1, 3]])
        manually_created_feature.append(
            [[reversed_feature + 3, 3, 1], [feature + 13, 2, 2], [reversed_feature + 4, 1, 3]])
        manually_created_feature.append([[feature + 13, 3, 1], [reversed_feature + 4, 2, 2], [feature + 14, 1, 3]])
        manually_created_feature.append([[reversed_feature + 6, 3, 1], [feature + 14, 2, 2], [feature + 1, 1, 3]])
        manually_created_feature.append([[feature + 7, 1, 1], [feature + 1, 2, 2], [feature + 14, 3, 1]])

        # lines (112.5 degree)
        manually_created_feature.append([[reversed_feature + 12, 1, 1], [feature + 1, 3, 3]])
        manually_created_feature.append([[feature + 3, 1, 1], [reversed_feature + 11, 2, 3]])
        manually_created_feature.append([[reversed_feature + 13, 1, 1], [feature + 2, 1, 3]])
        manually_created_feature.append([[feature + 4, 1, 1], [reversed_feature + 12, 1, 3]])

        manually_created_feature.append([[reversed_feature + 14, 2, 1], [feature + 3, 1, 3]])
        manually_created_feature.append([[reversed_feature + 13, 1, 2], [reversed_feature + 1, 3, 1]])
        manually_created_feature.append([[feature + 4, 1, 1], [reversed_feature + 7, 3, 1]])
        manually_created_feature.append([[reversed_feature + 14, 1, 1], [reversed_feature + 5, 3, 1]])

        manually_created_feature.append([[reversed_feature + 1, 1, 1], [reversed_feature + 8, 3, 2]])
        manually_created_feature.append([[feature + 5, 1, 1], [reversed_feature + 14, 2, 3]])
        manually_created_feature.append([[reversed_feature + 7, 1, 1], [reversed_feature + 3, 3, 3]])
        manually_created_feature.append([[reversed_feature + 8, 2, 1], [reversed_feature + 4, 1, 3]])

        manually_created_feature.append([[reversed_feature + 3, 3, 1], [reversed_feature + 10, 1, 3]])
        manually_created_feature.append([[reversed_feature + 9, 3, 1], [feature + 1, 1, 3]])
        manually_created_feature.append([[reversed_feature + 6, 3, 1], [reversed_feature + 11, 1, 2]])
        manually_created_feature.append([[reversed_feature + 5, 1, 1], [reversed_feature + 10, 3, 2]])

        # lines (90 degree)
        manually_created_feature.append([[feature + 3, 1, 1], [feature + 15, 1, 2], [feature + 1, 2, 3]])
        manually_created_feature.append([[feature + 9, 1, 1], [feature + 11, 2, 3]])
        manually_created_feature.append([[feature + 4, 1, 1], [feature + 16, 1, 2], [feature + 2, 1, 3]])
        manually_created_feature.append([[feature + 10, 2, 1], [feature + 8, 1, 3]])

        manually_created_feature.append([[reversed_feature + 1, 2, 1], [feature + 17, 1, 1], [feature + 3, 1, 2]])
        manually_created_feature.append([[feature + 11, 3, 1], [feature + 9, 1, 2]])
        manually_created_feature.append([[feature + 6, 1, 1], [feature + 18, 2, 1], [feature + 5, 3, 1]])
        manually_created_feature.append([[feature + 10, 1, 1], [feature + 12, 3, 2]])

        manually_created_feature.append(
            [[reversed_feature + 1, 1, 1], [feature + 19, 2, 1], [reversed_feature + 3, 3, 2]])
        manually_created_feature.append([[feature + 11, 1, 1], [feature + 13, 3, 3]])
        manually_created_feature.append(
            [[reversed_feature + 2, 1, 1], [feature + 20, 2, 2], [reversed_feature + 4, 1, 3]])
        manually_created_feature.append([[feature + 12, 2, 1], [feature + 14, 1, 3]])

        manually_created_feature.append([[reversed_feature + 3, 3, 1], [feature + 21, 2, 2], [feature + 1, 1, 3]])
        manually_created_feature.append([[feature + 7, 1, 2], [feature + 13, 3, 1]])
        manually_created_feature.append(
            [[reversed_feature + 5, 1, 1], [feature + 22, 2, 2], [reversed_feature + 6, 3, 1]])
        manually_created_feature.append([[feature + 8, 1, 1], [feature + 14, 3, 2]])

        # lines (67.5 degree)

        manually_created_feature.append([[reversed_feature + 9, 1, 1], [feature + 1, 2, 3]])
        manually_created_feature.append([[feature + 4, 1, 1], [reversed_feature + 11, 2, 3]])
        manually_created_feature.append([[reversed_feature + 14, 2, 1], [feature + 2, 1, 3]])
        manually_created_feature.append([[reversed_feature + 1, 2, 1], [feature + 8, 1, 2]])

        manually_created_feature.append([[feature + 3, 1, 2], [feature + 11, 2, 1]])
        manually_created_feature.append([[feature + 5, 3, 1], [reversed_feature + 13, 1, 2]])
        manually_created_feature.append([[reversed_feature + 8, 3, 2], [feature + 4, 1, 1]])
        manually_created_feature.append([[reversed_feature + 3, 3, 2], [feature + 10, 1, 1]])

        manually_created_feature.append([[reversed_feature + 1, 1, 1], [feature + 13, 3, 2]])
        manually_created_feature.append([[feature + 4, 1, 3], [reversed_feature + 7, 1, 1]])
        manually_created_feature.append([[reversed_feature + 2, 1, 1], [reversed_feature + 10, 1, 3]])
        manually_created_feature.append([[feature + 1, 1, 3], [feature + 12, 2, 1]])

        manually_created_feature.append([[reversed_feature + 3, 3, 1], [feature + 7, 1, 2]])
        manually_created_feature.append([[reversed_feature + 5, 1, 1], [reversed_feature + 9, 3, 1]])
        manually_created_feature.append([[reversed_feature + 12, 1, 1], [reversed_feature + 6, 3, 1]])
        manually_created_feature.append([[feature + 3, 1, 1], [feature + 14, 2, 2]])

        # lines (45 degree)
        manually_created_feature.append([[feature + 4, 1, 1], [feature + 23, 1, 2], [feature + 1, 2, 3]])
        manually_created_feature.append([[feature + 10, 2, 1], [feature + 23, 1, 2], [reversed_feature + 11, 2, 3]])
        manually_created_feature.append([[reversed_feature + 1, 2, 1], [feature + 24, 1, 1], [feature + 2, 1, 2]])
        manually_created_feature.append([[feature + 11, 2, 1], [feature + 24, 1, 1], [reversed_feature + 12, 1, 2]])

        manually_created_feature.append([[feature + 5, 2, 1], [feature + 25, 1, 1], [feature + 3, 1, 2]])
        manually_created_feature.append([[feature + 12, 3, 2], [feature + 25, 2, 1], [reversed_feature + 13, 1, 2]])
        manually_created_feature.append([[reversed_feature + 3, 3, 2], [feature + 26, 2, 1], [feature + 6, 1, 1]])
        manually_created_feature.append([[feature + 13, 2, 2], [feature + 26, 2, 1], [reversed_feature + 14, 1, 1]])

        manually_created_feature.append(
            [[reversed_feature + 1, 1, 1], [feature + 27, 1, 1], [reversed_feature + 4, 2, 2]])
        manually_created_feature.append([[reversed_feature + 7, 1, 1], [feature + 27, 2, 2], [feature + 14, 1, 3]])
        manually_created_feature.append([[reversed_feature + 2, 1, 1], [feature + 28, 2, 2], [feature + 1, 1, 3]])
        manually_created_feature.append([[reversed_feature + 8, 2, 1], [feature + 28, 2, 2], [feature + 7, 1, 2]])

        manually_created_feature.append(
            [[reversed_feature + 5, 1, 1], [feature + 29, 2, 2], [reversed_feature + 3, 3, 1]])
        manually_created_feature.append([[feature + 8, 1, 1], [feature + 29, 2, 2], [feature + 14, 3, 1]])
        manually_created_feature.append([[reversed_feature + 6, 2, 1], [feature + 3, 1, 1], [feature + 30, 1, 2]])
        manually_created_feature.append([[feature + 9, 1, 1], [feature + 30, 1, 2], [reversed_feature + 10, 2, 2]])
    return manually_created_feature
