from eve_v1.filters.feature_generator.generator import get_kernel
import numpy as np

LINE_FILTERS_NUMBER = 6
ANGLE_FILTERS_NUMBER = 8


def create_second_level_filters(kernel_depth):
    kernel_size = 2
    filters = []
    reversed_step = int(kernel_depth / 2)
    features_types = [0, reversed_step]

    # Input: feature_horizontal, feature_first_diag, feature_vertical,
    # feature_second_diag, feature_first_diag_down, feature_second_diag_down,
    # feature_opposite_horizontal, feature_opposite_first_diag, feature_opposite_vertical,
    # feature_opposite_second_diag, feature_opposite_first_diag_down, feature_opposite_second_diag_down
    # Output: 60 features (30 features (without generalized) - line by 6 rotations and 3 angles by 8 rotations + 30 reversed features)

    for feature in features_types:
        reversed_feature = reversed_step - feature

        # lines (180 degree)
        # parameters: kernel number, column, raw
        filters.append(
            get_kernel(kernel_size, kernel_depth, [[reversed_feature + 1, 1, 1], [reversed_feature + 1, 2, 1]]))
        filters.append(get_kernel(kernel_size, kernel_depth, [[feature + 2, 1, 1], [feature + 2, 2, 2]]))
        filters.append(get_kernel(kernel_size, kernel_depth, [[feature + 3, 1, 1], [feature + 3, 1, 2]]))
        filters.append(get_kernel(kernel_size, kernel_depth, [[feature + 4, 2, 1], [feature + 4, 1, 2]]))
        filters.append(get_kernel(kernel_size, kernel_depth, [[feature + 5, 1, 1], [feature + 5, 2, 2]]))
        filters.append(get_kernel(kernel_size, kernel_depth, [[feature + 6, 1, 2], [feature + 6, 2, 1]]))

        # 135 degree
        filters.append(
            get_kernel(kernel_size, kernel_depth, [[reversed_feature + 5, 1, 1], [reversed_feature + 1, 2, 2]]))
        filters.append(get_kernel(kernel_size, kernel_depth, [[feature + 3, 1, 1], [feature + 2, 1, 2]]))
        filters.append(get_kernel(kernel_size, kernel_depth, [[feature + 4, 1, 1], [feature + 3, 1, 2]]))
        filters.append(get_kernel(kernel_size, kernel_depth, [[feature + 6, 1, 1], [feature + 1, 2, 1]]))
        filters.append(get_kernel(kernel_size, kernel_depth, [[feature + 1, 1, 1], [feature + 5, 2, 1]]))
        filters.append(
            get_kernel(kernel_size, kernel_depth, [[reversed_feature + 2, 1, 1], [reversed_feature + 3, 2, 2]]))
        filters.append(
            get_kernel(kernel_size, kernel_depth, [[reversed_feature + 3, 2, 1], [reversed_feature + 4, 1, 2]]))
        filters.append(
            get_kernel(kernel_size, kernel_depth, [[reversed_feature + 1, 1, 2], [reversed_feature + 6, 2, 1]]))

        # 90 degree
        filters.append(get_kernel(kernel_size, kernel_depth, [[feature + 3, 1, 1], [reversed_feature + 1, 2, 2]]))
        filters.append(get_kernel(kernel_size, kernel_depth, [[feature + 4, 1, 1], [feature + 2, 1, 2]]))
        filters.append(get_kernel(kernel_size, kernel_depth, [[feature + 1, 2, 1], [feature + 3, 1, 2]]))
        filters.append(get_kernel(kernel_size, kernel_depth, [[feature + 6, 1, 1], [feature + 5, 2, 1]]))
        filters.append(get_kernel(kernel_size, kernel_depth, [[feature + 1, 1, 1], [reversed_feature + 3, 2, 2]]))
        filters.append(
            get_kernel(kernel_size, kernel_depth, [[reversed_feature + 2, 1, 1], [reversed_feature + 4, 1, 2]]))
        filters.append(
            get_kernel(kernel_size, kernel_depth, [[reversed_feature + 1, 1, 2], [reversed_feature + 3, 2, 1]]))
        filters.append(
            get_kernel(kernel_size, kernel_depth, [[reversed_feature + 5, 1, 1], [reversed_feature + 6, 2, 1]]))

        # 45 degree
        filters.append(get_kernel(kernel_size, kernel_depth, [[feature + 4, 1, 1], [reversed_feature + 1, 2, 2]]))
        filters.append(get_kernel(kernel_size, kernel_depth, [[feature + 1, 2, 1], [feature + 2, 1, 2]]))
        filters.append(get_kernel(kernel_size, kernel_depth, [[feature + 3, 1, 2], [feature + 5, 2, 1]]))
        filters.append(get_kernel(kernel_size, kernel_depth, [[feature + 6, 1, 1], [reversed_feature + 3, 2, 2]]))
        filters.append(get_kernel(kernel_size, kernel_depth, [[feature + 1, 1, 1], [reversed_feature + 4, 1, 2]]))
        filters.append(
            get_kernel(kernel_size, kernel_depth, [[reversed_feature + 2, 1, 1], [reversed_feature + 1, 1, 2]]))
        filters.append(
            get_kernel(kernel_size, kernel_depth, [[reversed_feature + 5, 1, 1], [reversed_feature + 3, 2, 1]]))
        filters.append(get_kernel(kernel_size, kernel_depth, [[feature + 3, 1, 1], [reversed_feature + 6, 2, 1]]))

    return LINE_FILTERS_NUMBER, ANGLE_FILTERS_NUMBER, np.array(filters)
