import numpy as np


def convert_third_level_feature(coords, feature_offsets):
    next_level_conversion_data = []
    for feature_coords, feature_offset_raw in zip(coords, feature_offsets):
        for coord in feature_coords:
            for feature_offset in feature_offset_raw:
                if len(coord) > 0:
                    # column, raw
                    x = coord[0] + feature_offset[1]
                    y = coord[1] + feature_offset[2]
                    next_level_conversion_data.append([[x, y], feature_offset[0]])
    return next_level_conversion_data


def convert_second_level_feature(previous_coords_feature, feature_offsets):
    next_level_conversion_data = []
    for feature in previous_coords_feature:
        for feature_offset in feature_offsets[feature[1] - 1]:
            # column, raw
            x = feature[0][0] + feature_offset[1]
            y = feature[0][1] + feature_offset[2]
            next_level_conversion_data.append([[x, y], feature_offset[0]])
    return next_level_conversion_data


def convert_first_level_feature(previous_coords_feature):
    next_level_conversion_data = []
    for feature in previous_coords_feature:
        pixel_offsets = get_pixel_offset(feature[1])
        for pixel_offset in pixel_offsets:
            # column, raw
            x = feature[0][0] + pixel_offset[0]
            y = feature[0][1] + pixel_offset[1]
            next_level_conversion_data.append([x, y])
    return next_level_conversion_data


def get_coord(features):
    # raw, column
    coords = []
    for feature in features:
        # raw idx and col idx
        raw, col = np.where(feature > 0)
        x_y = []
        if len(raw) > 0:
            for y, x in zip(raw, col):
                x_y.append([x, y])
        coords.append(x_y)
    # column, raw
    return coords


def get_pixel_offset(first_level_feature):
    # column, raw
    if first_level_feature == 1:
        return [[0, 0], [1, 0]]
    elif first_level_feature == 2:
        return [[0, 0], [1, 1]]
    elif first_level_feature == 3:
        return [[0, 0], [0, 1]]
    elif first_level_feature == 4:
        return [[1, 0], [0, 1]]
    elif first_level_feature == 5:
        return [[0, 0], [1, 1]]
    elif first_level_feature == 6:
        return [[1, 0], [0, 1]]
    elif first_level_feature == 7:
        return [[0, 1], [1, 1]]
    elif first_level_feature == 8:
        return [[1, 0], [2, 1]]
    elif first_level_feature == 9:
        return [[1, 0], [1, 1]]
    elif first_level_feature == 10:
        return [[2, 0], [1, 1]]
    elif first_level_feature == 11:
        return [[0, 1], [1, 2]]
    elif first_level_feature == 12:
        return [[1, 1], [0, 2]]
    else:
        raise ValueError('Feature are not correctly constructed.')
