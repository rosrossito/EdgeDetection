import numpy as np


def convert_third_level_feature(coords, feature_offsets):
    next_level_conversion_data = []
    for feature_coords, feature_offset_raw in zip(coords, feature_offsets):
        for coord in feature_coords:
            for feature_offset in feature_offset_raw:
                if len(coord)>0:
                    x = coord[0] + feature_offset[2]
                    y = coord[1] + feature_offset[1]
                    next_level_conversion_data.append([[x, y], feature_offset[0]])
    return next_level_conversion_data


def convert_second_level_feature(previous_coords_feature, feature_offsets):
    next_level_conversion_data = []
    for feature in previous_coords_feature:
        for feature_offset in feature_offsets[feature[1]]:
            x = feature[0][0] + feature_offset[2]
            y = feature[0][1] + feature_offset[1]
            next_level_conversion_data.append([[x, y], feature_offset[0]])
    return next_level_conversion_data


def convert_first_level_feature(previous_coords_feature):
    next_level_conversion_data = []
    for feature in previous_coords_feature:
        pixel_offset = get_pixel_offset(feature[1])
        x = feature[0][0] + pixel_offset[0]
        y = feature[0][1] + pixel_offset[1]
        next_level_conversion_data.append([x,y])
    return next_level_conversion_data


def get_pixel_offset(first_level_feature):
    pass


def get_coord(features):
    coords = []
    for feature in features:
        coords.append(np.where(feature > 0)[0])
    return coords
