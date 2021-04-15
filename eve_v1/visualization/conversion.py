import numpy as np


def convert_to_picture(coords, feature_offsets):
    img = np.zeros((14, 14), dtype="float32")
    for feature_coords, feature_offset_raw in zip(coords, feature_offsets):
        for coord in feature_coords:
            for feature_offset in feature_offset_raw:
                x = coord[0]+feature_offset[2]
                y = coord[1]+feature_offset[1]
                transform_coords((x, y), feature_offset[0])
        # img()

    return img

def transform_coords(feature, anchor_coordinates):
    get_feature_pixel(feature)

def get_coord(features):
    coords = []
    for feature in features:
        coords.append(np.where(feature > 0)[0])
    return coords

def get_feature_pixel(feature):
    return
