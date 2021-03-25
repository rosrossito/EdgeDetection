import torch
import numpy as np


def get_binary_feature_map(feature_map, threshold):
    binary_feature_map = []
    for map in feature_map:
        binary_feature_map.append(np.where(map < threshold, 0, 0.5))
    return np.array(binary_feature_map)

