import numpy as np


def get_binary_feature_map(feature_map, threshold):
    binary_feature_map = []
    for feature in feature_map:
        binary_feature_map.append(np.where(feature < threshold, 0, 0.5))
    return np.array(binary_feature_map)


def get_binary_feature_map_with_different_thresholds(feature_map, thresholds):
    binary_feature_map = []
    for feature, threshold in zip(np.swapaxes(feature_map, 0, 1), thresholds):
        binary_feature_map.append(np.where(feature < threshold, 0, 0.5))
    return np.swapaxes(np.array(binary_feature_map), 0, 1)


def get_thresholds(filters):
    thresholds = []
    for filter in filters:
        thresholds.append(np.sum(filter)/2)
    return np.array(thresholds)
