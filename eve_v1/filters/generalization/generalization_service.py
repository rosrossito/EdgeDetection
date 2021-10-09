import numpy as np


# direction generalization - rotation
def generalize_direction(layer, line_filters_number, angle_filters_number):
    reversed_features = int(len(layer) / 2)
    height, width = layer[0].shape
    generalized_feature = []
    generalized_feature_direction = np.zeros((height, width))

    # direction generalization is applied only to edges
    feature_numbers = reversed_features - line_filters_number
    for i in range(line_filters_number, line_filters_number + angle_filters_number):
        for k in range(i, feature_numbers, angle_filters_number):
            generalized_feature_direction = np.add(generalized_feature_direction, layer[k])
        generalized_feature.append(generalized_feature_direction)
        generalized_feature_direction = np.zeros((height, width))
    return np.array(generalized_feature)


# edge generalization - rotation
def generalize_angles(layer, line_filters_number, angle_filters_number):
    reversed_features = int(len(layer) / 2)

    height, width = layer[0].shape
    generalized_feature = []
    generalized_feature_lines = np.zeros((height, width))
    generalized_feature_angles = np.zeros((height, width))

    for feature in layer[0:line_filters_number]:
        generalized_feature_lines = np.add(generalized_feature_lines, feature)

    generalized_feature.append(generalized_feature_lines)

    for i in range(line_filters_number, reversed_features):
        if ((i - line_filters_number) != 0) & ((i - line_filters_number) % angle_filters_number == 0):
            generalized_feature.append(generalized_feature_angles)
            generalized_feature_angles = layer[i]
        else:
            generalized_feature_angles = np.add(generalized_feature_angles, layer[i])
    generalized_feature.append(generalized_feature_angles)

    generalized_feature_lines = np.zeros((height, width))
    generalized_feature_angles = np.zeros((height, width))

    for feature in layer[reversed_features:reversed_features + line_filters_number]:
        generalized_feature_lines = np.add(generalized_feature_lines, feature)

    generalized_feature.append(generalized_feature_lines)

    start_point = reversed_features + line_filters_number
    for i in range(start_point, len(layer)):
        if ((i - start_point) != 0) & ((i - start_point) % angle_filters_number == 0):
            generalized_feature.append(generalized_feature_angles)
            generalized_feature_angles = layer[i]
        else:
            generalized_feature_angles = np.add(generalized_feature_angles, layer[i])
    generalized_feature.append(generalized_feature_angles)

    return np.array(generalized_feature)

def concat_features(generalized_direction_feature, generalized_angle_feature, layer):
    return np.concatenate((generalized_direction_feature, generalized_angle_feature, layer), axis=0)
