import numpy as np


def generalize(layer, line_filers_number, angle_filters_number):
    reversed_features = int(len(layer) / 2)

    height, width = layer[0].shape
    generalized_feature = []
    generalized_feature_lines = np.zeros((height, width))
    generalized_feature_angles = np.zeros((height, width))

    for feature in layer[0:line_filers_number]:
        generalized_feature_lines = np.add(generalized_feature_lines, feature)

    generalized_feature.append(generalized_feature_lines)

    for i in range(line_filers_number, reversed_features):
        if ((i - line_filers_number) != 0) & ((i - line_filers_number) % angle_filters_number == 0):
            generalized_feature.append(generalized_feature_angles)
            generalized_feature_angles = layer[i]
        else:
            generalized_feature_angles = np.add(generalized_feature_angles, layer[i])
    generalized_feature.append(generalized_feature_angles)

    for feature in layer[reversed_features:reversed_features + line_filers_number]:
        generalized_feature_lines = np.add(generalized_feature_lines, feature)

    generalized_feature.append(generalized_feature_lines)

    start_point = reversed_features + line_filers_number
    for i in range(start_point, len(layer)):
        if ((i - start_point) != 0) & ((i - start_point) % angle_filters_number == 0):
            generalized_feature.append(generalized_feature_angles)
            generalized_feature_angles = layer[i]
        else:
            generalized_feature_angles = np.add(generalized_feature_angles, layer[i])
    generalized_feature.append(generalized_feature_angles)

    return np.concatenate((np.array(generalized_feature), layer), axis=0)
