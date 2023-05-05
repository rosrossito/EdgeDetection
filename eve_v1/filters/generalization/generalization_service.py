import numpy as np


# orientation generalization - rotation
def generalize_orientation(layer, line_filters_number, angle_filters_number):
    reversed_features = int(len(layer[0]) / 2)
    depth = len(layer)
    height, width = layer[0][0].shape
    generalized_feature = []
    generalized_feature_direction = np.zeros((depth, height, width))

    # orientation generalization applied to edges
    # edges are more important for defining image shape and its borders,
    # angles are more important for recognizing image (although the size of image)
    # Todo: add edges orientation generalization

    # orientation generalization applied to angles
    feature_numbers = reversed_features - line_filters_number
    for i in range(line_filters_number, line_filters_number + angle_filters_number):
        for k in range(i, feature_numbers, angle_filters_number):
            generalized_feature_direction = np.add(generalized_feature_direction, layer[:, k])
        generalized_feature.append(generalized_feature_direction)
        generalized_feature_direction = np.zeros((height, width))
    return np.swapaxes(np.array(generalized_feature), 0, 1)


# angles generalization - rotation
def generalize_angles(layer, line_filters_number, angle_filters_number):
    reversed_features = int(len(layer[0]) / 2)
    depth = len(layer)
    height, width = layer[0][0].shape
    generalized_feature = []
    generalized_feature_lines = np.zeros((depth, height, width))
    generalized_feature_angles = np.zeros((depth, height, width))

    swapped_layer = np.swapaxes(layer, 0, 1)
    for feature in swapped_layer[0:line_filters_number]:
        generalized_feature_lines = np.add(generalized_feature_lines, feature)

    generalized_feature.append(generalized_feature_lines)

    for i in range(line_filters_number, reversed_features):
        if ((i - line_filters_number) != 0) & ((i - line_filters_number) % angle_filters_number == 0):
            generalized_feature.append(generalized_feature_angles)
            generalized_feature_angles = layer[:, i]
        else:
            generalized_feature_angles = np.add(generalized_feature_angles, layer[:, i])
    generalized_feature.append(generalized_feature_angles)

    generalized_feature_lines = np.zeros((depth, height, width))
    generalized_feature_angles = np.zeros((depth, height, width))

    for feature in swapped_layer[reversed_features:reversed_features + line_filters_number]:
        generalized_feature_lines = np.add(generalized_feature_lines, feature)

    generalized_feature.append(generalized_feature_lines)

    start_point = reversed_features + line_filters_number
    for i in range(start_point, len(layer[0])):
        if ((i - start_point) != 0) & ((i - start_point) % angle_filters_number == 0):
            generalized_feature.append(generalized_feature_angles)
            generalized_feature_angles = layer[:, i]
        else:
            generalized_feature_angles = np.add(generalized_feature_angles, layer[:, i])
    generalized_feature.append(generalized_feature_angles)

    return np.swapaxes(np.array(generalized_feature), 0, 1)


def generalize_space(layer):
    # Horizontal generalization feature map (kernel)
    # Argumentation: to find the same combination of edges/angles despite the placement we need to
    # understand their relative to each other location (can be generalized for other dimensins - not only space (x,y axes)
    # but also direction and angles), so the goal is to encode difference between two cells. We can do that by encode
    # difference for each axes
    # Implementation:
    # Helping map:
    # First raw of map - sum up vertical columns of kernel (with size nxn), than build the columns that reflect
    # activation of every two feature from first raw:
    # - in the first column: first and second, first and third etc.
    # - in the second column: second and third, second and forth etc.
    # In other words:
    # - in the first raw: difference equal to one: first and second, second and third etc.
    # - in the second raw: difference equal to two: first and third, second and forth etc.
    # - etc.
    # After that sum up raws and receive vector of horizontal location differences with size nx1
    # First of all applied to generalized features: direction and angles
    # In such way we can generalize any feature (kernel) but there is no necessity to apply this to all kernels.
    # Because the same as space we encode by 2 attributes - x and y, we can encode every angle/edge by their
    # direction and kind (e.a. angles degrees, edge actually has 180 degrees))
    # To preserve kernel size we can add just zero elements

    depth = len(layer)
    height, width = layer[0][0].shape
    generalized_feature = []
    swapped_layer = np.swapaxes(layer, 0, 1)

    for feature in swapped_layer:
        generalized_feature_space = np.zeros((depth, height, width))

        columns = np.sum(feature, axis=1)
        for period in range(1, columns[0].size):
            value = np.zeros((depth))
            for i in range(period, columns[0].size):
                value = value + columns[:, i] + columns[:, i - period]
            np.swapaxes(generalized_feature_space, 0, 2)[period-1][0] = value


        # Vertical generalization (the same strategy but using not column but raws)
        raws = np.sum(feature, axis=2)
        for period in range(1, raws[0].size):
            value = np.zeros((depth))
            for i in range(period, raws[0].size):
                value = value + raws[:, i] + raws[:, i - period]
            np.swapaxes(generalized_feature_space, 0, 2)[period-1][1] = value

        generalized_feature.append(generalized_feature_space)

    return np.swapaxes(np.array(generalized_feature), 0, 1)


def concat_features(generalized_direction_feature,
                    generalized_kind_feature,
                    generalized_space_orientation_feature,
                    generalized_space_kind_feature,
                    layer):
    return np.concatenate((generalized_direction_feature,
                           generalized_kind_feature,
                           generalized_space_orientation_feature,
                           generalized_space_kind_feature,
                           layer), axis=1)
