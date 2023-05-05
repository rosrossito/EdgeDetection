from eve_v1.filters.first_level.elementary_feature_map import get_elementary_feature_map
from eve_v1.filters.fourth_level.fourth_level_feature_map import get_fourth_level_feature_map
from eve_v1.filters.generalization.generalization_service import concat_features
from eve_v1.filters.second_level.second_level_feature_map import get_second_level_feature_map
from eve_v1.filters.third_level.third_level_feature_map import get_third_level_feature_map
import numpy as np


def build_feature(gray_img_tensor):
    elementary_feature_map = get_elementary_feature_map(gray_img_tensor) # shape 12x14x14
    second_level_feature_with_generalization, second_level_feature_map, second_level_manually_created_features = \
        get_second_level_feature_map(elementary_feature_map)
    third_level_feature_with_generalization, third_level_feature_map, third_level_manually_created_features = \
        get_third_level_feature_map(second_level_feature_map, second_level_manually_created_features)

    # Todo: add 4-th level
    fourth_level_feature_with_generalization, fourth_level_feature_map = get_fourth_level_feature_map(
        third_level_feature_map, third_level_manually_created_features)


    return np.concatenate((elementary_feature_map,
                  second_level_feature_with_generalization,
                  third_level_feature_with_generalization), axis=1)

    # Todo Refine visualisation (add angles)
    # Todo Look at book - idea to make one more set of general features
    # maybe to merge similar features for the output (direct and reversed). They used in different places to build higher features
    # but represent the same
    # Todo: now we have generalization by angle, need to add generalization by direction (map of directions?) and
    #  spatial generalization (the same angle (or similar if some are absent - pyramid) in different layers) and
    #  maybe layer generalization
    # + affine transformation - incline
    # Todo Back to notes description and create CNN
