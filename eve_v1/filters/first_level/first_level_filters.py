import matplotlib.pyplot as plt
import numpy as np

from eve_v1.visualization.vizualizer import viz_filter


def create_first_level_filters():

    # Defining the Filters
    filter_1 =  np.array([
        [-1, 1],
        [0, 0],
    ])

    filter_2 =  np.array([
        [1, -1],
        [0, 0],
    ])

    filter_3 =  np.array([
        [-1, 0],
        [1, 0],
    ])

    filter_4 = np.array([
        [1, 0],
        [-1, 0],
    ])

    filter_D1 = np.array([
        [1, 0],
        [0, -1],
    ])

    filter_D2 = np.array([
        [0, 1],
        [-1, 0],
    ])

    filter_D1_opposite = np.array([
        [-1, 0],
        [0, 1],
    ])

    filter_D2_opposite = np.array([
        [0, -1],
        [1, 0],
    ])

    filters = np.array([filter_1, filter_2, filter_3, filter_4, filter_D1, filter_D2, filter_D1_opposite,
                        filter_D2_opposite])

    # viz_filter(filters)

    return filters
