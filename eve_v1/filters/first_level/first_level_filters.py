import matplotlib.pyplot as plt
import numpy as np

from eve_v1.visualization.vizualizer import viz_filter


def create_first_level_filters():
    filter_vals = np.array([
        [-1, 1],
        [0, 0],
    ])

    print('Filter shape: ', filter_vals.shape)
    # Defining the Filters
    filter_1 = filter_vals
    filter_2 = -filter_1
    filter_3 = filter_1.T
    filter_4 = -filter_3

    filter_D1 = np.array([
        [1, 0],
        [0, -1],
    ])

    filter_D2 = np.array([
        [0, 1],
        [-1, 0],
    ])

    filters = np.array([filter_1, filter_2, filter_3, filter_4, filter_D1, filter_D2])

    # viz_filter(filters)

    return filters
