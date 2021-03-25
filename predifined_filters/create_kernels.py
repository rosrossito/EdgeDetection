import math
from copy import deepcopy

import numpy as np

from predifined_filters.draw_kernels import draw_edge, draw_first_angle, draw_forth_angle, \
    draw_third_angle, draw_second_angle

SIZES = [3, 5, 7, 13, 21, 33, 55, 89, 143, 231, 375, 605, 719]
STEP = 1
EDGES = np.arange(24, dtype=int) * STEP
ANGLES = [0, 7.5, 15, 22.5, 30, 37.5, 45, 52.5, 60, 67.5, 75, 82.5, 90, 97.5, 105, 112.5, 120, 127.5, 135, 142.5, 150,
          157.5, 165, 172.5]
# ALL_EDGES = [EDGES[0], EDGES[1], EDGES[2], EDGES[3], EDGES[4], EDGES[5], EDGES[6], EDGES[7], EDGES[8], EDGES[9],
#                EDGES[10], EDGES[11], EDGES[12], EDGES[13], EDGES[14], EDGES[15], EDGES[16], EDGES[17], EDGES[18],
#                EDGES[19], EDGES[20], EDGES[21], EDGES[22], EDGES[23]],
# 0%, 7.5%, 15%, 22.5%, 30%, 37.5%, 45%, 52.5%, 60%, 67.5, 75%, 82.5%, 90%, 97.5%, 105%, 112.5%, 120%, 127.5%, 135%, 142.5%, 150%, 157.5%, 165%, 172.5%


SIZES_EDGES = (
    (SIZES[0], [EDGES[0], EDGES[6], EDGES[12], EDGES[18]]),  # 0%, 45%, 90%, 135%
    (SIZES[1], [EDGES[0], EDGES[3], EDGES[6], EDGES[9], EDGES[12], EDGES[15], EDGES[18], EDGES[21]]),
    # 0%, 22.5%, 45%, 67.5%, 90%, 112.5%, 135%, 157.5%
    (SIZES[2], [EDGES[0], EDGES[2], EDGES[4], EDGES[6], EDGES[8], EDGES[10], EDGES[12], EDGES[14], EDGES[16], EDGES[18],
                EDGES[20], EDGES[22]]),  # 0%, 15%, 30%, 45%, 60%, 75%, 90%, 105%, 120%, 135%, 150%, 165%
    (SIZES[3], EDGES),
    (SIZES[4], EDGES),
    (SIZES[5], EDGES),
    (SIZES[6], EDGES),
    (SIZES[7], EDGES),
    (SIZES[8], EDGES),
    (SIZES[9], EDGES),
    (SIZES[10], EDGES),
    (SIZES[11], EDGES),
    (SIZES[12], EDGES)
)


def create_edge_kernels():
    kernelEdgeBank = {}
    for (size, edges) in SIZES_EDGES:
        kernel_of_size = []
        kernel_skeleton = np.zeros((size, size), dtype=int)
        for edge in edges:
            kernel = deepcopy(kernel_skeleton)
            half_size = math.floor(size / 2)
            kernel_of_size.append((edge, draw_edge(kernel, half_size, edge)))
        kernelEdgeBank[size] = kernel_of_size
    return kernelEdgeBank


def create_angle_kernels(kernelEdgeBank):
    kernelCornerBank = {}
    # np.zeros((size, size), dtype=int)

    for size in SIZES:
        edge_kernel = kernelEdgeBank[size]
        kernel_of_size = []
        kernel_skeleton = np.zeros((size, size), dtype=int)
        for i in range(0, len(edge_kernel)):
            for j in range(i + 1, len(edge_kernel)):
                kernel = deepcopy(kernel_skeleton)
                half_size = math.floor(size / 2)
                first_edge, first_kernel = edge_kernel[i]
                second_edge, second_kernel = edge_kernel[j]
                kernel_of_size.append(
                    draw_first_angle(kernel, half_size, first_edge, second_edge))
                kernel_of_size.append(
                    draw_second_angle(kernel, half_size, first_edge, second_edge))
                kernel_of_size.append(
                    draw_third_angle(kernel, half_size, first_edge, second_edge))
                kernel_of_size.append(
                    draw_forth_angle(kernel, half_size, first_edge, second_edge))
        kernelCornerBank[size] = kernel_of_size
    return kernelCornerBank
