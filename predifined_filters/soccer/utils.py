import math
from copy import deepcopy

import numpy as np

# edges for field detection
from increased_number_edges.create_48_edge_kernels import ANGLES
from increased_number_edges.draw_48_edge_kernels import draw_edge_with_angle

KERNEL_SIZES_VERTICAL = np.arange(49, 89, 2, dtype=int)  # from 0 to 45 degree and from 135 to 179
KERNEL_SIZES_DIAGONAL = np.arange(51, 123, 2,
                                  dtype=int)  # from 46 to 77 and from 103 to 134: 50+45-45=51 and 90+77-45=122
KERNEL_SIZES_HORIZONTAL = np.arange(121, 179, 2, dtype=int)  # from 78 to 102
KERNEL_SIZES_TOTAL = np.arange(49, 179, 2, dtype=int)


def create_edge_kernels_for_soccer_with_parameters(iH, iW):
    # minimal_kernel_size = get_minimal_kernel_size(int(round(iH/3)))
    minimal_kernel_size = get_minimal_kernel_size(iH - 6)
    kernel_sizes = np.arange(minimal_kernel_size, iH, 2, dtype=int)
    kernelEdgeBank = {}
    for kernel_size in kernel_sizes:
        kernelEdgeBank[kernel_size] = create_edge_kernel_of_size_and_angle(kernel_size)
    return kernelEdgeBank, minimal_kernel_size


def get_minimal_kernel_size(size):
    if size % 2 == 0:
        kernel_size = size - 1
    else:
        kernel_size = size
    return kernel_size


def create_edge_kernels_for_soccer():
    kernelEdgeBank = {}
    for kernel_size in KERNEL_SIZES_TOTAL:
        kernelEdgeBank[kernel_size] = create_edge_kernel_of_size_and_angle(kernel_size)
    return kernelEdgeBank


def create_edge_kernel_of_size_and_angle(size):
    kernel_of_size_and_angle = []
    kernel_skeleton = np.zeros((size, size), dtype=int)
    for angle in ANGLES:
        kernel = deepcopy(kernel_skeleton)
        half_size = math.floor(size / 2)
        kernel_of_size_and_angle.append((angle, draw_edge_with_angle(kernel, half_size, angle)))
    return kernel_of_size_and_angle
