import numpy as np


def get_kernel(kernel_size, kernel_depth, position):
    kernel = np.zeros((kernel_depth, kernel_size, kernel_size), dtype=int)
    for pos in position:
        depth = pos[0] - 1
        y_pos = pos[1] - 1 #column
        x_pos = pos[2] - 1 #raw
        kernel[depth, x_pos, y_pos] = 1
    return kernel
