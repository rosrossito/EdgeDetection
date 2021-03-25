LINE_45 = 45
LINE_90 = 90
LINE_135 = 135
LINE_180 = 180

ANGLES = [0, 3.75, 7.5, 11.25, 15, 18.75, 22.5, 26.25, 30, 33.75,
          37.5, 41.25, 45, 48.75, 52.5, 56.25, 60, 63.75, 67.5, 71.25,
          75, 78.75, 82.5, 86.25, 90, 93.75, 97.5, 101.25, 105, 108.75,
          112.5, 116.25, 120, 123.75, 127.5, 131.25, 135, 138.75, 142.5, 146.25,
          150, 153.75, 157.5, 161.25, 165, 168.75, 172.5, 176.25]


def draw_edge_with_angle(kernel, half_size, angle):
    kernel[half_size, half_size] = 1
    kernel_coordinates_positive = calculate_kernel_coordinates_with_angle_positive(angle, half_size)
    draw(kernel, kernel_coordinates_positive, half_size, "", True)
    kernel_coordinates_negative = calculate_kernel_coordinates_with_angle_negative(angle, half_size)
    draw_negatives(kernel, kernel_coordinates_negative, half_size, True)
    return kernel


def draw_edge(kernel, half_size, edge_number):
    continue_draw = True
    size_counter = 0
    x_up = half_size
    y_up = half_size
    while continue_draw:
        kernel[x_up, y_up] = 1
        if abs(x_up - half_size) == half_size or abs(y_up - half_size) == half_size:
            break
        kernel_coordinates_up, x_up, y_up = calculate_kernel_coordinates_up(edge_number, x_up, y_up)
        continue_draw = draw(kernel, kernel_coordinates_up, half_size, size_counter, continue_draw)

    continue_draw = True
    size_counter = 0
    x_down = half_size
    y_down = half_size
    while continue_draw:
        kernel[x_down, y_down] = 1
        if abs(x_down - half_size) == half_size or abs(y_down - half_size) == half_size:
            break
        kernel_coordinates_down, x_down, y_down = calculate_kernel_coordinates_down(edge_number, x_down, y_down)
        continue_draw = draw(kernel, kernel_coordinates_down, half_size, size_counter, continue_draw)

    return kernel


def draw_first_angle(kernel, half_size, first_edge, second_edge):
    continue_draw = True
    size_counter = 0
    x_first_edge = half_size
    y_first_edge = half_size
    while continue_draw:
        kernel[x_first_edge, y_first_edge] = 1
        if abs(x_first_edge - half_size) == half_size or abs(y_first_edge - half_size) == half_size:
            break
        kernel_coordinates_first_edge, x_first_edge, y_first_edge = calculate_kernel_coordinates_up(first_edge,
                                                                                                    x_first_edge,
                                                                                                    y_first_edge)
        continue_draw = draw(kernel, kernel_coordinates_first_edge, half_size, size_counter, continue_draw)

    continue_draw = True
    size_counter = 0
    x_second_edge = half_size
    y_second_edge = half_size
    while continue_draw:
        kernel[x_second_edge, y_second_edge] = 1
        if abs(x_second_edge - half_size) == half_size or abs(y_second_edge - half_size) == half_size:
            break
        kernel_coordinates_second_edge, x_second_edge, y_second_edge = calculate_kernel_coordinates_up(second_edge,
                                                                                                       x_second_edge,
                                                                                                       y_second_edge)
        continue_draw = draw(kernel, kernel_coordinates_second_edge, half_size, size_counter, continue_draw)
    # return direction, angles degree, kernel
    return ANGLES[first_edge], ANGLES[second_edge] - ANGLES[first_edge], kernel


def draw_second_angle(kernel, half_size, first_edge, second_edge):
    continue_draw = True
    size_counter = 0
    x_first_edge = half_size
    y_first_edge = half_size
    while continue_draw:
        kernel[x_first_edge, y_first_edge] = 1
        if abs(x_first_edge - half_size) == half_size or abs(y_first_edge - half_size) == half_size:
            break
        kernel_coordinates_first_edge, x_first_edge, y_first_edge = calculate_kernel_coordinates_down(first_edge,
                                                                                                      x_first_edge,
                                                                                                      y_first_edge)
        continue_draw = draw(kernel, kernel_coordinates_first_edge, half_size, size_counter, continue_draw)

    continue_draw = True
    size_counter = 0
    x_second_edge = half_size
    y_second_edge = half_size
    while continue_draw:
        kernel[x_second_edge, y_second_edge] = 1
        if abs(x_second_edge - half_size) == half_size or abs(y_second_edge - half_size) == half_size:
            break
        kernel_coordinates_second_edge, x_second_edge, y_second_edge = calculate_kernel_coordinates_down(second_edge,
                                                                                                         x_second_edge,
                                                                                                         y_second_edge)
        continue_draw = draw(kernel, kernel_coordinates_second_edge, half_size, size_counter, continue_draw)
    return ANGLES[first_edge] + 180, (ANGLES[second_edge] + 180) - (ANGLES[first_edge] + 180), kernel


def draw_third_angle(kernel, half_size, first_edge, second_edge):
    continue_draw = True
    size_counter = 0
    x_first_edge = half_size
    y_first_edge = half_size
    while continue_draw:
        kernel[x_first_edge, y_first_edge] = 1
        if abs(x_first_edge - half_size) == half_size or abs(y_first_edge - half_size) == half_size:
            break
        kernel_coordinates_first_edge, x_first_edge, y_first_edge = calculate_kernel_coordinates_up(first_edge,
                                                                                                    x_first_edge,
                                                                                                    y_first_edge)
        continue_draw = draw(kernel, kernel_coordinates_first_edge, half_size, size_counter, continue_draw)

    continue_draw = True
    size_counter = 0
    x_second_edge = half_size
    y_second_edge = half_size
    while continue_draw:
        kernel[x_second_edge, y_second_edge] = 1
        if abs(x_second_edge - half_size) == half_size or abs(y_second_edge - half_size) == half_size:
            break
        kernel_coordinates_second_edge, x_second_edge, y_second_edge = calculate_kernel_coordinates_down(second_edge,
                                                                                                         x_second_edge,
                                                                                                         y_second_edge)
        continue_draw = draw(kernel, kernel_coordinates_second_edge, half_size, size_counter, continue_draw)
    return ANGLES[first_edge], (ANGLES[second_edge] + 180) - ANGLES[first_edge], kernel


def draw_forth_angle(kernel, half_size, first_edge, second_edge):
    continue_draw = True
    size_counter = 0
    x_first_edge = half_size
    y_first_edge = half_size
    while continue_draw:
        kernel[x_first_edge, y_first_edge] = 1
        if abs(x_first_edge - half_size) == half_size or abs(y_first_edge - half_size) == half_size:
            break
        kernel_coordinates_first_edge, x_first_edge, y_first_edge = calculate_kernel_coordinates_down(first_edge,
                                                                                                      x_first_edge,
                                                                                                      y_first_edge)
        continue_draw = draw(kernel, kernel_coordinates_first_edge, half_size, size_counter, continue_draw)

    continue_draw = True
    size_counter = 0
    x_second_edge = half_size
    y_second_edge = half_size
    while continue_draw:
        kernel[x_second_edge, y_second_edge] = 1
        if abs(x_second_edge - half_size) == half_size or abs(y_second_edge - half_size) == half_size:
            break
        kernel_coordinates_second_edge, x_second_edge, y_second_edge = calculate_kernel_coordinates_up(second_edge,
                                                                                                       x_second_edge,
                                                                                                       y_second_edge)
        continue_draw = draw(kernel, kernel_coordinates_second_edge, half_size, size_counter, continue_draw)
    return ANGLES[second_edge], (ANGLES[first_edge] + 180) - ANGLES[second_edge], kernel


def draw(kernel, kernel_coordinates, half_size, size_counter, continue_draw):
    for x_coor, y_coor in kernel_coordinates:
        kernel[x_coor, y_coor] = 1
        if abs(x_coor - half_size) == half_size or abs(y_coor - half_size) == half_size:
            continue_draw = False
            break
    return continue_draw


def draw_negatives(kernel, kernel_coordinates, half_size, continue_draw):
    kernel[half_size + 1, half_size + 1] = -1
    for x_coor, y_coor in kernel_coordinates:
        kernel[x_coor, y_coor] = -1
        if abs(x_coor - half_size) == half_size or abs(y_coor - half_size) == half_size:
            continue_draw = False
            break
    return continue_draw


def calculate_kernel_coordinates_up(edge_number, x_up, y_up):
    if edge_number == 0:
        kernel_coordinates_up = (
            (x_up - 1, y_up), (x_up - 2, y_up), (x_up - 3, y_up), (x_up - 4, y_up),
            (x_up - 5, y_up))
        x_up = x_up - 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 1:
        kernel_coordinates_up = (
            (x_up - 1, y_up), (x_up - 2, y_up), (x_up - 3, y_up), (x_up - 4, y_up), (x_up - 5, y_up), (x_up - 6, y_up),
            (x_up - 6, y_up + 1), (x_up - 7, y_up + 1), (x_up - 8, y_up + 1), (x_up - 9, y_up + 1),
            (x_up - 10, y_up + 1), (x_up - 11, y_up + 1))
        x_up = x_up - 12
        y_up = y_up + 1
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 2:
        kernel_coordinates_up = (
            (x_up - 1, y_up), (x_up - 2, y_up), (x_up - 3, y_up), (x_up - 3, y_up + 1),
            (x_up - 4, y_up + 1), (x_up - 5, y_up + 1))
        x_up = x_up - 6
        y_up = y_up + 1
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 3:
        kernel_coordinates_up = (
            (x_up - 1, y_up), (x_up - 2, y_up), (x_up - 2, y_up + 1), (x_up - 3, y_up + 1), (x_up - 4, y_up + 1),
            (x_up - 5, y_up + 1),
            (x_up - 6, y_up + 1), (x_up - 6, y_up + 2), (x_up - 7, y_up + 2), (x_up - 8, y_up + 2),
            (x_up - 9, y_up + 2),
            (x_up - 10, y_up + 2), (x_up - 10, y_up + 3), (x_up - 11, y_up + 3))
        x_up = x_up - 12
        y_up = y_up + 3
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 4:
        kernel_coordinates_up = (
            (x_up - 1, y_up), (x_up - 2, y_up + 1), (x_up - 3, y_up + 1), (x_up - 4, y_up + 1),
            (x_up - 5, y_up + 2))
        x_up = x_up - 6
        y_up = y_up + 2
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 5:
        kernel_coordinates_up = (
            (x_up - 1, y_up), (x_up - 1, y_up + 1), (x_up - 2, y_up + 1), (x_up - 3, y_up + 1), (x_up - 4, y_up + 2),
            (x_up - 5, y_up + 2), (x_up - 6, y_up + 2), (x_up - 6, y_up + 3), (x_up - 7, y_up + 3),
            (x_up - 8, y_up + 3),
            (x_up - 9, y_up + 4), (x_up - 10, y_up + 4), (x_up - 11, y_up + 4), (x_up - 11, y_up + 5)
        )
        x_up = x_up - 12
        y_up = y_up + 5
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 6:
        kernel_coordinates_up = (
            (x_up - 1, y_up), (x_up - 1, y_up + 1), (x_up - 2, y_up + 1), (x_up - 3, y_up + 1),
            (x_up - 3, y_up + 2),
            (x_up - 4, y_up + 2), (x_up - 5, y_up + 2), (x_up - 5, y_up + 3))
        x_up = x_up - 6
        y_up = y_up + 3
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 7:
        kernel_coordinates_up = (
            (x_up - 1, y_up), (x_up - 1, y_up + 1), (x_up - 2, y_up + 1), (x_up - 3, y_up + 2),
            (x_up - 4, y_up + 2), (x_up - 4, y_up + 3), (x_up - 5, y_up + 3), (x_up - 6, y_up + 3),
            (x_up - 6, y_up + 4), (x_up - 7, y_up + 4), (x_up - 8, y_up + 4), (x_up - 8, y_up + 5),
            (x_up - 9, y_up + 5), (x_up - 10, y_up + 6), (x_up - 11, y_up + 6), (x_up - 11, y_up + 7)
        )
        x_up = x_up - 12
        y_up = y_up + 7
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 8:
        kernel_coordinates_up = (
            (x_up - 1, y_up), (x_up - 1, y_up + 1), (x_up - 2, y_up + 1), (x_up - 2, y_up + 2),
            (x_up - 3, y_up + 2),
            (x_up - 4, y_up + 2), (x_up - 4, y_up + 3), (x_up - 5, y_up + 3), (x_up - 5, y_up + 4))
        x_up = x_up - 6
        y_up = y_up + 4
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 9:
        kernel_coordinates_up = (
            (x_up - 1, y_up), (x_up - 1, y_up + 1), (x_up - 2, y_up + 1), (x_up - 2, y_up + 2),
            (x_up - 3, y_up + 2), (x_up - 3, y_up + 3), (x_up - 4, y_up + 3),
            (x_up - 5, y_up + 4), (x_up - 6, y_up + 4), (x_up - 6, y_up + 5), (x_up - 7, y_up + 5),
            (x_up - 7, y_up + 6), (x_up - 8, y_up + 6), (x_up - 9, y_up + 7), (x_up - 10, y_up + 7),
            (x_up - 10, y_up + 8), (x_up - 11, y_up + 8), (x_up - 11, y_up + 9)
        )
        x_up = x_up - 12
        y_up = y_up + 9
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 10:
        kernel_coordinates_up = (
            (x_up - 1, y_up), (x_up - 1, y_up + 1), (x_up - 2, y_up + 1), (x_up - 2, y_up + 2),
            (x_up - 3, y_up + 2),
            (x_up - 3, y_up + 3), (x_up - 4, y_up + 3), (x_up - 4, y_up + 4), (x_up - 5, y_up + 4),
            (x_up - 5, y_up + 5))
        x_up = x_up - 6
        y_up = y_up + 5
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 11:
        kernel_coordinates_up = (
            (x_up - 1, y_up), (x_up - 1, y_up + 1), (x_up - 2, y_up + 1), (x_up - 2, y_up + 2),
            (x_up - 3, y_up + 3),
            (x_up - 4, y_up + 3), (x_up - 4, y_up + 4), (x_up - 5, y_up + 4), (x_up - 5, y_up + 5),
            (x_up - 6, y_up + 5), (x_up - 6, y_up + 6), (x_up - 7, y_up + 6), (x_up - 7, y_up + 7),
            (x_up - 8, y_up + 7), (x_up - 8, y_up + 8), (x_up - 9, y_up + 9), (x_up - 10, y_up + 9),
            (x_up - 10, y_up + 10), (x_up - 11, y_up + 10), (x_up - 11, y_up + 11))
        x_up = x_up - 12
        y_up = y_up + 11
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 12:
        kernel_coordinates_up = (
            (x_up - 1, y_up + 1), (x_up - 2, y_up + 2), (x_up - 3, y_up + 3), (x_up - 4, y_up + 4),
            (x_up - 5, y_up + 5))
        x_up = x_up - 6
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 13:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up - 1, y_up + 1), (x_up - 1, y_up + 2), (x_up - 2, y_up + 2),
            (x_up - 3, y_up + 3),
            (x_up - 3, y_up + 4), (x_up - 4, y_up + 4), (x_up - 4, y_up + 5), (x_up - 5, y_up + 5),
            (x_up - 5, y_up + 6), (x_up - 6, y_up + 6), (x_up - 6, y_up + 7), (x_up - 7, y_up + 7),
            (x_up - 7, y_up + 8), (x_up - 8, y_up + 8), (x_up - 9, y_up + 9), (x_up - 9, y_up + 10),
            (x_up - 10, y_up + 10), (x_up - 10, y_up + 11), (x_up - 11, y_up + 11))
        x_up = x_up - 11
        y_up = y_up + 12
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 14:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up - 1, y_up + 1), (x_up - 1, y_up + 2), (x_up - 2, y_up + 2),
            (x_up - 2, y_up + 3), (x_up - 3, y_up + 3), (x_up - 3, y_up + 4), (x_up - 4, y_up + 4),
            (x_up - 4, y_up + 5), (x_up - 5, y_up + 5))
        x_up = x_up - 5
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 15:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up - 1, y_up + 1), (x_up - 1, y_up + 2), (x_up - 2, y_up + 2),
            (x_up - 2, y_up + 3), (x_up - 3, y_up + 3), (x_up - 3, y_up + 4),
            (x_up - 4, y_up + 5), (x_up - 4, y_up + 6), (x_up - 5, y_up + 6), (x_up - 5, y_up + 7),
            (x_up - 6, y_up + 7), (x_up - 6, y_up + 8), (x_up - 7, y_up + 9), (x_up - 7, y_up + 10),
            (x_up - 8, y_up + 10), (x_up - 8, y_up + 11), (x_up - 9, y_up + 11)
        )
        x_up = x_up - 9
        y_up = y_up + 12
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 16:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up - 1, y_up + 1), (x_up - 1, y_up + 2), (x_up - 2, y_up + 2),
            (x_up - 2, y_up + 3), (x_up - 2, y_up + 4), (x_up - 3, y_up + 4), (x_up - 3, y_up + 5),
            (x_up - 4, y_up + 5))
        x_up = x_up - 4
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 17:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up - 1, y_up + 1), (x_up - 1, y_up + 2), (x_up - 2, y_up + 3),
            (x_up - 2, y_up + 4), (x_up - 3, y_up + 4), (x_up - 3, y_up + 5), (x_up - 3, y_up + 6),
            (x_up - 4, y_up + 6), (x_up - 4, y_up + 7), (x_up - 4, y_up + 8), (x_up - 5, y_up + 8),
            (x_up - 5, y_up + 9), (x_up - 6, y_up + 10), (x_up - 6, y_up + 11), (x_up - 7, y_up + 11)
        )
        x_up = x_up - 7
        y_up = y_up + 12
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 18:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up - 1, y_up + 1), (x_up - 1, y_up + 2), (x_up - 1, y_up + 3),
            (x_up - 2, y_up + 3), (x_up - 2, y_up + 4), (x_up - 2, y_up + 5), (x_up - 3, y_up + 5))
        x_up = x_up - 3
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 19:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up - 1, y_up + 1), (x_up - 1, y_up + 2), (x_up - 1, y_up + 3), (x_up - 2, y_up + 4),
            (x_up - 2, y_up + 5), (x_up - 2, y_up + 6), (x_up - 3, y_up + 6), (x_up - 3, y_up + 7),
            (x_up - 3, y_up + 8),
            (x_up - 4, y_up + 9), (x_up - 4, y_up + 10), (x_up - 4, y_up + 11), (x_up - 5, y_up + 11)
        )
        x_up = x_up - 5
        y_up = y_up + 12
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 20:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up - 1, y_up + 2), (x_up - 1, y_up + 3), (x_up - 1, y_up + 4),
            (x_up - 2, y_up + 5))
        x_up = x_up - 2
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 21:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up, y_up + 2), (x_up - 1, y_up + 2), (x_up - 1, y_up + 3), (x_up - 1, y_up + 4),
            (x_up - 1, y_up + 5),
            (x_up - 1, y_up + 6), (x_up - 2, y_up + 6), (x_up - 2, y_up + 7), (x_up - 2, y_up + 8),
            (x_up - 2, y_up + 9),
            (x_up - 2, y_up + 10), (x_up - 3, y_up + 10), (x_up - 3, y_up + 11))
        x_up = x_up - 3
        y_up = y_up + 12
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 22:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up, y_up + 2), (x_up, y_up + 3), (x_up - 1, y_up + 3),
            (x_up - 1, y_up + 4),
            (x_up - 1, y_up + 5))
        x_up = x_up - 1
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 23:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up, y_up + 2), (x_up, y_up + 3), (x_up, y_up + 4), (x_up, y_up + 5), (x_up, y_up + 6),
            (x_up - 1, y_up + 6), (x_up - 1, y_up + 7), (x_up - 1, y_up + 8), (x_up - 1, y_up + 9),
            (x_up - 1, y_up + 10), (x_up - 1, y_up + 11))
        x_up = x_up - 1
        y_up = y_up + 12
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 24:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up, y_up + 2), (x_up, y_up + 3), (x_up, y_up + 4), (x_up, y_up + 5))
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 25:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up, y_up + 2), (x_up, y_up + 3), (x_up, y_up + 4), (x_up, y_up + 5), (x_up, y_up + 6),
            (x_up + 1, y_up + 6), (x_up + 1, y_up + 7), (x_up + 1, y_up + 8), (x_up + 1, y_up + 9),
            (x_up + 1, y_up + 10), (x_up + 1, y_up + 11))
        x_up = x_up + 1
        y_up = y_up + 12
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 26:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up, y_up + 2), (x_up, y_up + 3), (x_up + 1, y_up + 3), (x_up + 1, y_up + 4),
            (x_up + 1, y_up + 5))
        x_up = x_up + 1
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 27:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up, y_up + 2), (x_up + 1, y_up + 2), (x_up + 1, y_up + 3), (x_up + 1, y_up + 4),
            (x_up + 1, y_up + 5),
            (x_up + 1, y_up + 6), (x_up + 2, y_up + 6), (x_up + 2, y_up + 7), (x_up + 2, y_up + 8),
            (x_up + 2, y_up + 9),
            (x_up + 2, y_up + 10), (x_up + 3, y_up + 10), (x_up + 3, y_up + 11))
        x_up = x_up + 3
        y_up = y_up + 12
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 28:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up + 1, y_up + 2), (x_up + 1, y_up + 3), (x_up + 1, y_up + 4),
            (x_up + 2, y_up + 5))
        x_up = x_up + 2
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 29:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up + 1, y_up + 1), (x_up + 1, y_up + 2), (x_up + 1, y_up + 3), (x_up + 2, y_up + 4),
            (x_up + 2, y_up + 5), (x_up + 2, y_up + 6), (x_up + 3, y_up + 6), (x_up + 3, y_up + 7),
            (x_up + 3, y_up + 8),
            (x_up + 4, y_up + 9), (x_up + 4, y_up + 10), (x_up + 4, y_up + 11), (x_up + 5, y_up + 11)
        )
        x_up = x_up + 5
        y_up = y_up + 12
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 30:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up + 1, y_up + 1), (x_up + 1, y_up + 2), (x_up + 1, y_up + 3),
            (x_up + 2, y_up + 3), (x_up + 2, y_up + 4), (x_up + 2, y_up + 5), (x_up + 3, y_up + 5))
        x_up = x_up + 3
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 31:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up + 1, y_up + 1), (x_up + 1, y_up + 2), (x_up + 2, y_up + 3),
            (x_up + 2, y_up + 4), (x_up + 3, y_up + 4), (x_up + 3, y_up + 5), (x_up + 3, y_up + 6),
            (x_up + 4, y_up + 6), (x_up + 4, y_up + 7), (x_up + 4, y_up + 8), (x_up + 5, y_up + 8),
            (x_up + 5, y_up + 9), (x_up + 6, y_up + 10), (x_up + 6, y_up + 11), (x_up + 7, y_up + 11)
        )
        x_up = x_up + 7
        y_up = y_up + 12
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 32:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up + 1, y_up + 1), (x_up + 1, y_up + 2), (x_up + 2, y_up + 2),
            (x_up + 2, y_up + 3), (x_up + 2, y_up + 4), (x_up + 3, y_up + 4), (x_up + 3, y_up + 5),
            (x_up + 4, y_up + 5))
        x_up = x_up + 4
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 33:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up + 1, y_up + 1), (x_up + 1, y_up + 2), (x_up + 2, y_up + 2),
            (x_up + 2, y_up + 3), (x_up + 3, y_up + 3), (x_up + 3, y_up + 4),
            (x_up + 4, y_up + 5), (x_up + 4, y_up + 6), (x_up + 5, y_up + 6), (x_up + 5, y_up + 7),
            (x_up + 6, y_up + 7), (x_up + 6, y_up + 8), (x_up + 7, y_up + 9), (x_up + 7, y_up + 10),
            (x_up + 8, y_up + 10), (x_up + 8, y_up + 11), (x_up + 9, y_up + 11)
        )
        x_up = x_up + 9
        y_up = y_up + 12
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 34:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up + 1, y_up + 1), (x_up + 1, y_up + 2), (x_up + 2, y_up + 2),
            (x_up + 2, y_up + 3), (x_up + 3, y_up + 3), (x_up + 3, y_up + 4), (x_up + 4, y_up + 4),
            (x_up + 4, y_up + 5), (x_up + 5, y_up + 5))
        x_up = x_up + 5
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 35:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up + 1, y_up + 1), (x_up + 1, y_up + 2), (x_up + 2, y_up + 2),
            (x_up + 3, y_up + 3),
            (x_up + 3, y_up + 4), (x_up + 4, y_up + 4), (x_up + 4, y_up + 5), (x_up + 5, y_up + 5),
            (x_up + 5, y_up + 6), (x_up + 6, y_up + 6), (x_up + 6, y_up + 7), (x_up + 7, y_up + 7),
            (x_up + 7, y_up + 8), (x_up + 8, y_up + 8), (x_up + 9, y_up + 9), (x_up + 9, y_up + 10),
            (x_up + 10, y_up + 10), (x_up + 10, y_up + 11), (x_up + 11, y_up + 11))
        x_up = x_up + 11
        y_up = y_up + 12
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 36:
        kernel_coordinates_up = (
            (x_up + 1, y_up + 1), (x_up + 2, y_up + 2), (x_up + 3, y_up + 3), (x_up + 4, y_up + 4),
            (x_up + 5, y_up + 5))
        x_up = x_up + 6
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 37:
        kernel_coordinates_up = (
            (x_up + 1, y_up), (x_up + 1, y_up + 1), (x_up + 2, y_up + 1), (x_up + 2, y_up + 2),
            (x_up + 3, y_up + 3),
            (x_up + 4, y_up + 3), (x_up + 4, y_up + 4), (x_up + 5, y_up + 4), (x_up + 5, y_up + 5),
            (x_up + 6, y_up + 5), (x_up + 6, y_up + 6), (x_up + 7, y_up + 6), (x_up + 7, y_up + 7),
            (x_up + 8, y_up + 7), (x_up + 8, y_up + 8), (x_up + 9, y_up + 9), (x_up + 10, y_up + 9),
            (x_up + 10, y_up + 10), (x_up + 11, y_up + 10), (x_up + 11, y_up + 11))
        x_up = x_up + 12
        y_up = y_up + 11
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 38:
        kernel_coordinates_up = (
            (x_up + 1, y_up), (x_up + 1, y_up + 1), (x_up + 2, y_up + 1), (x_up + 2, y_up + 2), (x_up + 3, y_up + 2),
            (x_up + 3, y_up + 3), (x_up + 4, y_up + 3), (x_up + 4, y_up + 4), (x_up + 5, y_up + 4),
            (x_up + 5, y_up + 5))
        x_up = x_up + 6
        y_up = y_up + 5
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 39:
        kernel_coordinates_up = (
            (x_up + 1, y_up), (x_up + 1, y_up + 1), (x_up + 2, y_up + 1), (x_up + 2, y_up + 2),
            (x_up + 3, y_up + 2), (x_up + 3, y_up + 3), (x_up + 4, y_up + 3),
            (x_up + 5, y_up + 4), (x_up + 6, y_up + 4), (x_up + 6, y_up + 5), (x_up + 7, y_up + 5),
            (x_up + 7, y_up + 6), (x_up + 8, y_up + 6), (x_up + 9, y_up + 7), (x_up + 10, y_up + 7),
            (x_up + 10, y_up + 8), (x_up + 11, y_up + 8), (x_up + 11, y_up + 9)
        )
        x_up = x_up + 12
        y_up = y_up + 9
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 40:
        kernel_coordinates_up = (
            (x_up + 1, y_up), (x_up + 1, y_up + 1), (x_up + 2, y_up + 1), (x_up + 2, y_up + 2), (x_up + 3, y_up + 2),
            (x_up + 4, y_up + 2), (x_up + 4, y_up + 3), (x_up + 5, y_up + 3), (x_up + 5, y_up + 4))
        x_up = x_up + 6
        y_up = y_up + 4
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 41:
        kernel_coordinates_up = (
            (x_up + 1, y_up), (x_up + 1, y_up + 1), (x_up + 2, y_up + 1), (x_up + 3, y_up + 2),
            (x_up + 4, y_up + 2), (x_up + 4, y_up + 3), (x_up + 5, y_up + 3), (x_up + 6, y_up + 3),
            (x_up + 6, y_up + 4), (x_up + 7, y_up + 4), (x_up + 8, y_up + 4), (x_up + 8, y_up + 5),
            (x_up + 9, y_up + 5), (x_up + 10, y_up + 6), (x_up + 11, y_up + 6), (x_up + 11, y_up + 7)
        )
        x_up = x_up + 12
        y_up = y_up + 7
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 42:
        kernel_coordinates_up = (
            (x_up + 1, y_up), (x_up + 1, y_up + 1), (x_up + 2, y_up + 1), (x_up + 3, y_up + 1), (x_up + 3, y_up + 2),
            (x_up + 4, y_up + 2), (x_up + 5, y_up + 2), (x_up + 5, y_up + 3))
        x_up = x_up + 6
        y_up = y_up + 3
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 43:
        kernel_coordinates_up = (
            (x_up + 1, y_up), (x_up + 1, y_up + 1), (x_up + 2, y_up + 1), (x_up + 3, y_up + 1), (x_up + 4, y_up + 2),
            (x_up + 5, y_up + 2), (x_up + 6, y_up + 2), (x_up + 6, y_up + 3), (x_up + 7, y_up + 3),
            (x_up + 8, y_up + 3),
            (x_up + 9, y_up + 4), (x_up + 10, y_up + 4), (x_up + 11, y_up + 4), (x_up + 11, y_up + 5)
        )
        x_up = x_up + 12
        y_up = y_up + 5
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 44:
        kernel_coordinates_up = (
            (x_up + 1, y_up), (x_up + 2, y_up + 1), (x_up + 3, y_up + 1), (x_up + 4, y_up + 1), (x_up + 5, y_up + 2))
        x_up = x_up + 6
        y_up = y_up + 2
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 45:
        kernel_coordinates_up = (
            (x_up + 1, y_up), (x_up + 2, y_up), (x_up + 2, y_up + 1), (x_up + 3, y_up + 1), (x_up + 4, y_up + 1),
            (x_up + 5, y_up + 1),
            (x_up + 6, y_up + 1), (x_up + 6, y_up + 2), (x_up + 7, y_up + 2), (x_up + 8, y_up + 2),
            (x_up + 9, y_up + 2),
            (x_up + 10, y_up + 2), (x_up + 10, y_up + 3), (x_up + 11, y_up + 3))
        x_up = x_up + 12
        y_up = y_up + 3
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 46:
        kernel_coordinates_up = (
            (x_up + 1, y_up), (x_up + 2, y_up), (x_up + 3, y_up), (x_up + 3, y_up + 1), (x_up + 4, y_up + 1),
            (x_up + 5, y_up + 1))
        x_up = x_up + 6
        y_up = y_up + 1
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 47:
        kernel_coordinates_up = (
            (x_up + 1, y_up), (x_up + 2, y_up), (x_up + 3, y_up), (x_up + 4, y_up), (x_up + 5, y_up), (x_up + 6, y_up),
            (x_up + 6, y_up + 1), (x_up + 7, y_up + 1), (x_up + 8, y_up + 1), (x_up + 9, y_up + 1),
            (x_up + 10, y_up + 1), (x_up + 11, y_up + 1))
        x_up = x_up + 12
        y_up = y_up + 1
        return kernel_coordinates_up, x_up, y_up


def calculate_kernel_coordinates_down(edge_number, x_down, y_down):
    if edge_number == 0:
        kernel_coordinates_down = (
            (x_down + 1, y_down), (x_down + 2, y_down), (x_down + 3, y_down), (x_down + 4, y_down),
            (x_down + 5, y_down))
        x_down = x_down + 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 1:
        kernel_coordinates_down = (
            (x_down + 1, y_down), (x_down + 2, y_down), (x_down + 3, y_down), (x_down + 4, y_down),
            (x_down + 5, y_down), (x_down + 6, y_down),
            (x_down + 6, y_down - 1), (x_down + 7, y_down - 1), (x_down + 8, y_down - 1), (x_down + 9, y_down - 1),
            (x_down + 10, y_down - 1), (x_down + 11, y_down - 1))
        x_down = x_down + 12
        y_down = y_down - 1
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 2:
        kernel_coordinates_down = (
            (x_down + 1, y_down), (x_down + 2, y_down), (x_down + 3, y_down), (x_down + 3, y_down - 1),
            (x_down + 4, y_down - 1), (x_down + 5, y_down - 1))
        x_down = x_down + 6
        y_down = y_down - 1
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 3:
        kernel_coordinates_down = (
            (x_down + 1, y_down), (x_down + 2, y_down), (x_down + 2, y_down - 1), (x_down + 3, y_down - 1),
            (x_down + 4, y_down - 1),
            (x_down + 5, y_down - 1),
            (x_down + 6, y_down - 1), (x_down + 6, y_down - 2), (x_down + 7, y_down - 2), (x_down + 8, y_down - 2),
            (x_down + 9, y_down - 2),
            (x_down + 10, y_down - 2), (x_down + 10, y_down - 3), (x_down + 11, y_down - 3))
        x_down = x_down + 12
        y_down = y_down - 3
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 4:
        kernel_coordinates_down = (
            (x_down + 1, y_down), (x_down + 2, y_down - 1), (x_down + 3, y_down - 1), (x_down + 4, y_down - 1),
            (x_down + 5, y_down - 2))
        x_down = x_down + 6
        y_down = y_down - 2
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 5:
        kernel_coordinates_down = (
            (x_down + 1, y_down), (x_down + 1, y_down - 1), (x_down + 2, y_down - 1), (x_down + 3, y_down - 1),
            (x_down + 4, y_down - 2),
            (x_down + 5, y_down - 2), (x_down + 6, y_down - 2), (x_down + 6, y_down - 3), (x_down + 7, y_down - 3),
            (x_down + 8, y_down - 3),
            (x_down + 9, y_down - 4), (x_down + 10, y_down - 4), (x_down + 11, y_down - 4), (x_down + 11, y_down - 5)
        )
        x_down = x_down + 12
        y_down = y_down - 5
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 6:
        kernel_coordinates_down = (
            (x_down + 1, y_down), (x_down + 1, y_down - 1), (x_down + 2, y_down - 1), (x_down + 3, y_down - 1),
            (x_down + 3, y_down - 2),
            (x_down + 4, y_down - 2), (x_down + 5, y_down - 2), (x_down + 5, y_down - 3))
        x_down = x_down + 6
        y_down = y_down - 3
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 7:
        kernel_coordinates_down = (
            (x_down + 1, y_down), (x_down + 1, y_down - 1), (x_down + 2, y_down - 1), (x_down + 3, y_down - 2),
            (x_down + 4, y_down - 2), (x_down + 4, y_down - 3), (x_down + 5, y_down - 3), (x_down + 6, y_down - 3),
            (x_down + 6, y_down - 4), (x_down + 7, y_down - 4), (x_down + 8, y_down - 4), (x_down + 8, y_down - 5),
            (x_down + 9, y_down - 5), (x_down + 10, y_down - 6), (x_down + 11, y_down - 6), (x_down + 11, y_down - 7)
        )
        x_down = x_down + 12
        y_down = y_down - 7
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 8:
        kernel_coordinates_down = (
            (x_down + 1, y_down), (x_down + 1, y_down - 1), (x_down + 2, y_down - 1), (x_down + 2, y_down - 2),
            (x_down + 3, y_down - 2),
            (x_down + 4, y_down - 2), (x_down + 4, y_down - 3), (x_down + 5, y_down - 3), (x_down + 5, y_down - 4))
        x_down = x_down + 6
        y_down = y_down - 4
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 9:
        kernel_coordinates_down = (
            (x_down + 1, y_down), (x_down + 1, y_down - 1), (x_down + 2, y_down - 1), (x_down + 2, y_down - 2),
            (x_down + 3, y_down - 2), (x_down + 3, y_down - 3), (x_down + 4, y_down - 3),
            (x_down + 5, y_down - 4), (x_down + 6, y_down - 4), (x_down + 6, y_down - 5), (x_down + 7, y_down - 5),
            (x_down + 7, y_down - 6), (x_down + 8, y_down - 6), (x_down + 9, y_down - 7), (x_down + 10, y_down - 7),
            (x_down + 10, y_down - 8), (x_down + 11, y_down - 8), (x_down + 11, y_down - 9)
        )
        x_down = x_down + 12
        y_down = y_down - 9
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 10:
        kernel_coordinates_down = (
            (x_down + 1, y_down), (x_down + 1, y_down - 1), (x_down + 2, y_down - 1), (x_down + 2, y_down - 2),
            (x_down + 3, y_down - 2),
            (x_down + 3, y_down - 3), (x_down + 4, y_down - 3), (x_down + 4, y_down - 4), (x_down + 5, y_down - 4),
            (x_down + 5, y_down - 5))
        x_down = x_down + 6
        y_down = y_down - 5
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 11:
        kernel_coordinates_down = (
            (x_down + 1, y_down), (x_down + 1, y_down - 1), (x_down + 2, y_down - 1), (x_down + 2, y_down - 2),
            (x_down + 3, y_down - 3),
            (x_down + 4, y_down - 3), (x_down + 4, y_down - 4), (x_down + 5, y_down - 4), (x_down + 5, y_down - 5),
            (x_down + 6, y_down - 5), (x_down + 6, y_down - 6), (x_down + 7, y_down - 6), (x_down + 7, y_down - 7),
            (x_down + 8, y_down - 7), (x_down + 8, y_down - 8), (x_down + 9, y_down - 9), (x_down + 10, y_down - 9),
            (x_down + 10, y_down - 10), (x_down + 11, y_down - 10), (x_down + 11, y_down - 11))
        x_down = x_down + 12
        y_down = y_down - 11
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 12:
        kernel_coordinates_down = (
            (x_down + 1, y_down - 1), (x_down + 2, y_down - 2), (x_down + 3, y_down - 3), (x_down + 4, y_down - 4),
            (x_down + 5, y_down - 5))
        x_down = x_down + 6
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 13:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down + 1, y_down - 1), (x_down + 1, y_down - 2), (x_down + 2, y_down - 2),
            (x_down + 3, y_down - 3),
            (x_down + 3, y_down - 4), (x_down + 4, y_down - 4), (x_down + 4, y_down - 5), (x_down + 5, y_down - 5),
            (x_down + 5, y_down - 6), (x_down + 6, y_down - 6), (x_down + 6, y_down - 7), (x_down + 7, y_down - 7),
            (x_down + 7, y_down - 8), (x_down + 8, y_down - 8), (x_down + 9, y_down - 9), (x_down + 9, y_down - 10),
            (x_down + 10, y_down - 10), (x_down + 10, y_down - 11), (x_down + 11, y_down - 11))
        x_down = x_down + 11
        y_down = y_down - 12
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 14:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down + 1, y_down - 1), (x_down + 1, y_down - 2), (x_down + 2, y_down - 2),
            (x_down + 2, y_down - 3), (x_down + 3, y_down - 3), (x_down + 3, y_down - 4), (x_down + 4, y_down - 4),
            (x_down + 4, y_down - 5), (x_down + 5, y_down - 5))
        x_down = x_down + 5
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 15:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down + 1, y_down - 1), (x_down + 1, y_down - 2), (x_down + 2, y_down - 2),
            (x_down + 2, y_down - 3), (x_down + 3, y_down - 3), (x_down + 3, y_down - 4),
            (x_down + 4, y_down - 5), (x_down + 4, y_down - 6), (x_down + 5, y_down - 6), (x_down + 5, y_down - 7),
            (x_down + 6, y_down - 7), (x_down + 6, y_down - 8), (x_down + 7, y_down - 9), (x_down + 7, y_down - 10),
            (x_down + 8, y_down - 10), (x_down + 8, y_down - 11), (x_down + 9, y_down - 11)
        )
        x_down = x_down + 9
        y_down = y_down - 12
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 16:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down + 1, y_down - 1), (x_down + 1, y_down - 2), (x_down + 2, y_down - 2),
            (x_down + 2, y_down - 3), (x_down + 2, y_down - 4), (x_down + 3, y_down - 4), (x_down + 3, y_down - 5),
            (x_down + 4, y_down - 5))
        x_down = x_down + 4
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 17:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down + 1, y_down - 1), (x_down + 1, y_down - 2), (x_down + 2, y_down - 3),
            (x_down + 2, y_down - 4), (x_down + 3, y_down - 4), (x_down + 3, y_down - 5), (x_down + 3, y_down - 6),
            (x_down + 4, y_down - 6), (x_down + 4, y_down - 7), (x_down + 4, y_down - 8), (x_down + 5, y_down - 8),
            (x_down + 5, y_down - 9), (x_down + 6, y_down - 10), (x_down + 6, y_down - 11), (x_down + 7, y_down - 11)
        )
        x_down = x_down + 7
        y_down = y_down - 12
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 18:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down + 1, y_down - 1), (x_down + 1, y_down - 2), (x_down + 1, y_down - 3),
            (x_down + 2, y_down - 3), (x_down + 2, y_down - 4), (x_down + 2, y_down - 5), (x_down + 3, y_down - 5))
        x_down = x_down + 3
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 19:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down + 1, y_down - 1), (x_down + 1, y_down - 2), (x_down + 1, y_down - 3),
            (x_down + 2, y_down - 4),
            (x_down + 2, y_down - 5), (x_down + 2, y_down - 6), (x_down + 3, y_down - 6), (x_down + 3, y_down - 7),
            (x_down + 3, y_down - 8),
            (x_down + 4, y_down - 9), (x_down + 4, y_down - 10), (x_down + 4, y_down - 11), (x_down + 5, y_down - 11)
        )
        x_down = x_down + 5
        y_down = y_down - 12
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 20:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down + 1, y_down - 2), (x_down + 1, y_down - 3), (x_down + 1, y_down - 4),
            (x_down + 2, y_down - 5))
        x_down = x_down + 2
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 21:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down, y_down - 2), (x_down + 1, y_down - 2), (x_down + 1, y_down - 3),
            (x_down + 1, y_down - 4),
            (x_down + 1, y_down - 5),
            (x_down + 1, y_down - 6), (x_down + 2, y_down - 6), (x_down + 2, y_down - 7), (x_down + 2, y_down - 8),
            (x_down + 2, y_down - 9),
            (x_down + 2, y_down - 10), (x_down + 3, y_down - 10), (x_down + 3, y_down - 11))
        x_down = x_down + 3
        y_down = y_down - 12
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 22:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down, y_down - 2), (x_down, y_down - 3), (x_down + 1, y_down - 3),
            (x_down + 1, y_down - 4),
            (x_down + 1, y_down - 5))
        x_down = x_down + 1
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 23:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down, y_down - 2), (x_down, y_down - 3), (x_down, y_down - 4),
            (x_down, y_down - 5), (x_down, y_down - 6),
            (x_down + 1, y_down - 6), (x_down + 1, y_down - 7), (x_down + 1, y_down - 8), (x_down + 1, y_down - 9),
            (x_down + 1, y_down - 10), (x_down + 1, y_down - 11))
        x_down = x_down + 1
        y_down = y_down - 12
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 24:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down, y_down - 2), (x_down, y_down - 3), (x_down, y_down - 4),
            (x_down, y_down - 5))
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 25:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down, y_down - 2), (x_down, y_down - 3), (x_down, y_down - 4),
            (x_down, y_down - 5), (x_down, y_down - 6),
            (x_down - 1, y_down - 6), (x_down - 1, y_down - 7), (x_down - 1, y_down - 8), (x_down - 1, y_down - 9),
            (x_down - 1, y_down - 10), (x_down - 1, y_down - 11))
        x_down = x_down - 1
        y_down = y_down - 12
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 26:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down, y_down - 2), (x_down, y_down - 3), (x_down - 1, y_down - 3),
            (x_down - 1, y_down - 4),
            (x_down - 1, y_down - 5))
        x_down = x_down - 1
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 27:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down, y_down - 2), (x_down - 1, y_down - 2), (x_down - 1, y_down - 3),
            (x_down - 1, y_down - 4),
            (x_down - 1, y_down - 5),
            (x_down - 1, y_down - 6), (x_down - 2, y_down - 6), (x_down - 2, y_down - 7), (x_down - 2, y_down - 8),
            (x_down - 2, y_down - 9),
            (x_down - 2, y_down - 10), (x_down - 3, y_down - 10), (x_down - 3, y_down - 11))
        x_down = x_down - 3
        y_down = y_down - 12
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 28:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down - 1, y_down - 2), (x_down - 1, y_down - 3), (x_down - 1, y_down - 4),
            (x_down - 2, y_down - 5))
        x_down = x_down - 2
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 29:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down - 1, y_down - 1), (x_down - 1, y_down - 2), (x_down - 1, y_down - 3),
            (x_down - 2, y_down - 4),
            (x_down - 2, y_down - 5), (x_down - 2, y_down - 6), (x_down - 3, y_down - 6), (x_down - 3, y_down - 7),
            (x_down - 3, y_down - 8),
            (x_down - 4, y_down - 9), (x_down - 4, y_down - 10), (x_down - 4, y_down - 11), (x_down - 5, y_down - 11)
        )
        x_down = x_down - 5
        y_down = y_down - 12
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 30:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down - 1, y_down - 1), (x_down - 1, y_down - 2), (x_down - 1, y_down - 3),
            (x_down - 2, y_down - 3), (x_down - 2, y_down - 4), (x_down - 2, y_down - 5), (x_down - 3, y_down - 5))
        x_down = x_down - 3
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 31:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down - 1, y_down - 1), (x_down - 1, y_down - 2), (x_down - 2, y_down - 3),
            (x_down - 2, y_down - 4), (x_down - 3, y_down - 4), (x_down - 3, y_down - 5), (x_down - 3, y_down - 6),
            (x_down - 4, y_down - 6), (x_down - 4, y_down - 7), (x_down - 4, y_down - 8), (x_down - 5, y_down - 8),
            (x_down - 5, y_down - 9), (x_down - 6, y_down - 10), (x_down - 6, y_down - 11), (x_down - 7, y_down - 11)
        )
        x_down = x_down - 7
        y_down = y_down - 12
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 32:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down - 1, y_down - 1), (x_down - 1, y_down - 2), (x_down - 2, y_down - 2),
            (x_down - 2, y_down - 3), (x_down - 2, y_down - 4), (x_down - 3, y_down - 4), (x_down - 3, y_down - 5),
            (x_down - 4, y_down - 5))
        x_down = x_down - 4
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 33:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down - 1, y_down - 1), (x_down - 1, y_down - 2), (x_down - 2, y_down - 2),
            (x_down - 2, y_down - 3), (x_down - 3, y_down - 3), (x_down - 3, y_down - 4),
            (x_down - 4, y_down - 5), (x_down - 4, y_down - 6), (x_down - 5, y_down - 6), (x_down - 5, y_down - 7),
            (x_down - 6, y_down - 7), (x_down - 6, y_down - 8), (x_down - 7, y_down - 9), (x_down - 7, y_down - 10),
            (x_down - 8, y_down - 10), (x_down - 8, y_down - 11), (x_down - 9, y_down - 11)
        )
        x_down = x_down - 9
        y_down = y_down - 12
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 34:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down - 1, y_down - 1), (x_down - 1, y_down - 2), (x_down - 2, y_down - 2),
            (x_down - 2, y_down - 3), (x_down - 3, y_down - 3), (x_down - 3, y_down - 4), (x_down - 4, y_down - 4),
            (x_down - 4, y_down - 5), (x_down - 5, y_down - 5))
        x_down = x_down - 5
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 35:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down - 1, y_down - 1), (x_down - 1, y_down - 2), (x_down - 2, y_down - 2),
            (x_down - 3, y_down - 3),
            (x_down - 3, y_down - 4), (x_down - 4, y_down - 4), (x_down - 4, y_down - 5), (x_down - 5, y_down - 5),
            (x_down - 5, y_down - 6), (x_down - 6, y_down - 6), (x_down - 6, y_down - 7), (x_down - 7, y_down - 7),
            (x_down - 7, y_down - 8), (x_down - 8, y_down - 8), (x_down - 9, y_down - 9), (x_down - 9, y_down - 10),
            (x_down - 10, y_down - 10), (x_down - 10, y_down - 11), (x_down - 11, y_down - 11))
        x_down = x_down - 11
        y_down = y_down - 12
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 36:
        kernel_coordinates_down = (
            (x_down - 1, y_down - 1), (x_down - 2, y_down - 2), (x_down - 3, y_down - 3), (x_down - 4, y_down - 4),
            (x_down - 5, y_down - 5))
        x_down = x_down - 6
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 37:
        kernel_coordinates_down = (
            (x_down - 1, y_down), (x_down - 1, y_down - 1), (x_down - 2, y_down - 1), (x_down - 2, y_down - 2),
            (x_down - 3, y_down - 3),
            (x_down - 4, y_down - 3), (x_down - 4, y_down - 4), (x_down - 5, y_down - 4), (x_down - 5, y_down - 5),
            (x_down - 6, y_down - 5), (x_down - 6, y_down - 6), (x_down - 7, y_down - 6), (x_down - 7, y_down - 7),
            (x_down - 8, y_down - 7), (x_down - 8, y_down - 8), (x_down - 9, y_down - 9), (x_down - 10, y_down - 9),
            (x_down - 10, y_down - 10), (x_down - 11, y_down - 10), (x_down - 11, y_down - 11))
        x_down = x_down - 12
        y_down = y_down - 11
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 38:
        kernel_coordinates_down = (
            (x_down - 1, y_down), (x_down - 1, y_down - 1), (x_down - 2, y_down - 1), (x_down - 2, y_down - 2),
            (x_down - 3, y_down - 2),
            (x_down - 3, y_down - 3), (x_down - 4, y_down - 3), (x_down - 4, y_down - 4), (x_down - 5, y_down - 4),
            (x_down - 5, y_down - 5))
        x_down = x_down - 6
        y_down = y_down - 5
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 39:
        kernel_coordinates_down = (
            (x_down - 1, y_down), (x_down - 1, y_down - 1), (x_down - 2, y_down - 1), (x_down - 2, y_down - 2),
            (x_down - 3, y_down - 2), (x_down - 3, y_down - 3), (x_down - 4, y_down - 3),
            (x_down - 5, y_down - 4), (x_down - 6, y_down - 4), (x_down - 6, y_down - 5), (x_down - 7, y_down - 5),
            (x_down - 7, y_down - 6), (x_down - 8, y_down - 6), (x_down - 9, y_down - 7), (x_down - 10, y_down - 7),
            (x_down - 10, y_down - 8), (x_down - 11, y_down - 8), (x_down - 11, y_down - 9)
        )
        x_down = x_down - 12
        y_down = y_down - 9
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 40:
        kernel_coordinates_down = (
            (x_down - 1, y_down), (x_down - 1, y_down - 1), (x_down - 2, y_down - 1), (x_down - 2, y_down - 2),
            (x_down - 3, y_down - 2),
            (x_down - 4, y_down - 2), (x_down - 4, y_down - 3), (x_down - 5, y_down - 3), (x_down - 5, y_down - 4))
        x_down = x_down - 6
        y_down = y_down - 4
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 41:
        kernel_coordinates_down = (
            (x_down - 1, y_down), (x_down - 1, y_down - 1), (x_down - 2, y_down - 1), (x_down - 3, y_down - 2),
            (x_down - 4, y_down - 2), (x_down - 4, y_down - 3), (x_down - 5, y_down - 3), (x_down - 6, y_down - 3),
            (x_down - 6, y_down - 4), (x_down - 7, y_down - 4), (x_down - 8, y_down - 4), (x_down - 8, y_down - 5),
            (x_down - 9, y_down - 5), (x_down - 10, y_down - 6), (x_down - 11, y_down - 6), (x_down - 11, y_down - 7)
        )
        x_down = x_down - 12
        y_down = y_down - 7
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 42:
        kernel_coordinates_down = (
            (x_down - 1, y_down), (x_down - 1, y_down - 1), (x_down - 2, y_down - 1), (x_down - 3, y_down - 1),
            (x_down - 3, y_down - 2),
            (x_down - 4, y_down - 2), (x_down - 5, y_down - 2), (x_down - 5, y_down - 3))
        x_down = x_down - 6
        y_down = y_down - 3
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 43:
        kernel_coordinates_down = (
            (x_down - 1, y_down), (x_down - 1, y_down - 1), (x_down - 2, y_down - 1), (x_down - 3, y_down - 1),
            (x_down - 4, y_down - 2),
            (x_down - 5, y_down - 2), (x_down - 6, y_down - 2), (x_down - 6, y_down - 3), (x_down - 7, y_down - 3),
            (x_down - 8, y_down - 3),
            (x_down - 9, y_down - 4), (x_down - 10, y_down - 4), (x_down - 11, y_down - 4), (x_down - 11, y_down - 5)
        )
        x_down = x_down - 12
        y_down = y_down - 5
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 44:
        kernel_coordinates_down = (
            (x_down - 1, y_down), (x_down - 2, y_down - 1), (x_down - 3, y_down - 1), (x_down - 4, y_down - 1),
            (x_down - 5, y_down - 2))
        x_down = x_down - 6
        y_down = y_down - 2
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 45:
        kernel_coordinates_down = (
            (x_down - 1, y_down), (x_down - 2, y_down), (x_down - 2, y_down - 1), (x_down - 3, y_down - 1),
            (x_down - 4, y_down - 1),
            (x_down - 5, y_down - 1),
            (x_down - 6, y_down - 1), (x_down - 6, y_down - 2), (x_down - 7, y_down - 2), (x_down - 8, y_down - 2),
            (x_down - 9, y_down - 2),
            (x_down - 10, y_down - 2), (x_down - 10, y_down - 3), (x_down - 11, y_down - 3))
        x_down = x_down - 12
        y_down = y_down - 3
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 46:
        kernel_coordinates_down = (
            (x_down - 1, y_down), (x_down - 2, y_down), (x_down - 3, y_down), (x_down - 3, y_down - 1),
            (x_down - 4, y_down - 1),
            (x_down - 5, y_down - 1))
        x_down = x_down - 6
        y_down = y_down - 1
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 47:
        kernel_coordinates_down = (
            (x_down - 1, y_down), (x_down - 2, y_down), (x_down - 3, y_down), (x_down - 4, y_down),
            (x_down - 5, y_down), (x_down - 6, y_down),
            (x_down - 6, y_down - 1), (x_down - 7, y_down - 1), (x_down - 8, y_down - 1), (x_down - 9, y_down - 1),
            (x_down - 10, y_down - 1), (x_down - 11, y_down - 1))
        x_down = x_down - 12
        y_down = y_down - 1
        return kernel_coordinates_down, x_down, y_down


def calculate_kernel_coordinates_with_angle_positive(angle, half_size):
    step_H, step_W = get_steps(angle)
    kernel_coordinates = []
    for i in range(1, half_size + 1):
        kernel_coordinates.append(
            (half_size + int(round(divide(i, step_H))), half_size - int(round(divide(i, step_W)))))
        kernel_coordinates.append(
            (half_size - int(round(divide(i, step_H))), half_size + int(round(divide(i, step_W)))))
    return tuple(kernel_coordinates)


def calculate_kernel_coordinates_with_angle_negative(angle, half_size):
    step_H, step_W, starting_point_H, starting_point_W = get_steps_with_starting_point(angle, half_size)
    kernel_coordinates = []
    for i in range(1, half_size + 1):
        kernel_coordinates.append(
            (starting_point_H + int(round(divide(i, step_H))), starting_point_W - int(round(divide(i, step_W)))))
        kernel_coordinates.append(
            (starting_point_H - int(round(divide(i, step_H))), starting_point_W + int(round(divide(i, step_W)))))
    return tuple(kernel_coordinates)


def get_steps(angle):
    if angle <= LINE_45:
        step_H = 1
        step_W = divide(LINE_45, angle)
    elif angle < LINE_90:
        step_H = divide(LINE_90 - LINE_45, LINE_90 - angle)
        step_W = 1
    elif angle == LINE_90:
        step_H = 0
        step_W = 1
    elif angle <= LINE_135:
        step_H = - divide(LINE_135 - LINE_90, angle - LINE_90)
        step_W = 1
    else:
        step_H = - 1
        step_W = divide(LINE_180 - LINE_135, LINE_180 - angle)
    return step_H, step_W

def get_steps_with_starting_point(angle, half_size):
    if angle <= LINE_45:
        step_H = 1
        step_W = divide(LINE_45, angle)
        starting_point_H = half_size
        starting_point_W = half_size + 1
    elif angle < LINE_90:
        step_H = divide(LINE_90 - LINE_45, LINE_90 - angle)
        step_W = 1
        starting_point_H = half_size + 1
        starting_point_W = half_size
    elif angle == LINE_90:
        step_H = 0
        step_W = 1
        starting_point_H = half_size + 1
        starting_point_W = half_size
    elif angle <= LINE_135:
        step_H = - divide(LINE_135 - LINE_90, angle - LINE_90)
        step_W = 1
        starting_point_H = half_size - 1
        starting_point_W = half_size
    else:
        step_H = - 1
        step_W = divide(LINE_180 - LINE_135, LINE_180 - angle)
        starting_point_H = half_size
        starting_point_W = half_size - 1
    return step_H, step_W, starting_point_H, starting_point_W


def divide(divided, divider):
    if divider == 0:
        result = 0
    else:
        result = divided / divider
    return result
