ANGLES = [0, 7.5, 15, 22.5, 30, 37.5, 45, 52.5, 60, 67.5, 75, 82.5, 90, 97.5, 105, 112.5, 120, 127.5, 135, 142.5, 150,
          157.5, 165, 172.5]


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
        kernel_coordinates_second_edge, x_second_edge, y_second_edge = calculate_kernel_coordinates_up(second_edge, x_second_edge,
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
        kernel_coordinates_second_edge, x_second_edge, y_second_edge = calculate_kernel_coordinates_down(second_edge, x_second_edge,
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
        kernel_coordinates_second_edge, x_second_edge, y_second_edge = calculate_kernel_coordinates_down(second_edge, x_second_edge,
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
        kernel_coordinates_second_edge, x_second_edge, y_second_edge = calculate_kernel_coordinates_up(second_edge, x_second_edge,
                                                                                     y_second_edge)
        continue_draw = draw(kernel, kernel_coordinates_second_edge, half_size, size_counter, continue_draw)
    return ANGLES[second_edge], (ANGLES[first_edge] + 180) - ANGLES[second_edge], kernel

def draw(kernel, kernel_coordinates, half_size, size_counter, continue_draw):
    for x_coor, y_coor in kernel_coordinates:
        # size_counter = size_counter + 1
        kernel[x_coor, y_coor] = 1
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
            (x_up - 1, y_up), (x_up - 2, y_up), (x_up - 3, y_up), (x_up - 3, y_up + 1),
            (x_up - 4, y_up + 1),
            (x_up - 5, y_up + 1))
        x_up = x_up - 6
        y_up = y_up + 1
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 2:
        kernel_coordinates_up = (
            (x_up - 1, y_up), (x_up - 2, y_up + 1), (x_up - 3, y_up + 1), (x_up - 4, y_up + 1),
            (x_up - 5, y_up + 2))
        x_up = x_up - 6
        y_up = y_up + 2
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 3:
        kernel_coordinates_up = (
            (x_up - 1, y_up), (x_up - 1, y_up + 1), (x_up - 2, y_up + 1), (x_up - 3, y_up + 1),
            (x_up - 3, y_up + 2),
            (x_up - 4, y_up + 2), (x_up - 5, y_up + 2), (x_up - 5, y_up + 3))
        x_up = x_up - 6
        y_up = y_up + 3
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 4:
        kernel_coordinates_up = (
            (x_up - 1, y_up), (x_up - 1, y_up + 1), (x_up - 2, y_up + 1), (x_up - 2, y_up + 2),
            (x_up - 3, y_up + 2),
            (x_up - 4, y_up + 2), (x_up - 4, y_up + 3), (x_up - 5, y_up + 3), (x_up - 5, y_up + 4))
        x_up = x_up - 6
        y_up = y_up + 4
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 5:
        kernel_coordinates_up = (
            (x_up - 1, y_up), (x_up - 1, y_up + 1), (x_up - 2, y_up + 1), (x_up - 2, y_up + 2),
            (x_up - 3, y_up + 2),
            (x_up - 3, y_up + 3), (x_up - 4, y_up + 3), (x_up - 4, y_up + 4), (x_up - 5, y_up + 4),
            (x_up - 5, y_up + 5))
        x_up = x_up - 6
        y_up = y_up + 5
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 6:
        kernel_coordinates_up = (
            (x_up - 1, y_up + 1), (x_up - 2, y_up + 2), (x_up - 3, y_up + 3), (x_up - 4, y_up + 4),
            (x_up - 5, y_up + 5))
        x_up = x_up - 6
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 7:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up - 1, y_up + 1), (x_up - 1, y_up + 2), (x_up - 2, y_up + 2),
            (x_up - 2, y_up + 3), (x_up - 3, y_up + 3), (x_up - 3, y_up + 4), (x_up - 4, y_up + 4),
            (x_up - 4, y_up + 5), (x_up - 5, y_up + 5))
        x_up = x_up - 5
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 8:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up - 1, y_up + 1), (x_up - 1, y_up + 2), (x_up - 2, y_up + 2),
            (x_up - 2, y_up + 3), (x_up - 2, y_up + 4), (x_up - 3, y_up + 4), (x_up - 3, y_up + 5),
            (x_up - 4, y_up + 5))
        x_up = x_up - 4
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 9:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up - 1, y_up + 1), (x_up - 1, y_up + 2), (x_up - 1, y_up + 3),
            (x_up - 2, y_up + 3), (x_up - 2, y_up + 4), (x_up - 2, y_up + 5), (x_up - 3, y_up + 5))
        x_up = x_up - 3
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 10:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up - 1, y_up + 2), (x_up - 1, y_up + 3), (x_up - 1, y_up + 4),
            (x_up - 2, y_up + 5))
        x_up = x_up - 2
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 11:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up, y_up + 2), (x_up, y_up + 3), (x_up - 1, y_up + 3),
            (x_up - 1, y_up + 4),
            (x_up - 1, y_up + 5))
        x_up = x_up - 1
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up
    
    elif edge_number == 12:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up, y_up + 2), (x_up, y_up + 3), (x_up, y_up + 4), (x_up, y_up + 5))
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up
    
    elif edge_number == 13:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up, y_up + 2), (x_up, y_up + 3), (x_up + 1, y_up + 3), (x_up + 1, y_up + 4),
            (x_up + 1, y_up + 5))
        x_up = x_up + 1
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 14:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up + 1, y_up + 2), (x_up + 1, y_up + 3), (x_up + 1, y_up + 4),
            (x_up + 2, y_up + 5))
        x_up = x_up + 2
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 15:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up + 1, y_up + 1), (x_up + 1, y_up + 2), (x_up + 1, y_up + 3),
            (x_up + 2, y_up + 3), (x_up + 2, y_up + 4), (x_up + 2, y_up + 5), (x_up + 3, y_up + 5))
        x_up = x_up + 3
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 16:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up + 1, y_up + 1), (x_up + 1, y_up + 2), (x_up + 2, y_up + 2),
            (x_up + 2, y_up + 3), (x_up + 2, y_up + 4), (x_up + 3, y_up + 4), (x_up + 3, y_up + 5),
            (x_up + 4, y_up + 5))
        x_up = x_up + 4
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 17:
        kernel_coordinates_up = (
            (x_up, y_up + 1), (x_up + 1, y_up + 1), (x_up + 1, y_up + 2), (x_up + 2, y_up + 2),
            (x_up + 2, y_up + 3), (x_up + 3, y_up + 3), (x_up + 3, y_up + 4), (x_up + 4, y_up + 4),
            (x_up + 4, y_up + 5), (x_up + 5, y_up + 5))
        x_up = x_up + 5
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 18:
        kernel_coordinates_up = (
            (x_up + 1, y_up + 1), (x_up + 2, y_up + 2), (x_up + 3, y_up + 3), (x_up + 4, y_up + 4),
            (x_up + 5, y_up + 5))
        x_up = x_up + 6
        y_up = y_up + 6
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 19:
        kernel_coordinates_up = (
            (x_up + 1, y_up), (x_up + 1, y_up + 1), (x_up + 2, y_up + 1), (x_up + 2, y_up + 2), (x_up + 3, y_up + 2),
            (x_up + 3, y_up + 3), (x_up + 4, y_up + 3), (x_up + 4, y_up + 4), (x_up + 5, y_up + 4),
            (x_up + 5, y_up + 5))
        x_up = x_up + 6
        y_up = y_up + 5
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 20:
        kernel_coordinates_up = (
            (x_up + 1, y_up), (x_up + 1, y_up + 1), (x_up + 2, y_up + 1), (x_up + 2, y_up + 2), (x_up + 3, y_up + 2),
            (x_up + 4, y_up + 2), (x_up + 4, y_up + 3), (x_up + 5, y_up + 3), (x_up + 5, y_up + 4))
        x_up = x_up + 6
        y_up = y_up + 4
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 21:
        kernel_coordinates_up = (
            (x_up + 1, y_up), (x_up + 1, y_up + 1), (x_up + 2, y_up + 1), (x_up + 3, y_up + 1), (x_up + 3, y_up + 2),
            (x_up + 4, y_up + 2), (x_up + 5, y_up + 2), (x_up + 5, y_up + 3))
        x_up = x_up + 6
        y_up = y_up + 3
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 22:
        kernel_coordinates_up = (
            (x_up + 1, y_up), (x_up + 2, y_up + 1), (x_up + 3, y_up + 1), (x_up + 4, y_up + 1), (x_up + 5, y_up + 2))
        x_up = x_up + 6
        y_up = y_up + 2
        return kernel_coordinates_up, x_up, y_up

    elif edge_number == 23:
        kernel_coordinates_up = (
            (x_up + 1, y_up), (x_up + 2, y_up), (x_up + 3, y_up), (x_up + 3, y_up + 1), (x_up + 4, y_up + 1),
            (x_up + 5, y_up + 1))
        x_up = x_up + 6
        y_up = y_up + 1
        return kernel_coordinates_up, x_up, y_up


def calculate_kernel_coordinates_down(edge_number, x_down, y_down):
    if edge_number == 0:
        kernel_coordinates_down = (
            (x_down + 1, y_down), (x_down + 2, y_down), (x_down + 3, y_down), (x_down + 4, y_down), (x_down + 5, y_down))
        x_down = x_down + 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 1:
        kernel_coordinates_down = (
            (x_down + 1, y_down), (x_down + 2, y_down), (x_down + 3, y_down), (x_down + 3, y_down - 1), (x_down + 4, y_down - 1),
            (x_down + 5, y_down - 1))
        x_down = x_down + 6
        y_down = y_down - 1
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 2:
        kernel_coordinates_down = (
            (x_down + 1, y_down), (x_down + 2, y_down - 1), (x_down + 3, y_down - 1), (x_down + 4, y_down - 1), (x_down + 5, y_down - 2))
        x_down = x_down + 6
        y_down = y_down - 2
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 3:
        kernel_coordinates_down = (
            (x_down + 1, y_down), (x_down + 1, y_down - 1), (x_down + 2, y_down - 1), (x_down + 3, y_down - 1), (x_down + 3, y_down - 2),
            (x_down + 4, y_down - 2), (x_down + 5, y_down - 2), (x_down + 5, y_down - 3))
        x_down = x_down + 6
        y_down = y_down - 3
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 4:
        kernel_coordinates_down = (
            (x_down + 1, y_down), (x_down + 1, y_down - 1), (x_down + 2, y_down - 1), (x_down + 2, y_down - 2), (x_down + 3, y_down - 2),
            (x_down + 4, y_down - 2), (x_down + 4, y_down - 3), (x_down + 5, y_down - 3), (x_down + 5, y_down - 4))
        x_down = x_down + 6
        y_down = y_down - 4
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 5:
        kernel_coordinates_down = (
            (x_down + 1, y_down), (x_down + 1, y_down - 1), (x_down + 2, y_down - 1), (x_down + 2, y_down - 2), (x_down + 3, y_down - 2),
            (x_down + 3, y_down - 3), (x_down + 4, y_down - 3), (x_down + 4, y_down - 4), (x_down + 5, y_down - 4),
            (x_down + 5, y_down - 5))
        x_down = x_down + 6
        y_down = y_down - 5
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 6:
        kernel_coordinates_down = (
            (x_down + 1, y_down - 1), (x_down + 2, y_down - 2), (x_down + 3, y_down - 3), (x_down + 4, y_down - 4),
            (x_down + 5, y_down - 5))
        x_down = x_down + 6
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 7:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down + 1, y_down - 1), (x_down + 1, y_down - 2), (x_down + 2, y_down - 2),
            (x_down + 2, y_down - 3), (x_down + 3, y_down - 3), (x_down + 3, y_down - 4), (x_down + 4, y_down - 4),
            (x_down + 4, y_down - 5), (x_down + 5, y_down - 5))
        x_down = x_down + 5
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 8:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down + 1, y_down - 1), (x_down + 1, y_down - 2), (x_down + 2, y_down - 2),
            (x_down + 2, y_down - 3), (x_down + 2, y_down - 4), (x_down + 3, y_down - 4), (x_down + 3, y_down - 5),
            (x_down + 4, y_down - 5))
        x_down = x_down + 4
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 9:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down + 1, y_down - 1), (x_down + 1, y_down - 2), (x_down + 1, y_down - 3),
            (x_down + 2, y_down - 3), (x_down + 2, y_down - 4), (x_down + 2, y_down - 5), (x_down + 3, y_down - 5))
        x_down = x_down + 3
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 10:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down + 1, y_down - 2), (x_down + 1, y_down - 3), (x_down + 1, y_down - 4),
            (x_down + 2, y_down - 5))
        x_down = x_down + 2
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 11:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down, y_down - 2), (x_down, y_down - 3), (x_down + 1, y_down - 3), (x_down + 1, y_down - 4),
            (x_down + 1, y_down - 5))
        x_down = x_down + 1
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 12:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down, y_down - 2), (x_down, y_down - 3), (x_down, y_down - 4),
            (x_down, y_down - 5))
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 13:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down, y_down - 2), (x_down, y_down - 3), (x_down - 1, y_down - 3),
            (x_down - 1, y_down - 4), (x_down - 1, y_down - 5))
        x_down = x_down - 1
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 14:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down - 1, y_down - 2), (x_down - 1, y_down - 3), (x_down - 1, y_down - 4),
            (x_down - 2, y_down - 5))
        x_down = x_down - 2
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 15:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down - 1, y_down - 1), (x_down - 1, y_down - 2), (x_down - 1, y_down - 3),
            (x_down - 2, y_down - 3), (x_down - 2, y_down - 4), (x_down - 2, y_down - 5), (x_down - 3, y_down - 5))
        x_down = x_down - 3
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 16:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down - 1, y_down - 1), (x_down - 1, y_down - 2), (x_down - 2, y_down - 2),
            (x_down - 2, y_down - 3), (x_down - 2, y_down - 4), (x_down - 3, y_down - 4), (x_down - 3, y_down - 5),
            (x_down - 4, y_down - 5))
        x_down = x_down - 4
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 17:
        kernel_coordinates_down = (
            (x_down, y_down - 1), (x_down - 1, y_down - 1), (x_down - 1, y_down - 2), (x_down - 2, y_down - 2),
            (x_down - 2, y_down - 3), (x_down - 3, y_down - 3), (x_down - 3, y_down - 4), (x_down - 4, y_down - 4),
            (x_down - 4, y_down - 5), (x_down - 5, y_down - 5))
        x_down = x_down - 5
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 18:
        kernel_coordinates_down = (
            (x_down - 1, y_down - 1), (x_down - 2, y_down - 2), (x_down - 3, y_down - 3), (x_down - 4, y_down - 4),
            (x_down - 5, y_down - 5))
        x_down = x_down - 6
        y_down = y_down - 6
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 19:
        kernel_coordinates_down = (
            (x_down - 1, y_down), (x_down - 1, y_down - 1), (x_down - 2, y_down - 1), (x_down - 2, y_down - 2),
            (x_down - 3, y_down - 2),
            (x_down - 3, y_down - 3), (x_down - 4, y_down - 3), (x_down - 4, y_down - 4), (x_down - 5, y_down - 4),
            (x_down - 5, y_down - 5))
        x_down = x_down - 6
        y_down = y_down - 5
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 20:
        kernel_coordinates_down = (
            (x_down - 1, y_down), (x_down - 1, y_down - 1), (x_down - 2, y_down - 1), (x_down - 2, y_down - 2),
            (x_down - 3, y_down - 2),
            (x_down - 4, y_down - 2), (x_down - 4, y_down - 3), (x_down - 5, y_down - 3), (x_down - 5, y_down - 4))
        x_down = x_down - 6
        y_down = y_down - 4
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 21:
        kernel_coordinates_down = (
            (x_down - 1, y_down), (x_down - 1, y_down - 1), (x_down - 2, y_down - 1), (x_down - 3, y_down - 1),
            (x_down - 3, y_down - 2),
            (x_down - 4, y_down - 2), (x_down - 5, y_down - 2), (x_down - 5, y_down - 3))
        x_down = x_down - 6
        y_down = y_down - 3
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 22:
        kernel_coordinates_down = (
            (x_down - 1, y_down), (x_down - 2, y_down - 1), (x_down - 3, y_down - 1), (x_down - 4, y_down - 1),
            (x_down - 5, y_down - 2))
        x_down = x_down - 6
        y_down = y_down - 2
        return kernel_coordinates_down, x_down, y_down

    elif edge_number == 23:
        kernel_coordinates_down = (
            (x_down - 1, y_down), (x_down - 2, y_down), (x_down - 3, y_down), (x_down - 3, y_down - 1),
            (x_down - 4, y_down - 1),
            (x_down - 5, y_down - 1))
        x_down = x_down - 6
        y_down = y_down - 1
        return kernel_coordinates_down, x_down, y_down

