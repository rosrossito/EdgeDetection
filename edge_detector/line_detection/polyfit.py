import cv2
import matplotlib.pyplot as plt
import numpy as np

THRESHOLD_LINE = 4


def find_line(output, img):
    x_arr = []
    y_arr = []
    (iH, iW) = output.shape[:2]

    # find first pixel > 0 in column
    for x in np.arange(0, iW):
        for y in np.arange(0, iH):
            if output[y, x] > 0:
                y_arr.append(y * -1)
                x_arr.append(x)
                break

    plt.scatter(x_arr, y_arr)
    plt.show()

    filterred_y_arr = reject_outliers(y_arr)
    filterred_x_arr = range(len(filterred_y_arr))

    plt.scatter(filterred_x_arr, filterred_y_arr)
    plt.show()

    model = np.polyfit(filterred_x_arr, filterred_y_arr, 1)
    predict = np.poly1d(model)

    x_lin_reg = range(0, len(filterred_y_arr))
    y_lin_reg = predict(x_lin_reg)
    plt.scatter(filterred_x_arr, filterred_y_arr)
    plt.plot(x_lin_reg, y_lin_reg, c='r')
    plt.show()

    # plot the line on image
    y_start = - int(predict(0))
    y_end = - int(predict(iW))
    cv2.line(img, (0, y_start), (iW, y_end), (255, 0, 0), 2)
    cv2.imshow("Upper bound", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def reject_outliers(data):
    m = 3
    u = np.mean(data)
    s = np.std(data)
    filtered = [e for e in data if (u - m * s < e < u + m * s)]
    return filtered


def get_neighbours(y, x):
    return [(y - 1, x - 1), (y - 1, x), (y - 1, x + 1), (y, x + 1), (y + 1, x + 1), (y + 1, x), (y + 1, x - 1),
            (y, x - 1)]


def check_lines_appropraite(lines):
    lines_to_remove = []
    for key in lines:
        if len(lines[key]) < THRESHOLD_LINE:
            lines_to_remove.append(key)
    for key in lines_to_remove:
        lines.pop(key)


def is_rect_divided(rect):
    lines = {}
    counter = 1

    while rect:
        line = []
        current_y = rect[0][0]
        current_x = rect[0][1]
        rect.remove((current_y, current_x))
        line.append((current_y, current_x))
        check_neighbours(current_y, current_x, line, rect)
        lines[counter] = line
        counter = counter + 1
    check_lines_appropraite(lines)
    return len(lines) > 1, lines


def divide_into_two_side(rect):
    points_to_remove = []
    rect_y = [coord[0] for coord in rect]

    if (rect_y.count(min(rect_y))) > THRESHOLD_LINE:
        for point in rect:
            if (point[0] == min(rect_y)):
                points_to_remove.append(point)
    if (rect_y.count(max(rect_y))) > THRESHOLD_LINE:
        for point in rect:
            if (point[0] == max(rect_y)):
                points_to_remove.append(point)

    for point in points_to_remove:
        rect.remove(point)

    finish, lines = is_rect_divided(rect)

    # for now assume that we will have two lines but in the future need too check
    if finish:
        return lines
    else:
        divide_into_two_side(lines[1])


def separate_rect(y_x_arr):
    rects = {}
    counter = 1

    while y_x_arr:
        rect = []
        current_y = y_x_arr[0][0]
        current_x = y_x_arr[0][1]
        y_x_arr.remove((current_y, current_x))
        rect.append((current_y, current_x))
        check_neighbours(current_y, current_x, rect, y_x_arr)
        rects[counter] = rect
        counter = counter + 1
    return rects


def check_neighbours(current_y, current_x, rect, y_x_arr):
    neighbours = get_neighbours(current_y, current_x)
    for neighbour in neighbours:
        if neighbour in y_x_arr:
            y_x_arr.remove(neighbour)
            rect.append(neighbour)
            check_neighbours(neighbour[0], neighbour[1], rect, y_x_arr)


def find_rect(output, img):
    y_x_arr = []
    x_arr = []
    y_arr = []
    (iH, iW) = output.shape[:2]

    # find px change from up to down
    for x in np.arange(0, iW):
        for y in np.arange(1, iH):
            if (output[y, x] - output[y - 1, x]) != 0:
                y_arr.append(y * -1)
                x_arr.append(x)
                y_x_arr.append((y, x))

                # find px change from right to left
    for y in np.arange(0, iH):
        for x in np.arange(1, iW):
            if (output[y, x] - output[y, x - 1]) != 0:
                y_arr.append(y * -1)
                x_arr.append(x)
                y_x_arr.append((y, x))

    plt.scatter(x_arr, y_arr)
    plt.show()

    rects = separate_rect(y_x_arr)

    # filterred_y_arr = reject_outliers(y_arr)
    # filterred_x_arr = range(len(filterred_y_arr))

    # plt.scatter(filterred_x_arr, filterred_y_arr)
    # plt.show()

    for key in rects:
        sides = divide_into_two_side(rects[key])
        for side in sides:
            y = [coord[0] for coord in sides[side]]
            x = [coord[1] for coord in sides[side]]
            model = np.polyfit(y, x, 1)
            predict = np.poly1d(model)

            y_lin_reg = predict(x)
            plt.scatter(x, y)
            plt.plot(x, y_lin_reg, c='r')
            plt.show()

    # plot the lines on image
    y_start = - int(predict(0))
    y_end = - int(predict(iW))
    cv2.line(img, (0, y_start), (iW, y_end), (255, 0, 0), 2)
    cv2.imshow("Upper bound", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
