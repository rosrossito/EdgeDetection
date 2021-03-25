import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats

rho = 1
# 1 degree
theta = (np.pi/180) * 1
threshold = 15
min_line_length = 20
max_line_gap = 10

def h_transform(img, original_images):
    x_arr = []
    y_arr = []
    (iH, iW) = img.shape[:2]
    output = np.zeros((iH, iW), dtype="uint8")

    # find first pixel > 0 in column
    for x in np.arange(0, iW):
        for y in np.arange(0, iH):
            if img[y, x] > 0:
                y_arr.append(y)
                x_arr.append(x)
                output[y, x] = 255
                break

    cv2.imshow('line', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_arr, y_arr)

    # hough_lines_per_image = list(map(lambda img: hough_transform(img, rho, theta, threshold, min_line_length, max_line_gap),
    #                                  output))

    # output.convertTo(output, cv2.CV_32FC1, 1.0 / 255.0)
    # edges = cv2.Canny(output, 50, 150, apertureSize=3)


    # hough_lines_per_image = hough_transform(output, rho, theta, threshold, min_line_length, max_line_gap)
    # img_with_lines = list(map(lambda img, lines: draw_lines(img, lines), original_images, hough_lines_per_image))
    # show_image_list(img_with_lines, fig_size=(15, 15))

    # full_lane_drawn_images = trace_lane_line(hough_lines_per_image, left_lane_lines, region_top_left[1], make_copy=True)

    return output


def hough_transform(canny_img, rho, theta, threshold, min_line_len, max_line_gap):
    return cv2.HoughLinesP(canny_img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)


def draw_lines(img, lines, color=[255, 0, 0], thickness=10, make_copy=True):
    # Copy the passed image
    img_copy = np.copy(img) if make_copy else img

    for line in lines:
        x1, y1, x2, y2 = line
        # for x1, y1, x2, y2 in line:
        cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)

    return img_copy


# Convenience function used to show a list of images
def show_image_list(img_list, cols=2, fig_size=(15, 15), img_labels="img", show_ticks=True):
    img_count = len(img_list)
    rows = img_count / cols
    cmap = None
    plt.figure(figsize=fig_size)
    for i in range(0, img_count):
        img_name = img_labels[i]

        plt.subplot(rows, cols, i + 1)
        img = img_list[i]
        if len(img.shape) < 3:
            cmap = "gray"

        if not show_ticks:
            plt.xticks([])
            plt.yticks([])

        plt.title("img")
        plt.imshow(img, cmap=cmap)

    plt.tight_layout()
    plt.show()

    from scipy import stats

def find_lane_lines_formula(lines):
    xs = []
    ys = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            xs.append(x1)
            xs.append(x2)
            ys.append(y1)
            ys.append(y2)

    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)

    # Remember, a straight line is expressed as f(x) = Ax + b. Slope is the A, while intercept is the b
    return (slope, intercept)


def trace_lane_line(img, lines, top_y, make_copy=True):
    A, b = find_lane_lines_formula(lines)
    # vert = get_vertices_for_img(img)

    img_shape = img.shape
    bottom_y = img_shape[0] - 1
    # y = Ax + b, therefore x = (y - b) / A
    x_to_bottom_y = (bottom_y - b) / A

    top_x_to_y = (top_y - b) / A

    new_lines = [[[int(x_to_bottom_y), int(bottom_y), int(top_x_to_y), int(top_y)]]]
    return draw_lines(img, new_lines, make_copy=make_copy)