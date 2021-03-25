
import matplotlib.pyplot as plt
import argparse
import os

import cv2
import numpy as np
import skimage.measure
from skimage.color import gray2rgb
from skimage.transform import rescale

from deteсt_horizontal_line import detect_horizont_line
from gaus_mixture.gaussian_mixture_model import get_vitalii_gaussian_mixture, get_gaussian_mixture
from increased_number_edges.create_48_edge_kernels import create_edge_kernel_of_size_and_angle

# VISIBILITY_THRESHOLD = 20  # make it different to pass to network images with different threshold
# SMOOTHING_THRESHOLD = 5
from line_detection.background_subtractor_MOG import substract
from line_detection.green_mask import find_grass, get_edges_for_grass, get_edges_for_grass_without_pooling
from line_detection.polyfit import find_line
from player_erasing.player_erase import erase

from player_erasing.player_load import load_players_labels_in_frame
from rect_lookup.find_rect import find_rectangle

VARIANCE_GAUSS = 30  # Pixels. If difference between y coordinates too big than gaussian mixture failed to detect tribunes correctly
VANISHING_LINE_TRESHOLD = 40
DISTANCE_TRESHOLD = 15
MAXIMUM_NUMBER_VANISHING_LINES = 20


def get_max_result(result, edge, max_result, max_edge):
    if result > max_result:
        return result, edge
    else:
        return max_result, max_edge


def output_kernel(kernel, edge):
    (iH, iW) = kernel.shape[:2]
    kernel_output = np.zeros((iH, iW), dtype="float32")
    kernel_output = cv2.add(kernel_output, kernel.astype('float32') * 255)
    kernel_output = kernel_output.astype("uint8")
    cv2.imshow("Edge: " + str(edge), kernel_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def output_kernels(kernelEdgeBank, size):
    kernel_output = np.zeros((size, size), dtype="float32")
    for edge, kernel in kernelEdgeBank[size]:
        kernel_output = cv2.add(kernel_output, kernel.astype('float32') * 255)
    kernel_output = kernel_output.astype("uint8")
    cv2.imshow("mixture", kernel_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_upper_bound(output, variance):
    x_arr = []
    y_arr = []
    (iH, iW) = output.shape[:2]

    # find first pixel > 0 in column
    for x in np.arange(0, iW):
        for y in np.arange(0, iH):
            if output[y, x] > 0:
                y_arr.append(y)
                x_arr.append(x)
                break

    # approximation
    # line = np.round(np.polyfit(y_arr, x_arr, 1)).astype(int)
    # plt.hist(y_arr)
    # plt.show()

    # plot the line on image
    # y_start = line[1]
    # y_end = line[0] * iW + line[1]
    # cv2.line(output, (y_start, 0), (y_end, iW), (255, 0, 0), 2)
    # cv2.imshow("Upper bound", output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # delete max and min element
    y_arr = sorted(y_arr)
    del y_arr[0]
    del y_arr[-1]

    is_variance_acceptable = (max(y_arr) - min(y_arr)) < variance
    print("Variance: " + str((max(y_arr) - min(y_arr))) + ", Max: " + str(max(y_arr)) + ", Min: " + str(min(y_arr)))
    cut_edge = round(int(sum(y_arr) / len(y_arr)))

    return cut_edge, is_variance_acceptable


def get_coordinates(cut_upper_edge, multiplicator, iHk, n, iWk, edge):
    if edge < 13:
        return [(n - 1 - 45) * multiplicator, (iHk + cut_upper_edge) * multiplicator, (
                n - 1 + iWk - 45) * multiplicator, cut_upper_edge * multiplicator]
    if edge > 36:
        return [(n - 1 - 45) * multiplicator, cut_upper_edge * multiplicator, (n - 1 + iWk - 45) * multiplicator, (
                iHk + cut_upper_edge) * multiplicator]


def predict(arg_im):
    if type(arg_im) is not np.ndarray:
        img = cv2.imread(arg_im)
    else:
        img = arg_im

    pos = load_players_labels_in_frame("ann/100.txt")
    img = erase(img, pos)
    save_image(img, 'erased')

    # # RGB - Blue
    # blue = img[:, :, 0]
    # cv2.imshow('B-RGB.jpg', blue)
    # # cv2.imwrite('B-RGB.jpg', img[:, :, 0])
    # # RGB - Green
    # green = img[:, :, 1]
    # cv2.imshow('G-RGB', green)
    # # cv2.imwrite('G-RGB.jpg', img[:, :, 1])
    # # RGB Red
    # red = img[:, :, 2]
    # cv2.imshow('R-RGB', red)
    # # cv2.imwrite('R-RGB.jpg', img[:, :, 2])
    # b = img.copy()
    # # set green and red channels to 0
    # b[:, :, 1] = 0
    # b[:, :, 2] = 0
    # blue = img[:, :, 0]
    # last_raw = blue[719, :]
    # middle_raw = blue[360, :]
    # first_quater_raw = blue[140, :]

    # gray_image = get_gaussian_mixture(img)
    grass_mask, gray_image = get_vitalii_gaussian_mixture(img)

    gray_image, cut_for_initial = preprocess_image(gray_image)
    img_rect = find_rectangle(img[cut_for_initial:img.shape[:2][0]][0:img.shape[:2][1]], gray_image)


    # gray_image = load_Image(args["image"])
    (iH, iW) = gray_image.shape[:2]
    output = np.zeros((iH, iW), dtype="float32")

    output = cv2.Canny(gray_image.astype("uint8"), 100, 200)
    # output = cv2.Canny(img, 5, 50)
    second_output = get_edges_for_grass(img)

    # potential_edges = create_edges(gray_image, iH, iW, output)
    # print("Total pixels: " + str(iH * iW))
    # print("Out of them potential edges before cleaning: " + str(len(potential_edges)))
    # print("Or in percentage: {0:.0%}".format(len(potential_edges) / (iH * iW)))

    # h_transform(output, gray_image)
    # find_line(output, img)

    # maxpool
    output = skimage.measure.block_reduce(output, (2, 2), np.max)
    output = skimage.measure.block_reduce(output, (2, 2), np.max)
    output = skimage.measure.block_reduce(output, (2, 2), np.max)

    save_image(gray_image, 'output_before_cleaning')
    save_image(output, 'output_edges')

    cut_upper_edge, is_variance_acceptable = find_upper_bound(output, VARIANCE_GAUSS)

    if not is_variance_acceptable:
        return "Skip frame. Variance is not acceptable"

    (iHpool, iWpool) = output.shape[:2]

    img = cv2.add(img.astype('float32'), detect_horizont_line(output, iHpool, iWpool))

    output_cut = output[cut_upper_edge:iHpool][0:iWpool]

    output_with_borders = cv2.copyMakeBorder(output_cut, 0, 0, 45, 45, cv2.BORDER_CONSTANT)
    (iHcut, iWcut) = output_with_borders.shape[:2]

    output_conv_total = np.zeros((iHcut, iWcut), dtype="float32")

    if iHcut % 2 == 0:
        kernel_size = iHcut - 1
    else:
        kernel_size = iHcut
    kernelEdgeBank = create_edge_kernel_of_size_and_angle(kernel_size)
    # output_kernels(kernelEdgeBank, kernel_size)

    lines = {}
    edge_with_max_result = {}
    output_conv = np.zeros((iHcut, iWcut), dtype="float32")
    for key in kernelEdgeBank:
        # find vanishing lines
        if key == kernel_size:
            for edge, kernel in kernelEdgeBank[key]:
                if edge < 13 or edge > 12:
                    max_result = 0
                    max_edge = 0

                    print("Edge: " + str(edge) + ", kernel_sum: " + str(np.ndarray.sum(kernel)))
                    (iHs, iWs) = output_conv.shape[:2]
                    kernel_tr = np.transpose(kernel)
                    kernel_new = kernel_tr[~(kernel_tr == 0).all(1)]
                    kernel_out = np.transpose(kernel_new)

                    (iHk, iWk) = kernel_out.shape[:2]
                    for n in range(1, iWs):
                        if n - 1 + iWk < iWs:
                            roi_image = output_with_borders[0:iHk, n - 1:n - 1 + iWk]
                            mask = np.multiply(kernel_out, roi_image)
                            result = abs(sum((sum(mask))))

                            max_result, max_edge = get_max_result(result, edge, max_result, max_edge)

                            # percentage = round(result / 255 / np.ndarray.sum(kernel) * 100)
                            percentage = round(result / 255 / iHk * 100)

                            if percentage > VANISHING_LINE_TRESHOLD:
                                print("Bingo! " + "Edge: " + str(edge) + ", column: " + str(n) + " - " + str(result) +
                                      ", active pixels: " + str(result / 255) + ", percentage: " + str(percentage))
                                create_lines_output(kernel_out, lines, edge, percentage, edge_with_max_result, iHk, n,
                                                    iWk)

                    print("Maximum. " + "Edge: " + str(edge) + " - " + str(max_result) +
                          ", active pixels: " + str(max_result / 255) + ", percentage: " + str(
                        # round(max_result / 255 / np.ndarray.sum(kernel) * 100)))
                        round(max_result / 255 / iHk * 100)))

    lines = clear_redundant_lines(lines)

    vanishing_lines = []
    line_midfield = []
    for line in lines:
        output_conv[0:line[1][1], line[1][2] - 1:line[1][2] - 1 + line[1][3]] = line[1][0]
        output_conv_total = cv2.add(output_conv_total, output_conv)
        if line[0] == 0:
            line_midfield.append(get_coordinates(cut_upper_edge, 8, line[1][1], line[1][2], line[1][3], line[0]))
            vanishing_lines.append(get_coordinates(cut_upper_edge, 8, line[1][1], line[1][2], line[1][3], line[0]))
        else:
            vanishing_lines.append(get_coordinates(cut_upper_edge, 8, line[1][1], line[1][2], line[1][3], line[0]))

    image_rescaled = rescale(output_conv_total[:, 45:205], 8, anti_aliasing=False)
    image_rescaled_3d = skimage.color.gray2rgb(image_rescaled)
    # back to original height
    image_rescaled_3d = cv2.copyMakeBorder(image_rescaled_3d, cut_upper_edge * 8, 0, 0, 0, cv2.BORDER_CONSTANT)

    output_with_vanishing_lines = cv2.add(img.astype('float32'), image_rescaled_3d)

    if type(arg_im) is not np.ndarray:
        save_image(output_with_vanishing_lines,
                   'output_edges_filters_' + os.path.splitext(os.path.basename(arg_im))[0])
    return create_response(vanishing_lines, line_midfield), output_with_vanishing_lines


def clear_redundant_lines(lines):
    differences_column = []
    filtered_lines = []
    close_lines = []
    lines = sorted(lines.items(), key=lambda x: x[1][2], reverse=False)
    for i in range(1, len(lines)):
        differences_column.append((lines[i][1][2] - lines[i - 1][1][2]) > DISTANCE_TRESHOLD)
    if differences_column:
        for i in range(len(differences_column)):
            if differences_column[i]:
                if close_lines:
                    close_lines.append(lines[i])
                    maximum_percentage = close_lines[0][1][4]
                    maximum_line = close_lines[0]
                    for k in range(1, len(close_lines)):
                        if close_lines[k][1][4] > maximum_percentage:
                            maximum_percentage = close_lines[k][1][4]
                            maximum_line = close_lines[k]
                    filtered_lines.append(maximum_line)
                    close_lines = []
                else:
                    filtered_lines.append(lines[i])
                # if last element is true - append last line
                if i == (len(differences_column) - 1):
                    filtered_lines.append(lines[i + 1])
            else:
                close_lines.append(lines[i])
                # if last element is false
                if i == (len(differences_column) - 1):
                    close_lines.append(lines[i + 1])
                    maximum_percentage = close_lines[0][1][4]
                    maximum_line = close_lines[0]
                    for k in range(1, len(close_lines)):
                        if close_lines[k][1][4] > maximum_percentage:
                            maximum_percentage = close_lines[k][1][4]
                            maximum_line = close_lines[k]
                    filtered_lines.append(maximum_line)
                    close_lines = []
        filtered_lines = sorted(filtered_lines, key=lambda x: x[1][4], reverse=False)
    else:
        filtered_lines = lines
    while len(filtered_lines) > MAXIMUM_NUMBER_VANISHING_LINES:
        del filtered_lines[0]
    return filtered_lines


# def create_edges(gray_image, iH, iW, output):
#     # for coordinates in potential_edges:
#     # decide what we will do with this pixel acording to some rule
#     # if value of pixel is between two kind of surrounding values - asign to it value from closest surrounding value
#     # if surrounding values are the same and value of pixel differs from other surrounding values less than
#     # threshold_visibility - asign to it surrounding value
#
#     potential_edges = []
#     sizes = [2]
#     for size in sizes:
#         for y in np.arange(size, iH + 1):
#             if y % 10 == 0:
#                 print("size: " + str(size) + ", lines height: " + str(y))
#             for x in np.arange(size, iW + 1):
#                 roi = create_roi(gray_image, x, y, size)
#                 variance = np.max(roi) - np.min(roi)
#                 if variance < SMOOTHING_THRESHOLD:
#                     average = round(np.average(roi))
#                     fill_in_output_image(average, gray_image, x, y, size)
#                 elif size == 2:
#                     add_potential_edges(potential_edges, x, y, size)
#                     output[y - size, x - size] = 255
#     return potential_edges


def create_response(vanishing_lines, line_midfield):
    return {'line_upper_bound': None,
            'line_lower_bound': None,
            'line_midfield': np.asarray(line_midfield),
            'line_side_L': None, 'line_side_R': None,
            'point_van': None,
            'lines_van': np.asarray(vanishing_lines)}


def create_lines_output(kernel_out, lines, edge, percentage, edge_with_max_result, iHk, n, iWk):
    is_edge_present = edge_with_max_result.get(edge)
    if not is_edge_present or (edge_with_max_result[edge] < percentage):
        edge_with_max_result[edge] = percentage
        lines[edge] = [kernel_out.astype('float64') * 255, iHk, n, iWk, percentage]


def add_potential_edges(potential_edges, x, y, size):
    potential_edges.append([y - size, x - size])


def create_roi(gray_image, x, y, size):
    roi = gray_image[(y - size):y, (x - size):x]
    return roi


def save_image(image, base_path, ext='jpg'):
    cv2.imwrite('{}.{}'.format(base_path, ext), image)


def fill_in_output_image(average, gray_image, x, y, size):
    matrix = [[average] * size for i in range(size)]
    gray_image[(y - size):y, (x - size):x] = matrix

def preprocess_image(img):
    cut_upper_edge, is_variance_acceptable = find_upper_bound(img, VARIANCE_GAUSS*8)
    if not is_variance_acceptable:
        print("Skip frame. Variance is not acceptable")
        exit(0)
    (iH, iW) = img.shape[:2]
    return img[cut_upper_edge:iH][0:iW], cut_upper_edge




def get_train_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to the input image")
    ap.add_argument("-v", "--video", required=False,
                    help="path to the input video")
    return vars(ap.parse_args())


def main(args):
    predict(args["image"])


if __name__ == '__main__':
    main(get_train_args())
