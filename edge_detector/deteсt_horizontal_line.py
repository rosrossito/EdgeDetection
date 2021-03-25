import skimage
from skimage.color import gray2rgb
from skimage.transform import rescale
import cv2

from increased_number_edges.create_48_edge_kernels import create_edge_kernel_of_size, \
    create_edge_kernel_of_size_and_angle
import numpy as np

from line_detection.background_subtractor_MOG import substract
from line_detection.green_mask import find_grass

HORIZONTAL_LINE_TRESHOLD = 40


def detect_horizont_line(img, iH, iW):

    output_conv_total = np.zeros((iH, iW), dtype="float32")
    if iW % 2 == 0:
        kernel_size = iW - 1
    else:
        kernel_size = iW
    kernelEdgeBank = create_edge_kernel_of_size_and_angle(kernel_size)

    lines = []
    edge_with_max_result = {}
    output_conv = np.zeros((iH, iW), dtype="float32")
    for key in kernelEdgeBank:
        # find vanishing lines
        if key == kernel_size:
            for edge, kernel in kernelEdgeBank[key]:
                if edge > 85 and edge < 95:
                    max_result = 0
                    max_edge = 0

                    print("Edge: " + str(edge) + ", kernel_sum: " + str(np.ndarray.sum(kernel)))
                    (iHs, iWs) = output_conv.shape[:2]
                    # kernel_tr = np.transpose(kernel)
                    kernel_out = kernel[~(kernel == 0).all(1)]
                    # kernel_out = np.transpose(kernel_new)

                    (iHk, iWk) = kernel_out.shape[:2]
                    for n in range(1, iHs):
                        if n - 1 + iHk < iHs:
                            roi_image = img[n - 1:n - 1 + iHk, 0:iWk]
                            mask = np.multiply(kernel_out, roi_image)
                            result = abs(sum((sum(mask))))

                            # percentage = round(result / 255 / np.ndarray.sum(kernel) * 100)
                            percentage = round(result / 255 / iWk * 100)

                            if percentage > HORIZONTAL_LINE_TRESHOLD:
                                print("Bingo! " + "Edge: " + str(edge) + ", raw: " + str(n) + " - " + str(result) +
                                      ", active pixels: " + str(result / 255) + ", percentage: " + str(percentage))
                                create_lines_output(kernel_out, lines, edge, percentage, iHk, n,
                                                    iWk)

                    print("Maximum. " + "Edge: " + str(edge) + " - " + str(max_result) +
                          ", active pixels: " + str(max_result / 255) + ", percentage: " + str(
                        # round(max_result / 255 / np.ndarray.sum(kernel) * 100)))
                        round(max_result / 255 / iWk * 100)))

    # vanishing_lines = []
    # line_midfield = []
    for line in lines:
        output_conv[line[2] - 1:line[2] - 1 + line[1],0:line[3]] = line[0]
        output_conv_total = cv2.add(output_conv_total, output_conv)
        # if line[0] == 0:
        #     line_midfield.append(get_coordinates(cut_upper_edge, 8, line[1][1], line[1][2], line[1][3], line[0]))
        #     vanishing_lines.append(get_coordinates(cut_upper_edge, 8, line[1][1], line[1][2], line[1][3], line[0]))
        # else:
        #     vanishing_lines.append(get_coordinates(cut_upper_edge, 8, line[1][1], line[1][2], line[1][3], line[0]))

    image_rescaled = rescale(output_conv_total[:, :], 8, anti_aliasing=False)
    return skimage.color.gray2rgb(image_rescaled)
    # back to original height
    # image_rescaled_3d = cv2.copyMakeBorder(image_rescaled_3d, cut_upper_edge * 8, 0, 0, 0, cv2.BORDER_CONSTANT)


def create_lines_output(kernel_out, lines, edge, percentage, iHk, n, iWk):
    lines.append([kernel_out.astype('float64') * 255, iHk, n, iWk, percentage])

def get_max_result(result, edge, max_result, max_edge):
    if result > max_result:
        return result, edge
    else:
        return max_result, max_edge

