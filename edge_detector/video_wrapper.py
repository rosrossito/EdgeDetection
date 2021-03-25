import argparse

import cv2
import numpy as np
import skimage.measure
from skimage.color import gray2rgb
from skimage.transform import rescale

from deteсt_horizontal_line import detect_horizont_line
from edge_detector import save_image, find_upper_bound, get_max_result, create_lines_output, clear_redundant_lines, \
    get_coordinates, create_response, VANISHING_LINE_TRESHOLD
from gaus_mixture.gaussian_mixture_model import get_vitalii_gaussian_mixture
from increased_number_edges.create_48_edge_kernels import create_edge_kernel_of_size, \
    create_edge_kernel_of_size_and_angle

import argparse
import os
import time

import cv2

from line_detection.green_mask import get_edges_for_grass

VARIANCE_GAUSS = 30  # Pixels. If difference between y coordinates too big than gaussian mixture failed to detect tribunes correctly
DISTANCE_TRESHOLD = 10

def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='webcam', type=str)
    parser.add_argument('--output', default='output', type=str)
    # parser.add_argument('--inres', default='512,512', type=str)
    # parser.add_argument('--outres', default='1080,1920', type=str)
    parser.add_argument('--max-frames', default=10000, type=int)
    parser.add_argument('--fps', default=25.0 * 1.0, type=float)

    args, _ = parser.parse_known_args()
    # args.inres = tuple(int(x) for x in args.inres.split(','))
    # args.outres = tuple(int(x) for x in args.outres.split(','))
    os.makedirs(args.output, exist_ok=True)

    cap = cv2.VideoCapture(0 if args.video == 'webcam' else args.video)
    print(str(cap.isOpened()))
    # get vcap property
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # uses given video width and height
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fn = os.path.join(args.output, 'ctdet.' + os.path.basename(args.video)).replace('.mp4', '.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_fn, fourcc, args.fps, (width, height))
    k = 0
    tic = time.time()
    frame = 0
    detections = []
    while cap.isOpened():
        if k > args.max_frames:
            print("Bye")
            break
        if k > 0 and k % 100 == 0:
            toc = time.time()
            duration = toc - tic
            print("[%05d]: %.3f seconds / 100 iterations" % (k, duration))
            tic = toc

        k += 1
        ret, img = cap.read()
        if not ret:
            print("Done")
            break

        # detection
        if type(img) is not np.ndarray:
            img = cv2.imread(img)
        else:
            img = img


        # gray_image = get_gaussian_mixture(img)
        grass_mask, gray_image = get_vitalii_gaussian_mixture(img)

        # gray_image = load_Image(args["image"])
        (iH, iW) = gray_image.shape[:2]
        output = np.zeros((iH, iW), dtype="float32")

        output = cv2.Canny(gray_image.astype("uint8"), 100, 200)
        second_output = get_edges_for_grass(img)

        # potential_edges = create_edges(gray_image, iH, iW, output)
        # print("Total pixels: " + str(iH * iW))
        # print("Out of them potential edges before cleaning: " + str(len(potential_edges)))
        # print("Or in percentage: {0:.0%}".format(len(potential_edges) / (iH * iW)))

        # maxpool
        output = skimage.measure.block_reduce(output, (2, 2), np.max)
        output = skimage.measure.block_reduce(output, (2, 2), np.max)
        output = skimage.measure.block_reduce(output, (2, 2), np.max)

        save_image(gray_image, 'output_before_cleaning')
        save_image(output, 'output_edges')

        cut_upper_edge, is_variance_acceptable = find_upper_bound(output)

        if is_variance_acceptable:

            (iHpool, iWpool) = output.shape[:2]

            img = cv2.add(img.astype('float32'), detect_horizont_line(second_output, iHpool, iWpool))

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
                                    result = sum((sum(mask)))

                                    max_result, max_edge = get_max_result(result, edge, max_result, max_edge)

                                    # percentage = round(result / 255 / np.ndarray.sum(kernel) * 100)
                                    percentage = round(result / 255 / iHk * 100)

                                    if percentage > VANISHING_LINE_TRESHOLD:
                                        # print("Bingo! " + "Edge: " + str(edge) + ", column: " + str(n) + " - " + str(
                                        #     result) +
                                        #       ", active pixels: " + str(result / 255) + ", percentage: " + str(
                                        #     percentage))
                                        create_lines_output(kernel_out, lines, edge, percentage, edge_with_max_result,
                                                            iHk, n,
                                                            iWk)

                            # print("Maximum. " + "Edge: " + str(edge) + " - " + str(max_result) +
                            #       ", active pixels: " + str(max_result / 255) + ", percentage: " + str(
                            #     round(max_result / 255 / np.ndarray.sum(kernel) * 100)))

            lines = clear_redundant_lines(lines)

            vanishing_lines = []
            line_midfield = []
            for line in lines:
                output_conv[0:line[1][1], line[1][2] - 1:line[1][2] - 1 + line[1][3]] = line[1][0]
                output_conv_total = cv2.add(output_conv_total, output_conv)
                if line[0] == 0:
                    line_midfield.append(
                        get_coordinates(cut_upper_edge, 8, line[1][1], line[1][2], line[1][3], line[0]))
                    vanishing_lines.append(
                        get_coordinates(cut_upper_edge, 8, line[1][1], line[1][2], line[1][3], line[0]))
                else:
                    vanishing_lines.append(
                        get_coordinates(cut_upper_edge, 8, line[1][1], line[1][2], line[1][3], line[0]))

            image_rescaled = rescale(output_conv_total[:, 45:205], 8, anti_aliasing=False)
            image_rescaled_3d = skimage.color.gray2rgb(image_rescaled)
            # back to original height
            image_rescaled_3d = cv2.copyMakeBorder(image_rescaled_3d, cut_upper_edge * 8, 0, 0, 0, cv2.BORDER_CONSTANT)

            output_with_vanishing_lines = cv2.add(img.astype('uint8'), image_rescaled_3d.astype('uint8'))

            # save_image(output_with_vanishing_lines,
            #            'output_edges_filters_' + os.path.splitext(os.path.basename(args["image"]))[0])
            detection = create_response(vanishing_lines, line_midfield), output_with_vanishing_lines

            detections.append(detection)

            frame = frame + 1
            print(frame)
            out.write(output_with_vanishing_lines)
        else:
            out.write(img)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video saved to: %s" % out_fn)


if __name__ == '__main__':
    main()
