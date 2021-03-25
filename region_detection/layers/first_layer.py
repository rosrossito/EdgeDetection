import cv2
import numpy as np

from region_detection.convolution_regions import convolveFirstLayer, convolveFirstLayerWithSlopeByWidth, \
    convolveFirstLayerWithSlopeByHeight
from region_detection.util import get_smallest_kernels, invert_and_treshold_image, invertImage, cleanOutput


def first_layer_with_slope_convolution(gray_image):
    kernelBank = get_smallest_kernels()

    (iH, iW) = gray_image.shape[:2]

    convoleOutputSum = np.zeros((iH, iW), dtype="float32")

    for (kernelName, kernel, kH, kW) in kernelBank:
        print("[INFO] applying {} kernel".format(kernelName))

        if (kernelName=="smallestY"):
            convoleOutput = convolveFirstLayerWithSlopeByWidth(gray_image, kernel, kH, kW)
        else:
            convoleOutput = convolveFirstLayerWithSlopeByHeight(gray_image, kernel, kH, kW)

        # add two images
        convoleOutputSum = np.add(convoleOutputSum, convoleOutput)

        # Threshold.
        # Set values equal to or above threshold to 0.
        # Set values below threshold to 255.
        th, im_th = invert_and_treshold_image(convoleOutput, 20)
        # im_th = invertImage(convoleOutput, 255)

        # show the output images
        cv2.imshow("original", gray_image)
        cv2.imshow("{} - convole".format(kernelName), im_th)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # show the output images
    th_sum, im_th_sum = invert_and_treshold_image(convoleOutputSum, 20)
    # im_th_sum = invertImage(convoleOutputSum, 255)

    im_th_sum = cleanOutput(im_th_sum)

    cv2.imshow("original", gray_image)

    winname = "{} - convole".format("Total")
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, 40, 30)
    cv2.imshow(winname, im_th_sum)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    im_th_sum = cleanOutput(im_th_sum)

    return im_th_sum

def first_layer_convolution(gray_image):
    kernelBank = get_smallest_kernels()

    (iH, iW) = gray_image.shape[:2]

    convoleOutputSum = np.zeros((iH, iW), dtype="float32")

    for (kernelName, kernel, kH, kW) in kernelBank:
        print("[INFO] applying {} kernel".format(kernelName))
        convoleOutput = convolveFirstLayer(gray_image, kernel, kH, kW)
        # add two images
        convoleOutputSum = np.add(convoleOutputSum, convoleOutput)

        th, im_th = invert_and_treshold_image(convoleOutput, 10)
        # im_th = invertImage(convoleOutput)

        # show the output images
        cv2.imshow("original", gray_image)
        cv2.imshow("{} - convole".format(kernelName), im_th)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # show the output images
    th_sum, im_th_sum = invert_and_treshold_image(convoleOutputSum, 10)
    # im_th_sum = invertImage(convoleOutputSum)

    cv2.imshow("original", gray_image)
    cv2.imshow("{} - convole".format("Total"), im_th_sum)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return im_th_sum
