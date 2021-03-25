from math import ceil

import cv2
import numpy as np

from region_detection.convolution_regions import convolveNextLayersWithoutDownScaleByXAxis, \
    convolveNextLayersWithDownScale, convolveNextLayersWithoutDownScaleByYAxis
from region_detection.util import get_medium_kernels, invert_and_treshold_image, get_44_kernels, upscaleImage


def second_layer_convolution(input_layer):
    kernelBank = get_medium_kernels()
    # kernelBank = get_44_kernels()
    (kH, kW) = kernelBank[1][1].shape[:2]

    (iH, iW) = input_layer.shape[:2]

    newHeight = ceil(iH / (kH + 1))
    newWeight = ceil(iW / (kW + 1))

    # allocate memory for the input image, if Height/Weight of image is not proportional to kernel Height/Weight
    # taking care to "pad" the borders of the input image so they will be proportional
    padW = (kW + 1) - (iW % (kW + 1))
    padH = (kH + 1) - (iH % (kH + 1))

    input_layer_adjusted = cv2.copyMakeBorder(input_layer, 0, padH, 0, padW, cv2.BORDER_REPLICATE)


    # convoleOutputSum = np.zeros((newHeight, newWeight), dtype="float32")
    convoleOutputSum = np.zeros(input_layer_adjusted.shape[:2], dtype="float32")

    for (kernelName, kernel, kH, kW) in kernelBank:
        print("[INFO] applying {} kernel".format(kernelName))
        # convoleOutput = convolveNextLayersWithDownScale(input_layer_adjusted, kernel, kH, kW, newHeight, newWeight)
        if (kernelName=="mediumX"):
            convoleOutput = convolveNextLayersWithoutDownScaleByXAxis(input_layer_adjusted, kernel, kH, kW)
        if (kernelName=="mediumY"):
            convoleOutput = convolveNextLayersWithoutDownScaleByYAxis(input_layer_adjusted, kernel, kH, kW)

        # add two images
        convoleOutputSum = np.add(convoleOutputSum, convoleOutput)

        th, im_th = invert_and_treshold_image(convoleOutput, 50)

        # show the output images
        cv2.imshow("original", input_layer)
        cv2.imshow("{} - convole".format(kernelName), im_th)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # show the output images
    th_sum, im_th_sum = invert_and_treshold_image(convoleOutputSum, 50)

    cv2.imshow("original", input_layer)
    cv2.imshow("{} - convole".format("Total"), im_th_sum)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # upscaledImage = upscaleImage(im_th_sum)

    return im_th_sum
