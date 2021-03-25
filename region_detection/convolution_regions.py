from math import ceil

import numpy as np
from skimage.exposure import rescale_intensity

treashold = 150


def convolveFirstLayer(image, kernel, kH, kW):
    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    print(image.shape[:2])
    (iH, iW) = image.shape[:2]

    output = np.zeros((iH, iW), dtype="float32")

    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top to
    # bottom
    for y in np.arange(0, iH - kH):
        for x in np.arange(0, iW - kW):
            # extract the Region Of Interest(roi) of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y:y + kH, x:x + kW]

            # perform the actual convolution by taking the
            # element-wise multiplicate between the ROI and
            # the kernel, then summing the matrix
            k = np.math.fabs((roi * kernel).sum())

            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            output[y + kH - 1, x + kW - 1] = k

    # return the output image
    return output

def convolveFirstLayerWithSlopeByWidth(image, kernel, kH, kW):
    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    print(image.shape[:2])
    (iH, iW) = image.shape[:2]

    output = np.zeros((iH, iW), dtype="float32")

    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top to
    # bottom
    for y in np.arange(0, iH - kH):
        k_prev = "smooth"
        k_acc = 0
        maximal_pixel = (0, 0)
        max_k = 0
        for x in np.arange(0, iW - kW):
            # extract the Region Of Interest(roi) of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y:y + kH, x:x + kW]

            # perform the actual convolution by taking the
            # element-wise multiplication between the ROI and
            # the kernel, then summing the matrix
            k = (roi * kernel).sum()

            # find slope
            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            if k == 0:
                if k_prev == "smooth":
                    output[y + kH - 1, x + kW - 1] = 0
                else:
                    output[maximal_pixel[0], maximal_pixel[1]] = np.math.fabs(k_acc)
                    #output[y + kH - 1, x + kW - 1] = np.math.fabs(k_acc)
                    max_k = 0

                    k_acc = k
                    k_prev = "smooth"
            elif k > 0:
                if k_prev == "positive" or k_prev == "smooth":

                    maximal_pixel, max_k = save_max_k(k, max_k, (y + kH - 1, x + kW - 1), maximal_pixel)

                    k_acc = k + k_acc
                    output[y + kH - 1, x + kW - 1] = 0
                    k_prev = "positive"
                else:
                    output[maximal_pixel[0], maximal_pixel[1]] = np.math.fabs(k_acc)
                    # output[y + kH - 1, x + kW - 1] = np.math.fabs(k_acc)

                    max_k = 0
                    maximal_pixel, max_k = save_max_k(k, max_k, (y + kH - 1, x + kW - 1), maximal_pixel)

                    k_acc = k
                    k_prev = "positive"
            else:
                if k_prev == "negative" or k_prev == "smooth":

                    maximal_pixel, max_k = save_max_k(k, max_k, (y + kH - 1, x + kW - 1), maximal_pixel)

                    k_acc = k + k_acc
                    output[y + kH - 1, x + kW - 1] = 0
                    k_prev = "negative"
                else:
                    output[maximal_pixel[0], maximal_pixel[1]] = np.math.fabs(k_acc)
                    # output[y + kH - 1, x + kW - 1] = np.math.fabs(k_acc)

                    max_k = 0
                    maximal_pixel, max_k = save_max_k(k, max_k, (y + kH - 1, x + kW - 1), maximal_pixel)

                    k_acc = k
                    k_prev = "negative"

    # return the output image
    return output

def convolveFirstLayerWithSlopeByHeight(image, kernel, kH, kW):
    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    print(image.shape[:2])
    (iH, iW) = image.shape[:2]

    output = np.zeros((iH, iW), dtype="float32")

    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top to
    # bottom
    for x in np.arange(0, iW - kW):
        k_prev = "smooth"
        k_acc = 0
        maximal_pixel = (0, 0)
        max_k = 0
        for y in np.arange(0, iH - kH):
            # extract the Region Of Interest(roi) of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y:y + kH, x:x + kW]

            # perform the actual convolution by taking the
            # element-wise multiplication between the ROI and
            # the kernel, then summing the matrix
            k = (roi * kernel).sum()

            # find slope
            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            if k == 0:
                if k_prev == "smooth":
                    output[y + kH - 1, x + kW - 1] = 0
                else:
                    output[maximal_pixel[0], maximal_pixel[1]] = np.math.fabs(k_acc)
                    # output[y + kH - 1, x + kW - 1] = np.math.fabs(k_acc)
                    max_k = 0

                    k_acc = k
                    k_prev = "smooth"
            elif k > 0:
                if k_prev == "positive" or k_prev == "smooth":

                    maximal_pixel, max_k = save_max_k(k, max_k, (y + kH - 1, x + kW - 1), maximal_pixel)

                    k_acc = k + k_acc
                    output[y + kH - 1, x + kW - 1] = 0
                    k_prev = "positive"
                else:
                    output[maximal_pixel[0], maximal_pixel[1]] = np.math.fabs(k_acc)
                    # output[y + kH - 1, x + kW - 1] = np.math.fabs(k_acc)

                    max_k = 0
                    maximal_pixel, max_k = save_max_k(k, max_k, (y + kH - 1, x + kW - 1), maximal_pixel)

                    k_acc = k
                    k_prev = "positive"
            else:
                if k_prev == "negative" or k_prev == "smooth":

                    maximal_pixel, max_k = save_max_k(k, max_k, (y + kH - 1, x + kW - 1), maximal_pixel)

                    k_acc = k + k_acc
                    output[y + kH - 1, x + kW - 1] = 0
                    k_prev = "negative"
                else:
                    output[maximal_pixel[0], maximal_pixel[1]] = np.math.fabs(k_acc)
                    # output[y + kH - 1, x + kW - 1] = np.math.fabs(k_acc)

                    max_k = 0
                    maximal_pixel, max_k = save_max_k(k, max_k, (y + kH - 1, x + kW - 1), maximal_pixel)

                    k_acc = k
                    k_prev = "negative"

    # return the output image
    return output





def save_max_k(k, max_k, coord, maximal_pixel):
    if np.math.fabs(k) > max_k:
        max_k = np.math.fabs(k)
        maximal_pixel = coord
    return maximal_pixel, max_k

def convolveNextLayersWithoutDownScale(image, kernel, kH, kW):
    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    print(image.shape[:2])
    (iH, iW) = image.shape[:2]

    output = np.zeros((iH, iW), dtype="float32")

    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top to
    # bottom with stride kH/kW
    for y in np.arange(0, iH - kH, kH + 1):
        for x in np.arange(0, iW - kW, kW + 1):
            # extract the Region Of Interest(roi) of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y:y + kH, x:x + kW]
            roi_biased_horisontal = image[y:y + kH, x + 1:x + kW + 1]
            roi_biased_vertical = image[y + 1:y + kH + 1, x:x + kW]

            # perform the actual convolution by taking the
            # element-wise multiplicate between the ROI and
            # the kernel, then summing the matrix
            k = np.math.fabs((roi * kernel).sum())
            k_biased_horisontal = np.math.fabs((roi_biased_horisontal * kernel).sum())
            k_biased_vertical = np.math.fabs((roi_biased_vertical * kernel).sum())
            k_result = max(k, k_biased_horisontal, k_biased_vertical)

            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            for i in range(len(output)):
                for j in range(len(output[0])):
                    if (output[i][j] < (255 * treashold)):
                        output[i][j] = 0

            # = k_result

    # rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    # return the output image
    return output


def convolveNextLayersWithoutDownScaleByXAxis(image, kernel, kH, kW):
    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    print(image.shape[:2])
    (iH, iW) = image.shape[:2]

    output = np.zeros((iH, iW), dtype="float32")

    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top to
    # bottom with stride kH/kW
    for y in np.arange(0, iH - kH, kH):
        for x in np.arange(1, iW - kW, kW + 2):
            # extract the Region Of Interest(roi) of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y:y + kH, x:x + kW]
            roi_biased_forward = image[y:y + kH, x + 1:x + kW + 1]
            roi_biased_backward = image[y:y + kH, x - 1:x + kW - 1]

            # perform the actual convolution by taking the
            # element-wise multiplicate between the ROI and
            # the kernel, then summing the matrix
            k = np.math.fabs((roi * kernel).sum())
            k_biased_forward = np.math.fabs((roi_biased_forward * kernel).sum())
            k_biased_backward = np.math.fabs((roi_biased_backward * kernel).sum())
            k_result = max(k, k_biased_forward, k_biased_backward)

            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            for i in range(y, y + kH):
                for j in range(x - 1, x + kW + 1):
                    if (i == y + ceil(kH / 2)) & (k_result > 100):
                        output[i][j] = k_result
                    else:
                        output[i][j] = 0

    # rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    # return the output image
    return output


def convolveNextLayersWithoutDownScaleByYAxis(image, kernel, kH, kW):
    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    print(image.shape[:2])
    (iH, iW) = image.shape[:2]

    output = np.zeros((iH, iW), dtype="float32")

    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top to
    # bottom with stride kH/kW
    for x in np.arange(0, iW - kW, kW):
        for y in np.arange(1, iH - kH, kH + 2):
            # extract the Region Of Interest(roi) of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y:y + kH, x:x + kW]
            roi_biased_forward = image[y + 1:y + kH + 1, x:x + kW]
            roi_biased_backward = image[y - 1:y + kH - 1, x:x + kW]

            # perform the actual convolution by taking the
            # element-wise multiplicate between the ROI and
            # the kernel, then summing the matrix
            k = np.math.fabs((roi * kernel).sum())
            k_biased_forward = np.math.fabs((roi_biased_forward * kernel).sum())
            k_biased_backward = np.math.fabs((roi_biased_backward * kernel).sum())
            k_result = max(k, k_biased_forward, k_biased_backward)

            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            for j in range(x, x + kW):
                for i in range(y - 1, y + kH + 1):
                    if j == x + ceil(kW / 2):
                        output[i][j] = k_result
                    else:
                        output[i][j] = 0

    # rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    # return the output image
    return output


def convolveNextLayersWithDownScale(image, kernel, kH, kW, newHeight, newWeight):
    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    print(image.shape[:2])
    (iH, iW) = image.shape[:2]

    output = np.zeros((newHeight, newWeight), dtype="float32")

    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top to
    # bottom with stride kH/kW
    axisX = 0
    axisY = 0
    for y in np.arange(0, iH - kH, kH + 1):
        for x in np.arange(0, iW - kW, kW + 1):
            # extract the Region Of Interest(roi) of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y:y + kH, x:x + kW]
            roi_biased_horisontal = image[y:y + kH, x + 1:x + kW + 1]
            roi_biased_vertical = image[y + 1:y + kH + 1, x:x + kW]

            # perform the actual convolution by taking the
            # element-wise multiplicate between the ROI and
            # the kernel, then summing the matrix
            k = np.math.fabs((roi * kernel).sum())
            k_biased_horisontal = np.math.fabs((roi_biased_horisontal * kernel).sum())
            k_biased_vertical = np.math.fabs((roi_biased_vertical * kernel).sum())
            k_result = max(k, k_biased_horisontal, k_biased_vertical)

            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            output[axisY, axisX] = k_result
            axisX = axisX + 1
        axisY = axisY + 1
        axisX = 0

    # rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    # return the output image
    return output
