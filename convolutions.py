# import the necessary packages
import argparse
from math import ceil

import cv2
import numpy as np
from skimage.exposure import rescale_intensity

# 180 degree divided by 20 lines equal 9 degree between line
# So, approximation will be 4,5 degree from line to both sides.
# Which mean 4,5/180 = 2,5%. Or 6,25 out of 250.
treashold = 0.975


def invertImage(convoleOutputSum):
    inverseGrayImage = np.uint8(255) - convoleOutputSum
    cv2.imshow('Inversed image', inverseGrayImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convolve(image, kernel):
    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    print(image.shape[:2])
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # allocate memory for the output image, taking care to
    # "pad" the borders of the input image so the spatial
    # size (i.e., width and height) are not reduced
    padW = (kW - 1) // 2
    padH = (kH - 1) // 2

    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
                               cv2.BORDER_REPLICATE)
    output = np.zeros((iH + kH, iW + kW), dtype="float32")

    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top to
    # bottom
    # stride - kernel dimensions
    for y in np.arange(padH, iH + padH, kH):
        for x in np.arange(padW, iW + padW, kW):
            # extract the ROI of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y - padH:y + padH + 1, x - padW:x + padW + 1]

            # perform the actual convolution by taking the
            # element-wise multiplicate between the ROI and
            # the kernel, then summing the matrix
            k = np.math.fabs((roi * kernel).sum())

            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            for z1 in np.arange(y - padH, y + padH + 1):
                for z2 in np.arange(x - padW, x + padW + 2):
                    output[z1, z2] = k
                # output[y - padH, x - padW] = k

    # rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    # ______________________
    # Add relu/tnh/sigmoid

    # ???????????????? cv2.threshold change???
    for i in range(len(output)):
        for j in range(len(output[0])):
            if (output[i][j] < (255 * treashold)):
                output[i][j] = 0

    # return the output image
    return output


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
args = vars(ap.parse_args())

# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct a sharpening filter
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

# construct the Laplacian kernel used to detect edge-like
# regions of an image
laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]), dtype="int")

# construct the Sobel x-axis kernel
sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int")

sobelX = np.array((
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]), dtype="int")

# sobelX = np.array((
#     [-1, -1, 0, 1, 1],
#     [-1, -1, 0, 1, 1],
#     [-1, -1, 0, 1, 1],
#     [-1, -1, 0, 1, 1],
#     [-1, -1, 0, 1, 1],
#     [-1, -1, 0, 1, 1],
#     [-1, -1, 0, 1, 1]), dtype="int")

# construct the Sobel y-axis kernel
sobelY = np.array((
    [-1, -1, -1, -1, -1],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1]), dtype="int")

sobelY = np.array((
    [-1, -1, -1, -1, -1, -1, -1],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1]), dtype="int")

# sobelY = np.array((
#     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), dtype="int")
#
#
# sobelY = np.array((
#     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), dtype="int")

# sobelY = np.array((
#     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), dtype="int")

# sobelY = np.array((
#     [-1, -1, -1, -1, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1, -1],
#     [0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1]), dtype="int")

# sobelY = np.array((
#     [-1, -1, -1, -1, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1, -1],
#     [0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1]), dtype="int")


# construct the kernel bank, a list of kernels we're going
# to apply using both our custom `convole` function and
# OpenCV's `filter2D` function
kernelBank = (
    # ("small_blur", smallBlur),
    # ("large_blur", largeBlur),
    # ("sharpen", sharpen),
    # ("laplacian", laplacian),
    ("sobel_y", sobelY),
    ("sobel_x", sobelX)
)

# load the input image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create picture with all filters
(iH, iW) = image.shape[:2]
(kH, kW) = sobelY.shape[:2]
convoleOutputSum = np.zeros((iH + max(kH, kW), iW + max(kH, kW)), dtype="float32")

# loop over the kernels
for (kernelName, kernel) in kernelBank:
    # apply the kernel to the grayscale image using both
    # our custom `convole` function and OpenCV's `filter2D`
    # function
    print("[INFO] applying {} kernel".format(kernelName))
    convoleOutput = convolve(gray, kernel)
    opencvOutput = cv2.filter2D(gray, -1, kernel)

    # Create picture with all filters
    (iH, iW) = convoleOutput.shape[:2]
    (sH, sW) = convoleOutputSum.shape[:2]
    convoleOutput = cv2.copyMakeBorder(convoleOutput, ceil((sH - iH) / 2), ceil((sH - iH) / 2), ceil((sW - iW) / 2),
                                       ceil((sW - iW) / 2), cv2.BORDER_REPLICATE)
    # convoleOutputSum = np.add(convoleOutputSum, convoleOutput)
    convoleOutputSum = np.maximum(convoleOutputSum, convoleOutput)

    # show the output images
    cv2.imshow("original", gray)
    cv2.imshow("{} - convole".format(kernelName), convoleOutput)
    # cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# show the sum of output images
cv2.imshow("original", gray)
cv2.imshow("Convole sum", convoleOutputSum)
cv2.waitKey(0)
cv2.destroyAllWindows()

# invert sum image
invertImage(convoleOutputSum)
