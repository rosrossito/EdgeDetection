import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_Image(image):
    # load the input image and convert it to grayscale
    gray_image = cv2.imread(image)
    return cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)


def cleanOutput(image):
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] < 255 \
                    and image[i - 1][j + 1] == 255 \
                    and image[i - 1][j] == 255 \
                    and image[i - 1][j - 1] == 255 \
                    and image[i][j - 1] == 255 \
                    and image[i + 1][j - 1] == 255 \
                    and image[i + 1][j] == 255 \
                    and image[i + 1][j + 1] == 255 \
                    and image[i][j + 1] == 255:
                image[i][j] = 255

    return image


def get_smallest_kernels():
    # construct the smallest x-axis & y-axis kernel
    smallestY = np.array(([-1, 1]), dtype="int")
    kWY = 1
    kHY = smallestY.shape[0]

    smallestX = np.array(([-1], [1]), dtype="int")
    kWX = smallestX.shape[0]
    kHX = 1

    smallestLD = np.array(([-1, 0], [0, 1]), dtype="int")
    (kHLD, kWLD) = smallestLD.shape[:2]

    kernelBank = (
        ("smallestY", smallestY, kWY, kHY),
        ("smallestX", smallestX, kWX, kHX)
        # ,
        # ("smallestLD", smallestLD, kHLD, kWLD)
    )

    return kernelBank


def get_medium_kernels():
    mediumX = np.array((
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1]), dtype="int")

    (kWX, kHX) = mediumX.shape[:2]

    mediumY = np.array((
        [-1, -1, -1, 1, 1, 1],
        [-1, -1, -1, 1, 1, 1],
        [-1, -1, -1, 1, 1, 1],
        [-1, -1, -1, 1, 1, 1],
        [-1, -1, -1, 1, 1, 1],
        [-1, -1, -1, 1, 1, 1]), dtype="int")

    (kWY, kHY) = mediumY.shape[:2]

    kernelBank = (
        ("mediumX", mediumX, kWX, kHX),
        ("mediumY", mediumY, kWY, kHY)
    )

    return kernelBank


def get_44_kernels():
    mediumX = np.array((
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]), dtype="int")

    (kWX, kHX) = mediumX.shape[:2]

    mediumY = np.array((
        [-1, -1, 1, 1],
        [-1, -1, 1, 1],
        [-1, -1, 1, 1],
        [-1, -1, 1, 1]), dtype="int")

    (kWY, kHY) = mediumY.shape[:2]

    kernelBank = (
        ("mediumX", mediumX, kWX, kHX),
        ("mediumY", mediumY, kWY, kHY)
    )

    return kernelBank


def invert_and_treshold_image(im_in, treashold):
    return cv2.threshold(im_in, treashold, 255, cv2.THRESH_BINARY_INV)


def invertImage(im_in, treashold):
    for i in range(len(im_in)):
        for j in range(len(im_in[0])):
            im_in[i][j] = 255 - im_in[i][j]
            if (im_in[i][j] > treashold):
                im_in[i][j] = 255
            #
            # if (im_in[i][j] > treashold):
            #     im_in[i][j] = 0
            # else:
            #     im_in[i][j] = 255 - im_in[i][j]

    # return the output image
    return im_in


def upscaleImage(img):
    print('Original Dimensions : ', img.shape)

    scale_percent = 600  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print('Resized Dimensions : ', resized.shape)

    cv2.imshow("Resized image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return resized


def visualizePic(imgs, format=None, gray=False):
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)
        plt_idx = i+1
        plt.subplot(2, 2, plt_idx)
        plt.imshow(img, format)
    plt.show()