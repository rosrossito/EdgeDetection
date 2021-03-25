import argparse
import math

import cv2
import numpy as np
from skimage.measure import block_reduce

from region_detection.util import load_Image

VISIBILITY_THRESHOLD = 20  # make it different to pass to network images with different threshold
SMOOTHING_THRESHOLD = 20
KERNEL_SIZE = 2
IMAGES_SIZES = [512, 1024, 2048]


def main(args):
    gray_image = load_Image(args["image"])
    (iHinitial, iWinitial) = gray_image.shape[:2]
    gray_image = extend_image(iHinitial, iWinitial, gray_image)
    (iH, iW) = gray_image.shape[:2]

    convolved_images = []
    new_image = gray_image
    height = iH
    convolved_images.append(new_image)
    while height > 2:
        new_image = create_downsampled_image(new_image, KERNEL_SIZE)
        convolved_images.append(new_image)
        (height, _) = new_image.shape[:2]

    steps = math.ceil(len(convolved_images) - 1)
    for i in range(steps):
        last_image = convolved_images.pop()
        previous_image = convolved_images.pop()
        output = create_upsampled_image(last_image, previous_image, KERNEL_SIZE)
        convolved_images.append(output)

    output = cut_image(iHinitial, iWinitial, output)
    show_image(output, iH)

def cut_image (iHinitial, iWinitial, image):
    output = np.zeros((iHinitial, iWinitial))
    output[0:iHinitial, 0:iWinitial]  = image[0:iHinitial, 0:iWinitial]
    return output.astype("uint8")

def extend_image(iHinitial, iWinitial, gray_image):
    # need padding for correct image deconvolution
    iH = 0
    iW = 0
    length = len(IMAGES_SIZES)
    for i in reversed(range(length)):
        if iHinitial <= IMAGES_SIZES[i]:
            iH = IMAGES_SIZES[i]
        if iWinitial <= IMAGES_SIZES[i]:
            iW = IMAGES_SIZES[i]
    color = [0, 0, 0]
    return cv2.copyMakeBorder(gray_image, 0, iH - iHinitial, 0, iW - iWinitial, cv2.BORDER_CONSTANT, value=color)


def create_upsampled_image(last_image, previous_image, size):
    (iH, iW) = last_image.shape[:2]
    y_prev = 0
    for y in np.arange(0, iH):
        # x_prev = 0
        for x in np.arange(0, iW):
            if last_image[y, x] != 0:
                previous_image[y * size:y * size + size, x * size:x * size + size] = last_image[y, x]
        #     x_prev += size
        # y_prev += size
    return previous_image

def show_image(gray_image, iH):
    # show the output images
    cv2.imshow("{} - convole".format(iH), gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_downsampled_image(gray_image, size):
    # perform pooling
    mean_pool = block_reduce(gray_image, block_size=(size, size), func=np.mean)
    max_pool = block_reduce(gray_image, block_size=(size, size), func=np.max)
    min_pool = block_reduce(gray_image, block_size=(size, size), func=np.min)

    # create mask: 0 - if min max diff is more than smothing threshold, 255 - otherwise
    _, mask_edges = cv2.threshold((max_pool - min_pool), SMOOTHING_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    return cv2.bitwise_and(mean_pool, mean_pool, mask=mask_edges).astype(np.uint8)


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


def get_train_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to the input image")
    return vars(ap.parse_args())


if __name__ == '__main__':
    main(get_train_args())
