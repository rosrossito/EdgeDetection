import argparse

import cv2
import numpy as np
import skimage.measure
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from scipy import ndimage
from skimage.transform import rescale
from skimage.color import gray2rgb
from region_detection.util import load_Image
from scipy.signal import fftconvolve

from сolor_сlasses.luminance import Luminance


def main(args):
    # img = cv2.imread(args["image"])
    image = cv2.imread(args["image"])

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    (iH, iW) = image_rgb.shape[:2]

    # cv2.imshow("result", image_rgb)

    mask = np.ones((3, 3)) / 9.
    mask_red = mask * 0.299
    red_channel = image_rgb[:, :, 0]
    result_red_channel = fftconvolve(red_channel, mask_red, mode='same')

    mask_green = mask * 0.587
    green_channel = image_rgb[:, :, 1]
    result_green_channel = fftconvolve(green_channel, mask_green, mode='same')

    mask_blue = mask * 0.114
    blue_channel = image_rgb[:, :, 2]
    result_blue_channel = fftconvolve(blue_channel, mask_blue, mode='same')

    # simple grayimage transformation?
    result = (result_red_channel + result_green_channel + result_blue_channel).astype("uint8")

    lumin_values = sorted([e.value for e in Luminance])

    for y in range(iH):
        for x in range(iW):
            result[y, x] = min(lumin_values, key=lambda i: abs(i - result[y, x]))

    # result = ndimage.generic_filter(image_rgb, func, footprint=mask, mode='constant', cval=np.NaN)
    # result_out = (result * (255 / np.max(result)) * 20)
    cv2.imshow("result", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def func(x):
    # x[4] = -x[4]*(5 - np.count_nonzero(np.isnan(x)))
    print(x)
    return np.abs(np.nanmean(x))


def get_train_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to the input image")
    return vars(ap.parse_args())


if __name__ == '__main__':
    main(get_train_args())
