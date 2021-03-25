import argparse

import cv2
import numpy as np
import skimage.measure
from scipy import ndimage
from skimage.transform import rescale
from skimage.color import gray2rgb
from region_detection.util import load_Image


def main(args):
    # img = cv2.imread(args["image"])
    gray_image = load_Image(args["image"])
    cv2.imshow("result", gray_image)
    # gray_image = np.array([[1, 1, 1, 4],
    #               [1, 1, 1, 8],
    #               [1, 1, 1, 5],
    #               [0, 0, 0, 0]], dtype=np.float)

    mask = np.ones((3, 3))
    mask[0:1] = 0
    result = ndimage.generic_filter(gray_image, func, footprint=mask, mode='constant', cval=np.NaN)
    result_out = (result*(255/np.max(result))*20).astype("uint8")
    cv2.imshow("result", result_out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def func(x):
    x[4] = -x[4]*(5 - np.count_nonzero(np.isnan(x)))
    return np.abs(np.nanmean(x))


def get_train_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to the input image")
    return vars(ap.parse_args())


if __name__ == '__main__':
    main(get_train_args())
