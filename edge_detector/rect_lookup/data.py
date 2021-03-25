import math
import os
from copy import deepcopy

import cv2
import numpy as np
import skimage.io as io
import skimage.measure
import skimage.transform as trans
from skimage import img_as_ubyte

Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                       Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def testGeneratorOneImg(img, target_size=(720, 1280), flag_multi_class=False):
    img_copy = deepcopy(img)
    while True:
        img_copy[img_copy == 0] = -1
        img_copy[img_copy == 100] = -1
        img_copy[img_copy == 255] = 1

        # image = img / 255
        image = trans.resize(img_copy, target_size)
        image = np.reshape(image, image.shape + (1,)) if (not flag_multi_class) else image
        image = np.reshape(image, (1,) + image.shape)
        yield image


def saveResult(save_path, npyfile, flag_multi_class=False, num_class=2):
    for i, item in enumerate(npyfile):
        img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        # io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img_as_ubyte(img))
        return img_as_ubyte(img)


def labelVisualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


def convert_angle_to_polar(angle):
    if angle <= 90:
        return 90 - angle
    else:
        return 360 - (angle - 90)


def draw_lines(img, size_anles_coordinates, color=[100, 100, 100], thickness=1, make_copy=True):
    # Copy the passed image
    img_copy = np.copy(img) if make_copy else img
    h,w = img_copy.shape[:2]
    img_copy = cv2.resize(img_copy, (int(w/16), int(h/16)))

    cv2.imshow('INIT_IMAGE', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for lines in size_anles_coordinates:
        rho = math.floor(lines[0] / 2)
        theta = convert_angle_to_polar(lines[1])
        for coord in lines[2]:
            phi = theta * math.pi / 180.0
            y = int(round(rho * np.sin(phi)))
            x = int(round(rho * np.cos(phi)))
            y1 = (coord[0] - y)
            x1 = (coord[1] + x)
            y2 = (coord[0] + y)
            x2 = (coord[1] - x)
            cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)

    cv2.imshow('IMAGE_WITH_LINES', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img_copy
