import math
import os

import numpy as np
import skimage.io as io

from rect_lookup.data import testGeneratorOneImg, draw_lines, saveResult
from rect_lookup.model import rectnet
from rect_lookup.model_pic import rectnet1

THRESHOLD_RECT = 100
ANGLES = 180


def convert_indexes_to_coordinates(results, minimal_kernel_size):
    size_anles_coordinates = []
    rH, rW, rK = results.shape[1:4]

    reshaped_results_sum = np.reshape(results, (rK, rH * rW)).sum(axis=1)
    indexes = np.where(reshaped_results_sum > 0)[0]
    reshaped_results = np.reshape(results, (rK, rH, rW))
    for index in indexes:
        size = minimal_kernel_size + math.floor(index / ANGLES) * 2
        angle = index % ANGLES
        layer = reshaped_results[index, :, :]
        coord = np.where(layer == np.amax(layer))
        listOfCordinates = list(zip(coord[0], coord[1]))
        size_anles_coordinates.append((size, angle, listOfCordinates))
    return size_anles_coordinates


def find_rectangle(initial_image, img):
    (iH, iW) = img.shape[:2]

    model1 = rectnet1(input_size=(iH, iW, 1))

    model, act4, minimal_kernel_size = rectnet(input_size=(iH, iW, 1))
    testGene = testGeneratorOneImg(img, target_size=(iH, iW))

    # results1 = model1.predict_generator(testGene, 1, verbose=1)
    # saveResult("rect_lookup/test", results1)

    results = model.predict_generator(testGene, 1, verbose=1)
    size_anles_coordinates = convert_indexes_to_coordinates(results, minimal_kernel_size)

    img_with_lines = draw_lines(initial_image, size_anles_coordinates)
    io.imsave(os.path.join("rect_lookup/test", "%d_predict.png" % 1), img_with_lines)

    # find_rect(rect, rect)
