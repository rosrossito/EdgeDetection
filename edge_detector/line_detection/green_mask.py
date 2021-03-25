import cv2
import numpy as np
import skimage.measure

from line_detection.background_subtractor_MOG import substract


def find_grass(img):
    ## convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ## mask of green (36,25,25) ~ (86, 255,255)
    # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
    mask = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))

    ## slice the green
    imask = mask > 0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]

    ## save
    # cv2.imwrite("green.png", green)
    green = cv2.cvtColor(green, cv2.COLOR_HSV2BGR)
    green = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("green", green)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return green


def get_edges_for_grass(img):
    img_grass = find_grass(img)
    output = substract(img_grass)
    output = cv2.Canny(output.astype("uint8"), 100, 200)

    # maxpool
    output = skimage.measure.block_reduce(output, (2, 2), np.max)
    output = skimage.measure.block_reduce(output, (2, 2), np.max)
    output = skimage.measure.block_reduce(output, (2, 2), np.max)

    # cv2.imshow("green", output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return output



def get_edges_for_grass_without_pooling(img):
    img_grass = find_grass(img)
    output = substract(img_grass)
    output = cv2.Canny(output.astype("uint8"), 100, 200)

    cv2.imshow("green", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output

