import argparse

import cv2

from canny_detector import utils
from canny_detector.canny_edge_detector import cannyEdgeDetector


def run_detection():
    # gray_image = load_Image(args["image"])
    # res, weak, strong = threshold(gray_image)

    imgs = utils.load_data()

    detector = cannyEdgeDetector(imgs, sigma=1.4, kernel_size=5, lowthreshold=0.2, highthreshold=0.10,
                                     weak_pixel=100)
    img_final = detector.detect()
    output = img_final.astype("uint8")
    cv2.imshow("rainbow", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("the end")


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
                help="path to the input image")
args = vars(ap.parse_args())
run_detection()
