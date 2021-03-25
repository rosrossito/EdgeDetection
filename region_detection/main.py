import argparse

from playing.rainbow import rainbow
from region_detection.layers.first_layer import first_layer_convolution, first_layer_with_slope_convolution
from region_detection.layers.second_layer import second_layer_convolution

from region_detection.util import load_Image


def run_detection():
    gray_image = load_Image(args["image"])
    rainbow(gray_image)
    # output_layer1 = first_layer_convolution(gray_image)
    # output_layer1 = first_layer_with_slope_convolution(gray_image)
    # output_layer2 = second_layer_convolution(output_layer1)
    # output_layer2 = second_layer_convolution(gray_image)

    print("the end")


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
args = vars(ap.parse_args())
run_detection()
