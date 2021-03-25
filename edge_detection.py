from skimage.color import rgb2gray
import numpy as np
# import cv2
import matplotlib.pyplot as plt
# %matplotlib inline
from scipy import ndimage

# load image
image = plt.imread('1.jpeg')
plt.imshow(image)
plt.show()

# # REGION-BASED SEGMENTATION

# # convert image to grey
# gray = rgb2gray(image)
# plt.imshow(gray, cmap='gray')
# plt.show()


# # image shape
# print(gray.shape)
#
# # background vs foreground
# total_pixels = gray.reshape(gray.shape[0]*gray.shape[1])
# for i in range(total_pixels.shape[0]):
#     if total_pixels[i] > total_pixels.mean():
#         total_pixels[i] = 1
#     else:
#         total_pixels[i] = 0
# gray = total_pixels.reshape(gray.shape[0],gray.shape[1])
# plt.imshow(gray, cmap='gray')
# plt.show()
#
#
# # several treashold to identify different regions
# gray = rgb2gray(image)
# gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
# for i in range(gray_r.shape[0]):
#     if gray_r[i] > gray_r.mean():
#         gray_r[i] = 3
#     elif gray_r[i] > 0.5:
#         gray_r[i] = 2
#     elif gray_r[i] > 0.25:
#         gray_r[i] = 1
#     else:
#         gray_r[i] = 0
# gray = gray_r.reshape(gray.shape[0],gray.shape[1])
# plt.imshow(gray, cmap='gray')
# plt.show()


# EDGE DETECTION SEGMENTATION
# converting to grayscale
# image = plt.imread('index.png')
gray = rgb2gray(image)

# defining the sobel filters
# sobel_horizontal = np.array([np.array([1, 2, 1]), np.array([0, 0, 0]), np.array([-1, -2, -1])])
# print(sobel_horizontal, 'is a kernel for detecting horizontal edges')
#
# sobel_vertical = np.array([np.array([-1, 0, 1]), np.array([-2, 0, 2]), np.array([-1, 0, 1])])
# print(sobel_vertical, 'is a kernel for detecting vertical edges')
#
# out_h = ndimage.convolve(gray, sobel_horizontal, mode='reflect')
# out_v = ndimage.convolve(gray, sobel_vertical, mode='reflect')
# plt.imshow(out_h, cmap='gray')
# plt.show()
# plt.imshow(out_v, cmap='gray')
# plt.show()

print(gray.shape)

seven_horizontal = np.array([np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])])
out_seven_h = ndimage.convolve(gray, seven_horizontal, mode='reflect')
plt.imshow(out_seven_h, cmap='gray')
plt.show()


