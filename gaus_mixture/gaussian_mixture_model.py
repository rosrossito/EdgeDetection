from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
import cv2
import numpy as np

from cloud_point.utils.rein_extractors import ReinGrassExtractor
from cloud_point.utils.stripe_extractor import StripeExtractor


def get_gaussian_mixture(img):
    img2 = img.reshape(-1, img.shape[-1])
    gmm = GaussianMixture(n_components=5, covariance_type="full")
    gmm = gmm.fit(img2)
    clusters = gmm.predict(img2)
    segmented = clusters.reshape(img.shape[:-1])
    segmented_show = np.transpose(np.array([segmented] * 3), (1, 2, 0))
    segmented_show[segmented_show == 1] = 60
    segmented_show[segmented_show == 2] = 120
    segmented_show[segmented_show == 3] = 150
    segmented_show[segmented_show == 4] = 180
    segmented_show[segmented_show == 5] = 255
    segmented_image = segmented_show.astype("uint8")
    cv2.imshow("mixture", segmented_image)

    # blur
    # smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
    # largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
    # gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    # opencvOutput = cv2.filter2D(gray, -1, largeBlur)
    # cv2.imshow("blurred", opencvOutput )

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # plt.imshow(segmented)
    # plt.show()

    return cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)


def get_vitalii_gaussian_mixture(img):
    # rein_clu = ReinGrassExtractor(3, 3)
    # return rein_clu.fit_predict(img, 1, slice_factor=1)
    se = StripeExtractor(load_init=False)
    se.fit([img])
    # se.predict(img)

    # grass_img = (grass_img / (rein_clu.n_components) * 255).astype('uint8')
    # grass_img = np.array([grass_img] * 3).transpose(1, 2, 0)
    #
    # cv2.imshow('Frame', grass_img)

    grass_mask, gray_image = se.predict(img)
    gray_image[gray_image == 3] = 100
    gray_image[gray_image == 4] = 255

    grass_mask[grass_mask == 1] = 255
    # cv2.imshow('grass_mask', grass_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # cv2.imshow('gray_image', gray_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # get_upper_bound(img, grass_mask)

    return grass_mask, gray_image

#
# def get_upper_bound(image, grass_mask, base_name=None, do_debug=False):
#     # X, success = tools_IO.load_if_exists(self.folder_out + 'cache/' + base_name + '_line_up.dat')
#     # if (not do_debug) and success:
#     #     return X
#
#     hit = 255 - filter_ver(255 * grass_mask, 4, 13)
#     hit[int(0.5 * image.shape[0]):, :] = 0
#     line = hitmap_to_line(hit, min_len=100, do_ransac=True, base_name=base_name, do_debug=do_debug)
#
#     if (line is not None) and np.linalg.norm(line[:2] - line[2:]) < image.shape[1] / 4:
#         line = None
#
#     if line is not None:
#         line = HE.boxify_lines([line], (0, 0, image.shape[1], image.shape[0]))[0]
#
#     # tools_IO.write_cache(self.folder_out + 'cache/' + base_name + '_line_up.dat', line)
#     #
#     # if do_debug:
#     #     # cv2.imwrite(self.folder_out+'upper_filter.png',hit)
#     #     image_debug = tools_image.desaturate(64 * grass_mask)
#     #     image_debug = tools_draw_numpy.draw_lines(image_debug, [line], color=(0, 128, 255), w=1)
#     #     cv2.imwrite(self.folder_out + base_name + '_bounds.png', image_debug)
#
#     return line
#
#
# def filter_ver(self, gray2d, sobel_H, sobel_W, skip_agg=False):
#     sobel = np.full((sobel_H, sobel_W), +1, dtype=np.float32)
#     sobel[sobel.shape[0] // 2:, :] = +1
#     sobel[:sobel.shape[0] // 2, :] = -1
#     if sobel.sum() > 0:
#         sobel = sobel / sobel.sum()
#     filtered = cv2.filter2D(gray2d, 0, sobel)
#
#     if skip_agg:
#         return filtered
#
#     agg = tools_image.sliding_2d(filtered, -(sobel_H // 4), +(sobel_H // 4), -sobel_W, +sobel_W)
#     neg = numpy.roll(agg, -sobel_H // 4, axis=0)
#     pos = numpy.roll(agg, +sobel_H // 4, axis=0)
#     hit = ((255 - neg) + pos) / 2
#     hit[:sobel_H, :] = 128
#     hit[-sobel_H:, :] = 128
#
#
# return numpy.array(hit, dtype=numpy.uint8)
#
