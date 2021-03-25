import cv2
import numpy as numpy

PLAYER_SIZE_THRESHOLD = 100


def erase(img, positions):
    left_side_b = []
    left_side_g = []
    left_side_r = []
    right_side_b = []
    right_side_g = []
    right_side_r = []

    for pos in positions:
        pos = pos.astype(numpy.float)
        x1, y1, x2, y2 = pos
        x1, y1, x2, y2 = int(round(float(x1))), int(round(float(y1))), int(round(float(x2))), int(round(float(y2)))
        if y1 < 720 and y2 < 720 and (y2 - y1) < PLAYER_SIZE_THRESHOLD and (x2 - x1) < PLAYER_SIZE_THRESHOLD:
            for l in range(x1-10, round(x1 + (x2-x1)/2)):
                img[y1-10:y2+10,l,:] = img[y1-10:y2+10,x1 - 10,:]
            for r in range(x1 + round((x2-x1)/2), x2 + 10):
                img[y1-10:y2+10, r, :] = img[y1-10:y2+10, x2 + 10, :]


            # for l in range(y1, y2):
            #     left_side_b.append(img[l, x1, 0])
            #     left_side_g.append(img[l, x1, 1])
            #     left_side_r.append(img[l, x1, 2])
            #     right_side_b.append(img[l, x2, 0])
            #     right_side_g.append(img[l, x2, 1])
            #     right_side_r.append(img[l, x2, 2])
            #
            # values_left_b = numpy.average(left_side_b)
            # values_left_g = numpy.average(left_side_g)
            # values_left_r = numpy.average(left_side_r)
            # values_right_b = numpy.average(right_side_b)
            # values_right_g = numpy.average(right_side_g)
            # values_right_r = numpy.average(right_side_r)
            #
            # cv2.rectangle(img, (x1, y1), (x1 + round((x2 - x1) / 2), y2),
            #               (int(round(values_left_b)),
            #                int(round(values_left_g)),
            #                int(round(values_left_r))), -1)
            # cv2.rectangle(img, (x1 + round((x2 - x1) / 2), y1), (x2, y2),
            #               (int(round(values_right_b)),
            #                int(round(values_right_g)),
            #                int(round(values_right_r))), -1)

            # (values_left_b, counts_left_b) = numpy.unique(left_side_b, return_counts=True)
            # (values_left_g, counts_left_g) = numpy.unique(left_side_g, return_counts=True)
            # (values_left_r, counts_left_r) = numpy.unique(left_side_r, return_counts=True)
            # (values_right_b, counts_right_b) = numpy.unique(right_side_b, return_counts=True)
            # (values_right_g, counts_right_g) = numpy.unique(right_side_g, return_counts=True)
            # (values_right_r, counts_right_r) = numpy.unique(right_side_r, return_counts=True)
            #
            #
            # cv2.rectangle(img, (x1, y1), (x1 + round((x2 - x1) / 2), y2),
            #           (int(values_left_b[numpy.where(counts_left_b == max(counts_left_b))[0][0]]),
            #            int(values_left_g[numpy.where(counts_left_g == max(counts_left_g))[0][0]]),
            #            int(values_left_r[numpy.where(counts_left_r == max(counts_left_r))[0][0]])), -1)
            # cv2.rectangle(img, (x1 + round((x2 - x1) / 2), y1), (x2, y2),
            #           (int(values_right_b[numpy.where(counts_right_b == max(counts_right_b))[0][0]]),
            #            int(values_right_g[numpy.where(counts_right_g == max(counts_right_g))[0][0]]),
            #            int(values_right_r[numpy.where(counts_right_r == max(counts_right_r))[0][0]])), -1)

    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img
