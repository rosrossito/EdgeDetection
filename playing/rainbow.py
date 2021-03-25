import cv2
import numpy as np


def rainbow(image):
    k = 125
    (iH, iW) = image.shape[:2]
    output = np.zeros((iH, iW), dtype="float32")
    for y in np.arange(1, iH):
        for x in np.arange(1, iW):
    #         variance = np.var(np.array([[image[y-1, x-1], image[y, x - 1]], [image[y-1, x], image[y, x]]]))
    #         if variance>1000:
    #             output[y, x] = variance
    #         else:
    #             output[y, x] = 0


            if y==10:
                output[y, x] = 50
            elif y==11:
                output[y, x] = 100
            elif y == 20:
                output[y, x] = 50
            elif y == 21:
                output[y, x] = 50
            elif y == 30:
                output[y, x] = 100
            elif y == 31:
                output[y, x] = 100
            elif y == 40:
                output[y, x] = 250
            elif y == 50:
                output[y, x] = 100
            elif y == 60:
                output[y, x] = 50
            elif y == 70:
                output[y, x] = 30
            elif y == 80:
                output[y, x] = 30
            elif y == 81:
                output[y, x] = 30
            elif y == 82:
                output[y, x] = 30
            elif y == 83:
                output[y, x] = 30
            elif y == 84:
                output[y, x] = 30
            elif y == 85:
                output[y, x] = 30
            elif y == 86:
                output[y, x] = 30


        # if image[y, x] < k:
            #     output[y, x] = 250
            # elif image[y, x] < 2 * k:
            #     output[y, x] = 0
            # elif image[y, x] < 2 * k + k / 2:
            #     output[y, x] = 2 * k
            # elif image[y, x] < 3 * k:
            #     output[y, x] = 250
            # else:
            #     output[y, x] = 200

    # output_convolved = np.zeros((iH, iW), dtype="float32")
    #
    # for y in np.arange(0, iH, 2):
    #     for x in np.arange(0, iW, 2):
    #         if output[y, x] == 0 and output[y, x + 1] == 0:
    #             output_convolved[y, x] = 0
    #         elif output[y, x] != 0 and output[y, x + 1] != 0:
    #             output_convolved[y, x] = 0





    output = output.astype("uint8")
    cv2.imshow("rainbow", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output
