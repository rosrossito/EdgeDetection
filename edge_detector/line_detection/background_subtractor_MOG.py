import cv2

def substract(img):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgmask = fgbg.apply(img)
    # cv2.imshow('img', fgmask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return fgmask
