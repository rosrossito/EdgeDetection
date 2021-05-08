import cv2

def save_image(image, base_path, ext='jpg'):
    cv2.imwrite('{}.{}'.format(base_path, ext), image)

def load_Image(image):
    # load the input image and convert it to grayscale
    gray_image = cv2.imread(image)
    return cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
