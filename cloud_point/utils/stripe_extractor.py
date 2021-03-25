from cloud_point.utils.grass_extractor import GrassExtractor
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture


class StripeExtractor:
    is_filtered = True

    def __init__(self, load_init=True, is_filtered=True, filter_size=10):
        self.gex = GrassExtractor(load_init=load_init, is_filtered=True,
                                  filter_size=10, n_components=3)
        self.filter_size = filter_size

    def preprocess(self, imgs):
        new_imgs = []
        for img in imgs:
            res = img / img.mean(axis=(1, 2), keepdims=True)
            test_img = (res / res.max()) ** (0.5 ** 0.5)

            res = test_img / test_img.mean(axis=(0, 2), keepdims=True)
            test_img = ((res / res.max()) ** (0.5 ** 0.5))

            new_imgs.append((test_img * 255).astype('uint8'))
        return new_imgs

    def fit(self, imgs):
        self.gex.fit(self.preprocess(imgs))

    def predict(self, img):
        img = self.preprocess([img])[0]

        masks, grass_mask, skeleton = self.gex.predict(img)

        gm = GaussianMixture(n_components=2)

        grass_ind = (grass_mask / 255).astype('bool')
        sample = img[grass_ind]
        gm.fit(sample)
        pred = gm.predict(img.reshape(-1, 3))
        pred_img = pred.reshape(img.shape[:2])

        masks[(1 - grass_mask).astype('bool')] = 0
        stripes = masks
        stripes[grass_ind] = pred_img[grass_ind] + self.gex.n_components

        return grass_mask, self.filt(stripes) if self.is_filtered else (grass_mask, stripes)

    def filt(self, stripes):
        filtere_image = np.zeros(stripes.shape)
        for i in sorted(list(set(stripes.flatten())))[1:]:
            kernel = np.ones((self.filter_size, self.filter_size), np.uint8)
            res = cv2.morphologyEx(((stripes == i) * 255).astype('uint8'),
                                   cv2.MORPH_OPEN, kernel)

            inds = res > 0
            filtere_image[inds] = i

        return filtere_image

# se = StripeExtractor()
# se.fit(imgs[:10])
# masks = se.predict(imgs[1])
