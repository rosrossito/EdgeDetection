from collections import Counter
from sklearn.mixture import GaussianMixture

import cv2
import numpy as np
import os
import pickle

# Example
# gex = GrassExtractor(is_filtered=True, filter_size=10)
# masks, grass_mask, skeleton = gex.predict(img)

class GrassExtractor():
    
    clu = None
    fitted = False
    initialized = False
    
    def __init__(self, load_init=True, is_filtered=True, filter_size=10, n_components=4):
        self.is_filtered = is_filtered
        self.filter_size = filter_size
        self.n_components = n_components
        if load_init:
            self.initialized = load_init
            with open('../cloud_point/utils/grass_extractor_init_model_3.pkl', 'rb') as f:
                self.clu = pickle.load(f)
    
    def get_edges(self, mask, k):
        grass_mask = mask == k
        grass_mask = (grass_mask * 255).astype('uint8')
        
        kernel = np.ones((self.filter_size, self.filter_size), np.uint8)
        
        if self.is_filtered:
            res = cv2.morphologyEx(grass_mask, cv2.MORPH_OPEN, kernel)
        else:
            res = grass_mask
            
        sv = cv2.Sobel(res, 3, 0, 1, ksize=3)
        sv[sv < 0] = 0
        sv = (sv / sv.max() * 255).astype('uint8')

        sh = cv2.Sobel(res, 3, 1, 0, ksize=3)
        sh[sh < 0] = 0
        sh = (sh / sh.max() * 255).astype('uint8')

        res_skeleton = (sh / 2 + sv / 2) > 0

        return res_skeleton, grass_mask
    
    def fit(self, images):
        
        sample = []
        for img in images:
            points_x = np.random.randint(0, img.shape[0], (500, 1))
            points_y = np.random.randint(0, img.shape[1], (500, 1))
            points = np.append(points_x, points_y, axis=1)
            sample += [img[p[0], p[1]] for p in points]

        X = np.array(sample) 
        
        self.clu = GaussianMixture(n_components=self.n_components).fit(X)
        
    def predict_one(self, img):
        if not self.initialized:
            assert not self.fitted, 'Please fit the model or initialize it'
        
        img_X_all = img.reshape(-1, img.shape[-1])
        clusters = self.clu.predict(img_X_all)

        skeleton = clusters.reshape(img.shape[:-1])

        count = Counter(skeleton.flatten())
        counters = np.zeros(len(count))
        for i in count:
            counters[i] = count[i]
        masks = clusters.reshape(img.shape[:-1])

        skeleton, grass_mask = self.get_edges(masks, counters.argmax())
        
        return masks, grass_mask, skeleton
        
    def predict(self, imgs):       
        if isinstance(imgs, np.ndarray) and imgs.shape[0] == 1:
            return self.predict_one(imgs)
        elif isinstance(imgs, np.ndarray) and imgs.shape[-1] == 3 and len(imgs.shape) == 3:
            return self.predict_one(imgs)
        elif isinstance(imgs, np.ndarray) and len(imgs.shape) == 4 or isinstance(imgs, list):
            return [self.predict_one(img) for img in imgs]
        else:
            assert False, "Wrong num of channels. Img have a RGB format"
