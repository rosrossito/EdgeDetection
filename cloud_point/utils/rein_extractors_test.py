#!/usr/bin/env python
# coding: utf-8

# In[1]:
import tqdm as tqdm

from cloud_point.utils.rein_extractors import *


# In[2]:


cap = cv2.VideoCapture('soccer_ex06.avi')
imgs = []
while True:
    status, img  = cap.read()
    if status:
        imgs.append(img)
    if not status:
        break
cap.release()


# In[3]:


# clu = ReinGMM(theta, reg_covar=1)
rein_clu = ReinGrassExtractor(3, 3)

t = tqdm(enumerate(imgs))

for kimg, img in t:
    grass_img, grass_inds = rein_clu.fit_predict(img, kimg, slice_factor=1)

    
    grass_img = (grass_img / (rein_clu.n_components) * 255).astype('uint8')
    grass_img = np.array([grass_img] * 3).transpose(1, 2, 0)
    
    cv2.imshow('Frame', grass_img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
t.close()
cv2.destroyAllWindows()


# In[4]:


# clu = ReinGMM(theta, reg_covar=1)
rein_clu = ReinStripeExtractor(3, 3)

t = tqdm(enumerate(imgs))
for kimg, img in t:
    grass_img, grass_inds = rein_clu.fit_predict(img, kimg, slice_factor=1)

    
    grass_img = (grass_img / (rein_clu.n_components) * 255).astype('uint8')
    grass_img = np.array([grass_img] * 3).transpose(1, 2, 0)
    
    cv2.imshow('Frame', grass_img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
t.close()
cv2.destroyAllWindows()

