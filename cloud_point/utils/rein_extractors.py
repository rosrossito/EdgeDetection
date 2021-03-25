from collections import Counter
from matplotlib import pyplot as plt
from numpy import random as npr
from tqdm import tqdm
from scipy.stats import multivariate_normal

import cv2
import numpy as np
import time

def get_feature(img, size=1000):
    return img[np.random.randint(0, img.shape[0], size), 
               np.random.randint(0, img.shape[1], size)]

import theano
import theano.tensor as T

def est_mult_gaus(X, mu, sigma):
    Xm = X-mu
    m = X.shape[1]
    p = 1 / ((2 * np.pi) ** m * theano.tensor.nlinalg.det(sigma)) ** 0.5
    
    p *= np.exp(-0.5 * ((Xm.dot(theano.tensor.nlinalg.pinv(sigma))) * Xm).sum(axis=1))

    return p


x_t = T.matrix('float16')
mu_t = T.vector('float16')
sig_t = T.matrix('float16')
y_t = est_mult_gaus(x_t, mu_t, sig_t)
fmixt = theano.function([x_t, mu_t, sig_t], y_t)

def E_step(xx, theta, proba=True):
    n_components = len(theta['weight'])
    p2 = []
    for k in range(n_components):
        p2.append(fmixt(xx, theta['means'][k], theta['vars'][k]) * theta['weight'][k])
    p2 = np.array(p2).T
    
    if proba:
        p2 = (p2 / p2.sum(axis=1, keepdims=True))
        p2[np.isnan(p2)] = 1 / n_components
    return p2

def E_step_orig(x, theta):
    probas = []
    n_components = len(theta['weight'])
    for k in range(n_components):
        x_pdf = multivariate_normal.pdf(x, 
                                      theta['means'][k], 
                                      theta['vars'][k]) * theta['weight'][k]


        probas.append(x_pdf)

    probas = np.array(probas)
    probas = (probas / probas.sum(axis=0, keepdims=True)).T
    probas[np.isnan(probas)] = 1 / n_components
    
    return probas

def M_step(x, theta, reg_covar, dim):
    n_components = len(theta['weight'])
    
    probas = E_step(x, theta)
    new_means = x.T.dot(probas) / probas.sum(axis=0, keepdims=True)
    new_weights = probas.sum(axis=0) / probas.sum()

    new_vars = []
    for k in range(n_components):
        xs = np.zeros((x.shape[0], dim, dim))
        x_mu = (x - new_means[:, k]).T
        for ik, iline in enumerate(x_mu):
            for jk, jline in enumerate(x_mu):
                xs[:, ik, jk] = iline * jline
        norma = probas.T[k].sum()
        
        covar = (probas.T[k][:, None, None] * xs / norma).sum(axis=0)
        covar += np.diag(np.ones(dim)) * reg_covar
        new_vars.append(covar)

    new_vars = np.array(new_vars)
    
    return {
        'means': new_means.T,
        'weight': new_weights,
        'vars': new_vars,
    }

def slide_param(tmp_theta, new_theta, alpha):
    params = {}
    for key in tmp_theta:
        params[key] = tmp_theta[key] * alpha + (1 - alpha) * new_theta[key]
    return params

def prepare_img_origin(img):
    res = img / img.mean(axis=(1, 2), keepdims=True)
    res = res / res.mean(axis=(0, 2), keepdims=True)

    test_img = np.sqrt(res / res.max())

    return (test_img * 255).astype('uint8')

def prepare_img_t(img):
    res = img / img.mean(axis=(1, 2), keepdims=True)
    res = res / res.mean(axis=(0, 2), keepdims=True)

    test_img = np.sqrt(res / res.max())

    return (test_img * 255)

img_t = T.tensor3('float64')
new_img_t = prepare_img_t(img_t)
prepare_img_func = theano.function([img_t], new_img_t)

def prepare_img(img):
    return prepare_img_func(img).astype('uint8')

class ReinGMM:
    def __init__(self, params, main_alpha=0.025, reg_covar=1e-7):
        self.theta = params
        self.alpha = main_alpha
        self.reg_covar = reg_covar
        self.history = []
        self.n_components = len(self.theta['weight'])
        
    def fit(self, x, n_frame, iters=100):
        if n_frame < 5:
            for i in range(iters):
                self.theta = M_step(x, self.theta, self.reg_covar, x.shape[1])
        else:
            tmp_theta = M_step(x, self.theta, self.reg_covar, x.shape[1])
            if n_frame < 20:
                self.theta = slide_param(tmp_theta, self.theta, self.alpha)
            else:
                self.theta = slide_param(tmp_theta, self.theta, 0.8)
        
        self.history.append(self.theta)
        
    def predict(self, x):
        return E_step(x, self.theta, proba=False)

def predict_one(clu, img):
    img_X_all = img.reshape(-1, img.shape[-1])
    clusters = clu.predict(img_X_all).argmax(axis=1)

    count = Counter(clusters[::clusters.shape[0] // 100])

    counters = np.zeros(clu.n_components)
    for i in count:
        counters[i] = count[i]
    masks = clusters.reshape(img.shape[:-1])
    
    return masks, masks == counters.argmax()

class ReinGrassExtractor:
    def __init__(self, n_components, dim):
        self.n_components = n_components
        self.dim = dim

        self.theta = {'means': npr.randint(0, 255, size=(n_components, self.dim)),
                 'vars': np.array([np.diag(np.ones(self.dim) * 1000)] * self.n_components),
                 'weight': npr.rand(self.n_components)}

        self.theta['weight'] /= sum(self.theta['weight'])
    
        self.clu = ReinGMM(self.theta, reg_covar=1)
        
    def fit_predict(self, img, kimg, slice_factor=1):
        if slice_factor > 1:
            cut_img = img[::slice_factor, ::slice_factor]
        else:
            cut_img = img
            
        cut_img = prepare_img(cut_img)
        
        x = get_feature(cut_img)
        self.clu.fit(x, kimg, iters=100)
        grass_img, grass_inds = predict_one(self.clu, cut_img)
        return grass_img, grass_inds

class ReinStripeExtractor:
    def __init__(self, n_components, dim):
        self.n_components = n_components
        self.dim = dim

        self.theta = {'means': npr.randint(0, 255, size=(n_components, self.dim)),
                 'vars': np.array([np.diag(np.ones(self.dim) * 1000)] * self.n_components),
                 'weight': npr.rand(self.n_components)}

        self.theta['weight'] /= sum(self.theta['weight'])
        
        self.theta_stripes = {'means': [[80, 80, 80], [120, 120, 120]],
             'vars': np.array([np.diag(np.ones(dim) * 255)] * 2),
             'weight': npr.rand(2)}

        self.theta_stripes['weight'] /= sum(self.theta_stripes['weight'])
        
    
        self.clu = ReinGMM(self.theta, reg_covar=1)
        self.stripes_clu = ReinGMM(self.theta_stripes, 0.1, reg_covar=1)
        
    def fit_predict(self, img, kimg, slice_factor=1):
        cut_img = img[::slice_factor, ::slice_factor]
        cut_img = prepare_img(cut_img)
    
        
        x = get_feature(cut_img)
        self.clu.fit(x, kimg, iters=100)
        grass_img, grass_inds = predict_one(self.clu, cut_img)
        
        grass_pixels = cut_img[grass_inds]
        sample_inds = npr.choice(grass_pixels.shape[0], size=1000)
        self.stripes_clu.fit(grass_pixels[sample_inds], kimg, iters=20)

        clusters_s, _ = predict_one(self.stripes_clu, cut_img)

        stripes = np.zeros(grass_inds.shape)
        stripes[grass_inds] = clusters_s[grass_inds] + self.clu.n_components
        
        return stripes, None

