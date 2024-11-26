import os
import sys
import csv
import numpy as np
import scipy
import h5py as h5
import tensorflow as tf
from tensorflow import keras

from PIL import Image
import pandas as pd


class Validation:
    def __init__(self,val_data):
        self.val_data= val_data

    def l_c_s(self,im1, im2):
        sigma = 1.5
        truncate = 3.5
        k1 = 0.01
        k2 = 0.03

        r = int(truncate * sigma + 0.5)  # radius as in ndimage
        win_size = 2 * r + 1

        data_range = 1.0
        filter_func = scipy.ndimage.gaussian_filter
        filter_args = {'sigma': sigma, 'truncate': truncate, 'mode': 'reflect'}

        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

        # Compute (weighted) means
        ux = filter_func(im1, **filter_args)
        uy = filter_func(im2, **filter_args)

        # Compute (weighted) variances and covariances
        uxx = filter_func(im1 * im1, **filter_args)
        uyy = filter_func(im2 * im2, **filter_args)
        uxy = filter_func(im1 * im2, **filter_args)

        vx = cov_norm * (uxx - ux * ux)
        vy = cov_norm * (uyy - uy * uy)
        vxy = cov_norm * (uxy - ux * uy)

        vx = np.clip(vx, a_min=0, a_max=100)
        vy = np.clip(vy, a_min=0, a_max=100)
        vxy = np.clip(vxy, a_min=0, a_max=100)

        r = data_range
        c1 = (k1 * r) ** 2
        c2 = (k2 * r) ** 2
        c3 = c2 / 2

        a1, a2, b1, b2 = (
            2 * ux * uy + c1,
            2 * np.sqrt(vx) * np.sqrt(vy) + c2,
            ux**2 + uy**2 + c1,
            vx + vy + c2,
        )

        lum = a1 / b1
        con = a2 / b2
        struc = (vxy + c3) / (np.sqrt(vx) * np.sqrt(vy) + c3)

        pad = (win_size - 1) // 2
        lum = lum[pad:-pad, pad:-pad].mean(dtype=np.float64)
        con = con[pad:-pad, pad:-pad].mean(dtype=np.float64)
        struc = struc[pad:-pad, pad:-pad].mean(dtype=np.float64)

        ssim = lum * con * struc
        return lum, con, struc, ssim


    def gini(self,data):
        flattened = np.sort(np.ravel(data))
        npix = np.size(flattened)
        normalization = np.abs(np.mean(flattened)) * npix * (npix - 1)
        kernel = (2.0 * np.arange(1, npix + 1) - npix - 1) * np.abs(flattened)

        return np.sum(kernel) / normalization


    def norm_0_1(self,arr):
        return (arr + 1.0) / 2.0


    def get_stats(self, model):
        lum, cont, struc, ssim, mae = [], [], [], [], []
        gini_orig, gini_new = [], []

        for k in range(self.val_data.shape[0]):
            inp = self.val_data[k : k + 1,0:2]
            target = self.val_data[k : k + 1, -1]

            fake = model(inp).numpy()

            for i, img in enumerate(inp):
                inp[i] = self.norm_0_1(img)

            target = self.norm_0_1(target)
            fake = self.norm_0_1(fake)

            l, c, s, ss = self.l_c_s(target[0, :, :, 0], fake[0, :, :, 0])

            lum.append(l)
            cont.append(c)
            struc.append(s)
            ssim.append(ss)
            mae.append(np.mean(np.abs(target - fake)))
            gini_orig.append(self.gini(target[0, :, :, 0]))
            gini_new.append(self.gini(fake[0, :, :, 0]))

        lum = round(np.mean(lum), 4)
        cont = round(np.mean(cont), 4)
        struc = round(np.mean(struc), 4)
        ssim = round(np.mean(ssim), 4)
        mae = round(np.mean(mae), 4)
        gini_orig = round(np.mean(gini_orig), 4)
        gini_new = round(np.mean(gini_new), 4)

        return lum, cont, struc, ssim, mae, gini_orig, gini_new
