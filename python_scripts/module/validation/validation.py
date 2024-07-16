import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import RMSprop
from keras import layers

from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.layers import InputSpec
from keras.layers import Layer
from keras.layers import GroupNormalization as GNorm

import sys

import os
import numpy as np
import csv
import h5py as h5

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import image
from matplotlib.cbook import get_sample_data
from PIL import Image

import pandas as pd
from random import randint
from numpy import random
import scipy


def l_c_s(im1,im2):

    sigma=1.5
    truncate=3.5
    K1=0.01
    K2=0.03

    r = int(truncate * sigma + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1

    data_range = 1.0

    filter_func = scipy.ndimage.gaussian_filter
    filter_args = {'sigma': sigma, 'truncate': truncate, 'mode': 'reflect'}

    cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(im1, **filter_args)
    uy = filter_func(im2, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(im1 * im1, **filter_args)
    uyy = filter_func(im2 * im2, **filter_args)
    uxy = filter_func(im1 * im2, **filter_args)
    
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    
    vx= np.clip(vx,a_min=0,a_max=100)
    vy= np.clip(vy,a_min=0,a_max=100)
    vxy= np.clip(vxy,a_min=0,a_max=100)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    C3=C2/2

    A1, A2, B1, B2 = (
        2 * ux * uy + C1,
        2 * np.sqrt(vx)*np.sqrt(vy) + C2,
        ux**2 + uy**2 + C1,
        vx + vy + C2,
    )

    lum=A1/B1
    con= A2/B2
    struc= (vxy+C3)/(np.sqrt(vx)*np.sqrt(vy)+C3)


    pad = (win_size - 1) // 2

    lum= lum[pad:-pad,pad:-pad].mean(dtype=np.float64)
    con= con[pad:-pad,pad:-pad].mean(dtype=np.float64)
    struc= struc[pad:-pad,pad:-pad].mean(dtype=np.float64)

    ssim= lum*con*struc
    return lum,con,struc,ssim


def gini(data):
    flattened = np.sort(np.ravel(data))
    npix = np.size(flattened)
    normalization = np.abs(np.mean(flattened)) * npix * (npix - 1)
    kernel = (2.0 * np.arange(1, npix + 1) - npix - 1) * np.abs(flattened)

    return np.sum(kernel) / normalization


def nomr_0_1(arr):
    return (arr+1.0)/2.0



def per_ckpt(model):  
    lum=[]
    cont=[]
    struc=[]
    ssim=[]
    mae=[]
    gini_orig=[]
    gini_new=[]

    for k in range(val_inp.shape[0]):
        inp =  val_inp[k:k+1]
        target= val_label[k:k+1]
        
        fake= model(inp).numpy()
        
        i=0
        for img in inp:
            inp[i]= nomr_0_1(img)
            i=i+1
        
        target= nomr_0_1(target)
        fake = nomr_0_1(fake)
         
        l,c,s,ss = l_c_s(target[0,:,:,0],fake[0,:,:,0])
        
        lum.append(l)
        cont.append(c)
        struc.append(s)
        ssim.append(ss)
        mae.append(np.mean(np.abs( target- fake )))
        gini_orig.append(gini(target[0,:,:,0]))
        gini_new.append(gini(fake[0,:,:,0]))

    lum=round(np.mean(lum),4)
    cont=round(np.mean(cont),4)
    struc=round(np.mean(struc),4)
    ssim=round(np.mean(ssim),4)
    mae=round(np.mean(mae),4)
    gini_orig=round(np.mean(gini_orig),4)
    gini_new=round(np.mean(gini_new),4)

    return lum,cont,struc,ssim,mae,gini_orig,gini_new


