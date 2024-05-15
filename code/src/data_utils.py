import os
from osgeo import gdal
import imageio
import numpy as np

from time import time
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def clip_and_scale_image(img, img_type='naip', clip_min = None, clip_max=None):
    """
    Clips and scales bands to between [0, 1] for NAIP, RGB, and Landsat
    satellite images. Clipping applies for Landsat only.
    """
    if img_type in ['naip', 'rgb']:
        return img / 255
    if img_type == "no_transform":
        return img
    elif img_type == 'landsat':
        if clip_min is None:
            clip_min = 0
        if clip_max is None:
            clip_max = 10000
        return np.clip(img, clip_min, clip_max) / (clip_max - clip_min)
    elif img_type == "sentinel":
        # based on the training set evaluation the outliers land outside the following margin
        if clip_max is None:
            clip_max = [1914. , 1842. , 1872.5, 2309. , 2512.5, 4012.5, 4895. , 5052. ,
       1643. ,   21.5, 4220.5, 3070.5, 5497.5]
        if clip_min is None:
            clip_min = [816.0, 314.0, 100.5, 0.0, 174.0, 153.0, 128.0, 0.0, 40.0, 1.5, 2.0, 1.0, 44.0]

        clip_min = np.reshape(clip_min, (len(clip_min), 1, 1)) # bands first
        clip_max= np.reshape(clip_max, (len(clip_max), 1, 1)) # bands first

        img = (np.clip(img, clip_min, clip_max) - clip_min) / (clip_max - clip_min)
        return img
    else:
        raise ValueError("incorrect img type")

