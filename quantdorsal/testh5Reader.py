# -*- coding: utf-8 -*-
"""
Test script to check h5 file reader.
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import ilastik_module as il
import im_module as im



fn = 'classifiers/temp/Dorsal_Dapi_1.h5'

array = il.readH5(fn)
mask = il.makeProbMask(array)

bkg = array[7,:,:,0]
frg = array[7,:,:,1]
msk = mask[7,:,:,0]

print(frg.shape)

#imgplot1 = plt.imshow(bkg)
#imgplot2 = plt.imshow(frg)
imgplot3 = plt.imshow(msk)