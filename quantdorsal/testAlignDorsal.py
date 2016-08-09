# -*- coding: utf-8 -*-
"""
Script to test dorsal alignment procedure
"""


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
import ilastik_module as il
import im_module as im
import h5py

mu = np.random.rand()*4*math.pi-2*math.pi
variance = np.random.rand()*2*math.pi
sigma = math.sqrt(variance)
x = np.linspace(-2*math.pi, 2*math.pi, 100)
heap = mlab.normpdf(x,mu,sigma)
intensity = np.array([heap,heap])
print(intensity.shape)

aligned = im.alignDorsal(x,intensity)
print(aligned[0])


plt.plot(aligned[0],aligned[1][0],'r')
plt.plot(x,heap)
plt.show()
