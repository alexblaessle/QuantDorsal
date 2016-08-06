"""Test script to fit an ellipse to a mask."""

import numpy as np
import im_module as im
import matplotlib.pyplot as plt

#Define ellipse
alpha=30.
a=5
b=3
x0=5
y0=3

#Load mask
maskDilated=np.load("maskDilated.npy")

#Build meshgrid. Note, out of some reason we need to change here coordinates 1 and 0, basically transpose X,Y
x=np.arange(maskDilated.shape[1])
y=np.arange(maskDilated.shape[0])
X,Y=np.meshgrid(x,y)

#Extract coordinates that are 1
x=X[np.where(maskDilated>0.5)].flatten()
y=Y[np.where(maskDilated>0.5)].flatten()

#Fit ellipse
ell=im.fitEllipse(x,y)
center,lengths,rot=im.decodeEllipse(ell)
xF,yF=im.ellipseToArray(center,lengths,rot)

#Plot
fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)
ax.plot(x,y,'r.')
ax.plot(xF,yF,'b--')

plt.draw()
raw_input()




