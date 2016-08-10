"""Test script to fit an ellipse to a mask."""

import numpy as np
import im_module as im
import matplotlib.pyplot as plt
import analysis_module as am

#Define ellipse
alpha=30.
a=5
b=3
x0=5
y0=3

#Load mask
maskDilated=np.load("maskDilated.npy")

center,lengths,rot,xF,yF=am.fitEllipseToMask(maskDilated)



#Plot
fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)
ax.imshow(maskDilated)
#ax.plot(x,y,'r.')
ax.plot(xF,yF,'g--')

plt.draw()
raw_input()




