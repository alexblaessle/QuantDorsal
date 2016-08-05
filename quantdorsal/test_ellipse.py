"""Little script to test fit ellipse"""

import numpy as np
import im_module as im
import matplotlib.pyplot as plt

#Define ellipse
alpha=30.
a=5
b=3
x0=5
y0=3

#Generate ellipse
x,y=im.ellipseToArray([x0,y0],[a,b],alpha)

#Add noise
x=x+ np.random.normal(loc=0.0, scale=0.1, size=x.shape)
y=y+ np.random.normal(loc=0.0, scale=0.1, size=y.shape)

#Fit ellipse
ell=im.fitEllipse(x,y)
center,lengths,rot=im.decodeEllipse(ell)
xF,yF=im.ellipseToArray(center,lengths,rot)

#Plot
fig=plt.figure()
fig.show()
ax=fig.add_subplot(111)
ax.plot(x,y,'r')
ax.plot(xF,yF,'b--')

plt.draw()
raw_input()




