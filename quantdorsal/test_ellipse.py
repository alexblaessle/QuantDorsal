"""Little script to test fit ellipse"""

import numpy as np
import analysis_module as am
import matplotlib.pyplot as plt

#Define ellipse
alpha=30.
a=5
b=3
x0=5
y0=3
sigma=1.
mu=0.

#Generate ellipse
x,y=am.ellipseToArray([x0,y0],[a,b],alpha)

#Add noise
x=x+ np.random.normal(loc=0.0, scale=0.1, size=x.shape)
y=y+ np.random.normal(loc=0.0, scale=0.1, size=y.shape)

#Fit ellipse
ell=am.fitEllipse(x,y)
center,lengths,rot=am.decodeEllipse(ell)
xF,yF=am.ellipseToArray(center,lengths,rot)
xR,yR=am.ellipseToArray(center,lengths,0)

#Choose some points
xC,yC=xF[0:200:50],yF[0:200:50]

#Compute angle to choose points
angles=am.invEllipse(x,y,center,lengths,rot)
anglesC=am.invEllipse(xC,yC,center,lengths,rot)

#Generate values around ellipse
v=1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (angles - mu)**2 / (2 * sigma**2) )

print rot
print anglesC

#Plot
fig=plt.figure()
fig.show()
ax=fig.add_subplot(221)
#ax.plot(x,y,'r')
ax.plot(xF,yF,'b--')
#ax.plot(xC,yC,'g*')
ax.plot(xR,yR,'m-.')


ax=fig.add_subplot(222)
ax.plot(angles,'r')

#ax.plot(angles,x,'r')
#ax.plot(angles,y,'b')


ax=fig.add_subplot(223)
ax.plot(angles,v,'r')
#ax.plot(angles,y,'b')


plt.draw()
raw_input()




