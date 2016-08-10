"""Script to test histogram function/angular averaging."""

import numpy as np
import matplotlib.pyplot as plt


#angular vector
x=np.linspace(-np.pi,np.pi,200)

#Make normal distribution
sigma=0.5
mu=0
y=1/(sigma * np.sqrt(2 * np.pi)) * np.exp( -(x - mu)**2 / (2 * sigma**2))
y=3*y

#Add noise
y2=y+np.random.normal(loc=0., scale=0.1, size=y.shape)

#Histogram
#bins,hist=np.histogram(y2, bins=10)

bins=np.linspace(x.min(),x.max(),11)

#Find idxs
idx=np.digitize(x,bins)
bin_means = [y2[idx == i].mean() for i in range(1, len(bins))]


#xBin=x[0]+np.diff(bins)


print bins

xBin=np.diff(bins)/2.+bins[:-1]

print xBin
bin_means=np.asarray(bin_means)


plt.plot(x,y,'r')
plt.plot(x,y2,'b')
plt.plot(xBin,bin_means,'g')



plt.show()



