"""Script to make final plots for presentation."""

import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import im_module as im


fn=sys.argv[1]
fnOut=sys.argv[2]
resFolder="/data/streicha/confocal/Qbio_2016/results/"
res=np.load(fn)

#Filter nan datasets
j=0
angles=[]
signals=[]

N=0

for i in range(res.shape[1]):

	if bool(np.isnan(res[0,i,0,:]).sum()):
		continue
	if bool(np.isnan(res[1,i,0,:]).sum()):
		continue
	
	angles.append(res[0,i,0,:].copy())
	signals.append(res[1,i,0,:].copy())

	N=N+1
	
signals=np.asarray(signals)
meanSignal=np.mean(signals,axis=0)
stdSignal=np.std(signals,axis=0)

fig=plt.figure()
fig.show()

ax=fig.add_subplot(111)
ax.errorbar(angles[0],meanSignal,yerr=stdSignal)
ax=im.turnAxesForPub(ax,figWidthPt=500)
ax.set_title(fnOut+" (N="+str(N)+")")
ax.set_ylabel("Intensity (AU)")
ax.set_xlabel("Embryo Circumfarence (radians)")

xtick = np.linspace(-np.pi, np.pi, 5)
print xtick
xlabel = [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$+\frac{\pi}{2}$",   r"$+\pi$"]
ax.set_xticks(xtick)

ax.set_xticklabels(xlabel)
plt.draw()

fig.savefig(resFolder+fnOut+"_errorbar.eps")

fig=plt.figure()
fig.show()

ax=fig.add_subplot(111)
ax.plot(angles[0],1/meanSignal*stdSignal,'b')
#ax.plot(angles[0],1/meanSignalSingle*stdSignalSingle,'r',label="Single")
#ax.plot(angles[0],1/meanSignalDouble*stdSignalDouble,'g',label="Double")
plt.legend()

ax=im.turnAxesForPub(ax,figWidthPt=500)
ax.set_title(fnOut)
ax.set_ylabel("Coefficient of variation")
ax.set_xlabel("Embryo Circumfarence (radians)")

xtick = np.linspace(-np.pi, np.pi, 5)
print xtick
xlabel = [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$+\frac{\pi}{2}$",   r"$+\pi$"]
ax.set_xticks(xtick)

ax.set_xticklabels(xlabel)
plt.draw()
fig.savefig(resFolder+fnOut+"_coeff.eps")

try:	
	os.mkdir(resFolder+fnOut)
except:
	pass

for i in range(len(angles)):
	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.plot(angles[i],signals[i])
	plt.draw()
	fig.savefig(resFolder+fnOut+"/"+fnOut+"_embryo"+str(i)+".png")
	
	
	
	


raw_input("Press Enter")


