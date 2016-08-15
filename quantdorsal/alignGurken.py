"""Script to make final plots for presentation."""

import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import im_module as im

def getPeakWidth(angles,signals,thresh):
	
	idx=np.where(signals>=thresh)[0]
	
	widths=[]
	width=0
	lastIdx=idx[0]
	for i in idx:
		if  (i-lastIdx)>1:
			widths.append(width)	
			width=0
		else:
			width=width+(abs(angles[i]-angles[lastIdx]))
		lastIdx=i
	widths.append(width)

	return widths	
		

fn=sys.argv[1]
fnOut=sys.argv[2]
resFolder="/data/streicha/confocal/Qbio_2016/results/"
res=np.load(fn)

#List of datasets that should be swapped
typ=[-1,2,1,1,-1,2,2,2,2,2,1,1,1,2,1,2,1,1,1,1,1]
swap=[0,0,0,0,0,1,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0]



#Filter nan datasets
j=0
angles=[]
signals=[]

anglesSingle=[]
anglesDouble=[]
signalsSingle=[]
signalsDouble=[]

N=0
Ndouble=0
Nsingle=0
for i in range(res.shape[1]):

	if bool(np.isnan(res[0,i,0,:]).sum()):
		continue
	if bool(np.isnan(res[1,i,0,:]).sum()):
		continue
	
	print i	
	if typ[i]==-1:
		continue
	
	
	angles.append(res[0,i,0,:].copy())
	signals.append(res[1,i,0,:].copy())
	N=N+1
	if typ[i]==2:

		Ndouble=Ndouble+1
		if swap[i]==1:
			anglesDouble.append(res[0,i,0,::-1].copy())
		        signalsDouble.append(res[1,i,0,::-1].copy())
		else:
			anglesDouble.append(res[0,i,0,:].copy())
                        signalsDouble.append(res[1,i,0,:].copy())
	else:
		Nsingle=Nsingle+1
		anglesSingle.append(res[0,i,0,:].copy())
		signalsSingle.append(res[1,i,0,:].copy())	

#Compute mean/std	
signals=np.asarray(signals)
meanSignal=np.mean(signals,axis=0)
stdSignal=np.std(signals,axis=0)

signalsSingle=np.asarray(signalsSingle)
meanSignalSingle=np.mean(signalsSingle,axis=0)
stdSignalSingle=np.std(signalsSingle,axis=0)

signalsDouble=np.asarray(signalsDouble)
meanSignalDouble=np.mean(signalsDouble,axis=0)
stdSignalDouble=np.std(signalsDouble,axis=0)

#Compute with of peaks
print angles[0][np.where(meanSignalSingle>=1.0)]
print angles[0][np.where(meanSignalDouble>=1.0)]

print getPeakWidth(angles[0],meanSignalSingle,1.0)
print getPeakWidth(angles[0],meanSignalDouble,1.0)


#Plot everythiong
fig=plt.figure()
fig.show()

ax=fig.add_subplot(111)
ax.errorbar(angles[0],meanSignal,yerr=stdSignal)
ax=im.turnAxesForPub(ax,figWidthPt=500)

Nstring=" (N="+str(N)+")"
ax.set_title(fnOut+Nstring)
ax.set_ylabel("Intensity (AU)")
ax.set_xlabel("Embryo Circumfarence (radians)")

xtick = np.linspace(-np.pi, np.pi, 5)
print xtick
xlabel = [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$+\frac{\pi}{2}$",   r"$+\pi$"]
ax.set_xticks(xtick)

ax.set_xticklabels(xlabel)

plt.draw()

fig.savefig(resFolder+fnOut+"_errorbar.eps")

#Plot single
fig=plt.figure()
fig.show()

ax=fig.add_subplot(111)
ax.errorbar(angles[0],meanSignalSingle,yerr=stdSignalSingle)

ax=im.turnAxesForPub(ax,figWidthPt=500)
NString=" (N="+str(Nsingle)+")"
ax.set_title(fnOut+NString)
ax.set_ylabel("Intensity (AU)")
ax.set_xlabel("Embryo Circumfarence (radians)")

xtick = np.linspace(-np.pi, np.pi, 5)
print xtick
xlabel = [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$+\frac{\pi}{2}$",   r"$+\pi$"]
ax.set_xticks(xtick)

ax.set_xticklabels(xlabel)



plt.draw()

fig.savefig(resFolder+fnOut+"_single_errorbar.eps")

bar=np.ones(angles[0].shape)

ax.plot(angles[0],bar,'k')

ax=im.turnAxesForPub(ax,figWidthPt=500)
ax.set_title(fnOut)
ax.set_ylabel("Intensity (AU)")
ax.set_xlabel("Embryo Circumfarence (radians)")

xtick = np.linspace(-np.pi, np.pi, 5)
print xtick
xlabel = [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$+\frac{\pi}{2}$",   r"$+\pi$"]
ax.set_xticks(xtick)

ax.set_xticklabels(xlabel)


plt.draw()
fig.savefig(resFolder+fnOut+"_single_errorbar_bar.eps")

#Plot double
fig=plt.figure()
fig.show()

ax=fig.add_subplot(111)
ax.errorbar(angles[0],meanSignalDouble,yerr=stdSignalDouble)
plt.draw()

fig.savefig(resFolder+fnOut+"_double_errorbar.eps")

ax.plot(angles[0],bar,'k')

ax=im.turnAxesForPub(ax,figWidthPt=500)
NString=" (N="+str(Ndouble)+")"
ax.set_title(fnOut+NString)
ax.set_ylabel("Intensity (AU)")
ax.set_xlabel("Embryo Circumfarence (radians)")

xtick = np.linspace(-np.pi, np.pi, 5)
print xtick
xlabel = [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$+\frac{\pi}{2}$",   r"$+\pi$"]
ax.set_xticks(xtick)

ax.set_xticklabels(xlabel)


plt.draw()
fig.savefig(resFolder+fnOut+"_double_errorbar_bar.eps")



try:	
	os.mkdir(resFolder+fnOut)
except:
	pass

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

for i in range(len(angles)):
	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.plot(angles[i],signals[i])
	plt.draw()
	fig.savefig(resFolder+fnOut+"/"+fnOut+"_embryo"+str(i)+".png")
	
raw_input("Press Enter")


