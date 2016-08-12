#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test script for quantDorsal."""

#Importing modules
import im_module as im
import ilastik_module as ilm
import analysis_module as am
from term_module import *

#Numpy
import numpy as np

#System
import sys
import os

#Plotting
import matplotlib.pyplot as plt
from matplotlib import cm

#Set some global parameters
#classifier="classifiers/stacks_classifier.ilp"
classifier="classifiers/done_with_matt.ilp"
signalChannel=2
dapiChannel=0
probThresh=0.5
probIdx=0
proj='max'
bins=50
debug=False
bkgd=None
minPix=10000

#Flags what to do
ilastik=True

#Parse in filename
fnIn=sys.argv[1]

#Load image data
images=im.readImageData(fnIn,nChannel=3,destChannel=0)

#Save stacks to tifs
prefix=os.path.basename(fnIn).replace(".zip.lif","")
if prefix=="":
	prefix="out"
fnOut=os.path.dirname(fnIn)+"/"+prefix+"/"

try:
	os.mkdir(fnOut)
except OSError:
	pass

tifFiles=im.saveImageSeriesToStacks(images,fnOut,prefix=prefix,axes='ZYX',channel=dapiChannel,debug=True)

#Run ilastik
if ilastik:
	probFiles=ilm.runIlastik(tifFiles,classFile=classifier)
else:
	probFiles=ilm.getH5FilesFromFolder(fnOut)

#Filter out datasets that lack h5 file (out of whatever reason)
brokenIdx=ilm.filterBrokenH5(probFiles,tifFiles)
images = [i for j, i in enumerate(images) if j not in brokenIdx]
tifFiles = [i for j, i in enumerate(tifFiles) if j not in brokenIdx]

allSignalsAligned=[]
allAnglesAligned=[]

#Loop through all probability files
for i,fn in enumerate(probFiles):
	
	print i
	
	#Generate angular distributions	
	angles,signals=am.createSignalProfileFromH5(images[i],fn,signalChannel=signalChannel,probThresh=probThresh,probIdx=probIdx,
					     proj=proj,bins=bins,bkgd=bkgd,minPix=minPix,median=3,debug=debug)

	#Bookkeeping lists
	anglesAligned=[]
	signalsAligned=[]
	
	if angles==None:
		printWarning("Had to exclude dataset "  + os.path.basename(tifFiles[i]) + " due to minPix requirement.")
		continue
	
	#Align distributions
	for j in range(len(angles)):
		
		if np.isnan(angles[j].max()):
			printWarning("Had to exclude dataset "  + os.path.basename(tifFiles[i]) + " due to minPix requirement.")
			continue
		
		#Align with maximum value
		angleAligned,signalAligned=am.alignDorsal(angles[j],np.asarray([signals[j],signals[j]]))
		anglesAligned.append(angleAligned)
		signalsAligned.append(signalAligned[0,:])
	
	allAnglesAligned.append(anglesAligned)
	allSignalsAligned.append(signalsAligned)
	
figAll=plt.figure()
figAll.show()

#Create axis
axAll=figAll.add_subplot(1,1,1)	
for i in range(len(allAnglesAligned)):
		
	color=cm.jet(float(i)/len(allAnglesAligned))
	for j in range(len(allAnglesAligned[i])):
			
		axAll.plot(allAnglesAligned[i][j],allSignalsAligned[i][j],color=color,label="series = "+str(i))
		
		
		plt.draw()	

#Make stats figure
meanSignal,stdSignal=am.getStats(allSignalsAligned)
fig,axes=im.makeAxes([1,1])

print meanSignal.shape
print stdSignal.shape

meanSignal=meanSignal[0]
stdSignal=stdSignal[0]
print meanSignal.shape

print allAnglesAligned[0][0].shape
axes[0].errorbar(allAnglesAligned[0][0], meanSignal, yerr=stdSignal)
plt.draw()
fig.savefig(fnOut+"statsFig.png")

results=np.asarray([allAnglesAligned,allSignalsAligned])

#Save results
figAll.savefig(fnOut+"finalFig.png")
np.save(fnOut+"distributions.npy",results)

print "done"





	


