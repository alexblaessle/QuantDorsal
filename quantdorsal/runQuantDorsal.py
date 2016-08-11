#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test script for quantDorsal."""

#Importing modules
import im_module as im
import ilastik_module as ilm
import analysis_module as am

#Numpy
import numpy as np

#System
import sys
import os

#Plotting
import matplotlib.pyplot as plt
from matplotlib import cm

#Set some global parameters
classifier="classifiers/Dorsal_Dapi_alex3.ilp"
signalChannel=2
dapiChannel=0
probThresh=0.8
probIdx=0
proj='mean'
bins=50
debug=False
bkgd=None

#Flags what to do
ilastik=False

#Parse in filename
fnIn=sys.argv[1]

#Load image data
images=im.readImageData(fnIn,nChannel=3,destChannel=0)

#Save stacks to tifs
prefix=os.path.basename(fnIn).replace(".zip.lif","")
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

#probFiles=['../data/tifs/160804_toll10B_dapi_series0_c0_Probabilities.h5', '../data/tifs/160804_toll10B_dapi_series1_c0_Probabilities.h5']

allSignalsAligned=[]
allAnglesAligned=[]


#Loop through all probability files
for i,fn in enumerate(probFiles):
		
	#Generate angular distributions	
	angles,signals=am.createSignalProfileFromH5(images[i],fn,signalChannel=signalChannel,probThresh=probThresh,probIdx=probIdx,
					     proj=proj,bins=bins,bkgd=bkgd,debug=debug)

	#Bookkeeping lists
	anglesAligned=[]
	signalsAligned=[]
	
	#Align distributions
	for j in range(len(angles)):
		
		#Align with maximum value
		angleAligned,signalAligned=am.alignDorsal(angles[j],np.asarray([signals[j],signals[j]]))
		anglesAligned.append(angleAligned)
		signalsAligned.append(signalAligned[0,:])
	
	allAnglesAligned.append(anglesAligned)
	allSignalsAligned.append(signalsAligned)
	
#Plot angular distributions
#figSeries=plt.figure()
#figSeries.show()

figAll=plt.figure()
figAll.show()

#Create axis
axAll=figAll.add_subplot(1,1,1)
rowIdx=0

#rowIdx=np.mod(len(allAnglesAligned),3)
#Idx=np.mod(len(allAnglesAligned),3)

	
	
for i in range(len(allAnglesAligned)):
	
	#if columnIdx==1:
		#rowIdx=rowIdx+1
		
	#print rowIdx,columnIdx,i+1
	#axSeries=figSeries.add_subplot(rowIdx,columnIdx,i+1)
		
	color=cm.jet(float(i)/len(allAnglesAligned))
	for j in range(len(allAnglesAligned[i])):
		
		#axSeries.plot(allAnglesAligned[i][j],allSignalsAligned[i][j],color=color)
		#axSeries.set_title("series = "+str(i))
		
		
		axAll.plot(allAnglesAligned[i][j],allSignalsAligned[i][j],color=color,label="series = "+str(i))
		
		
		plt.draw()	
	
	
print "done"
raw_input()


	

