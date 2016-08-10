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

#Parse in filename
fnIn=sys.argv[1]

#Load bioformats image data
images,meta=im.readBioFormats(fnIn)

#Save stacks to tifs
prefix=os.path.basename(fnIn).replace(".zip.lif","")
fnOut=os.path.dirname(fnIn)+"/tifs/"

try:
	os.mkdir(fnOut)
except OSError:
	pass

tifFiles=im.saveImageSeriesToStacks(images,fnOut,prefix=prefix,axes='ZYX',channel=0,debug=True)

#Run ilastik
#print tifFiles
#tifFiles=tifFiles[0:2]
#print tifFiles
probFiles=ilm.runIlastik(tifFiles,classFile="classifiers/Dorsal_Dapi_alex3.ilp")

#Threshhold
#probFiles=['../data/tifs/160804_toll10B_dapi_series0_c0_Probabilities.h5', '../data/tifs/160804_toll10B_dapi_series1_c0_Probabilities.h5']

fig=plt.figure()
fig.show()

for i,fn in enumerate(probFiles):
		
	angles,signals=am.createSignalProfileFromH5(images[i],fn,signalChannel=2,probThresh=0.8,probIdx=0,maxInt=True,hist=False,bins=50,maskBkgd=True,debug=True)
	ax=fig.add_subplot(1,len(probFiles),i)
	
	for j in range(len(angles)):
		
		color=cm.jet(float(j)/len(angles))
		
		ax.plot(angles[j],signals[j],color=color,label="z = "+str(j))
		#raw_input()
plt.draw()	
	
print "done"
raw_input()


	

