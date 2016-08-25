#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test script for quantDorsal."""

#========================================================
#Importing modules
#========================================================

#QD Modules
import im_module as im
import ilastik_module as ilm
import analysis_module as am
import plot_module as pm
from term_module import *

#QD classes
import qDAnalysis

#Numpy
import numpy as np

#System
import sys
import os

#Plotting
import matplotlib.pyplot as plt
from matplotlib import cm

#Parsing
from optparse import OptionParser



#========================================================
#Set some global parameters
#========================================================

classifier="classifiers/Dorsal_Dapi_alex3.ilp"
#classifier="classifiers/done_with_matt.ilp"
signalChannel=2
dapiChannel=0
probThresh=0.5
probIdx=0
proj='max'
bins=50
debug=True
bkgd=None
minPix=10000
median=None
fitMethod='maxIntensity'


#========================================================
#Flags what to do
#========================================================

ilastik=False

#========================================================
#Load image data
#========================================================

#Parse in filename
fnIn=sys.argv[1]

#Build name/folder
name=os.path.basename(fnIn).split(".")[0]
folder=os.path.dirname(fnIn)

#Create analysis object
an=qDAnalysis.analysis(name)

#Load data
an.loadImageData(fnIn,nChannel=3,destChannel=0)

#========================================================
#Set some options
#========================================================

an.setChannelOpts("Dapi",debug=False,classifier=classifier,proj=proj,bins=bins,minPix=minPix)

#========================================================
#Write Dapi Channel as tifs
#========================================================

fnOut=folder+"/"+name+"/"
try:
	os.mkdir(fnOut)
except OSError:
	pass

an.saveDapiChannelToTif(fnOut,prefix=name,axes='ZYX',debug=True)

#========================================================
#Run ilastik
#========================================================

if ilastik:
	an.runIlastik("Dapi")
else:
	an.findCorrespondingH5Files("Dapi")

#========================================================
#Create signal profiles
#========================================================

an.createAllSignalProfiles("Dorsal","Dapi")

#========================================================
#Align signal profiles
#========================================================

an.alignSignalProfiles("Dorsal",sideChannels=[])


an.plotAlignedSignal("Dorsal",title="Toll10B")



raw_input()











	


