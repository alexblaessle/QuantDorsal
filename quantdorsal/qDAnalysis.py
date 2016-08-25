import im_module as im
import analysis_module as am
import io_module as iom
from term_module import *
import plot_module as pm
from qDExp import experiment

import os


import numpy as np

class analysis:

	def __init__(self,name):

		self.name=name
		self.exps=[]
		
		self.ilastik=True
		self.saveTif=True
		self.align=True
		
		self.fitMethod='maxIntensity'
	
		
	def addExp(self,image,fn,series):
		
		newExp=experiment(fn,image,series)	
		self.exps.append(newExp)
		
		return self.exps
	
	def loadData(self,images,fns):
		
		#Check input
		if len(images)!=len(fns):
			printError("Cannout load data, images and fns need to be the same length.")
			return
		
		#Add experiments
		for i in range(len(images)):
			self.addExp(images[i],fns[i],i)
	
		self.exps
	
	def loadImageData(self,fn,nChannel=3,destChannel=0):
		
		#Load data
		images,fnsLoaded=im.readImageData(fn,nChannel=nChannel,destChannel=destChannel)
		
		#Stick it into experiment objects
		return self.loadData(images,fnsLoaded)
		
	def saveChannelToTif(self,fnOut,channel,prefix="",axes='ZYX',debug=True):
		
		tifFiles=[]
		for exp in self.exps:
			tf=exp.saveChannelToTif(fnOut,channel,prefix=prefix,axes=axes,debug=debug)
			tifFiles.append(tf)
		
		return tifFiles
		
	def saveDapiChannelToTif(self,fnOut,prefix="",axes='ZYX',debug=True):
		
		tifFiles=[]
		for exp in self.exps:
			tf=exp.saveDapiChannelToTif(fnOut,prefix=prefix,axes=axes,debug=debug)
			tifFiles.append(tf)
			
		return tifFiles
	
	def runIlastik(self,channel):
		
		h5files=[]
		for exp in self.exps:
			h5fn=exp.runIlastik(channel)
			h5files.append(h5fn)
			
		return h5files
	
	def findCorrespondingH5Files(self,channel):
		
		for exp in self.exps:
			exp.getChannel(channel).findCorrespondingH5File()
			
	def createAllSignalProfiles(self,signalChannel,maskChannel,debug=False):
		
		for exp in self.exps:
			angles,signals=exp.createChannelSignalProfile(signalChannel,maskChannel,debug=debug)
	
	def checkChannels(self):

		"""Checks if all experiments have same channels."""
		
		b=True
		
		for i,exp in enumerate(self.exps):
			names=exp.getChannelNames()
			names.sort()
			
			if i>0:
				if names!=lastNames:
					printWarning(exp.name+ " has not the same channels than "+ self.exps[i-1])
					b=False	
					
			lastNames=list(names)
			
		return b	
	
	def alignSignalProfiles(self,signalChannel,sideChannels=[]):
		
		"""Aligns signal profiles.
					
		.. warning:: When we align here, then this is done per stack. Thus, single stacks might get aligned differently.
		   This is not a problem if there was a projection before, since then there is only a single stack. 
		   Otherwise this part should be used with caution. Will need to fix this later.
		
		"""
		
		
		#Collecting all results and aligning them
		anglesAligned=[]
		signalsAligned=[]
		
		#Loop through all experiments
		for exp in self.exps:
			
			#Grab channels and set back their aligned lists
			channel=exp.getChannel(signalChannel)
			channel.anglesAligned=[]
			channel.signalsAligned=[]
			
			if len(sideChannels)>0:
				for sideChannel in sideChannels:
					sideChannels[i]=exp.getChannel(sideChannel)
					sideChannels[i].anglesAligned=[]
					sideChannels[i].signalsAligned=[]
					
			#Check some reqs
			if self.checkMinPix(channel.angles):
				
				angles,signals=self.filterNaNStacks(channel.angles,channel.signals,name=exp.series)
				
				#Loop through all stacks
				for j in range(len(channel.angles)):
					
					#Build arrays for passing it to aligment function
					anglesSub=channel.angles[j]
					signalsSub=[channel.signals[j]]
									
					if len(sideChannels)>0:
						for sideChannel in sideChannels:
							signalsSub.append(sideChannel.signals[j])
					else:
						signalsSub.append(channel.signals[j])
						
					signalsSub=np.asarray(signalsSub)
					
					#Aligning
					angles,signals=angleAligned,signalAligned=am.alignDorsal(anglesSub,signalsSub,method=self.fitMethod)
					
					#Dumping results in respecting channels
					channel.anglesAligned.append(angles)
					channel.signalsAligned.append(signals[0,:])
					
					for i,sideChannel in enumerate(sideChannels):
						channel.anglesAligned.append(angles)
						channel.signalsAligned.append(signals[0,:])
					
		return 	
	
	def setOpts(self,debug=False,**kwargs):
		
		"""Sets options."""
		
		for kwarg in kwargs:
			if not hasattr(self,kwarg):
				printWarning("Cannot set attribute "+kwarg+" . Analysis has not this property." )
			else:
				if debug:
					printNote("Going to set "+kwarg+" from " + str(getattr(self,kwarg)) + " to "+ kwargs[kwarg])
				setattr(self,kwarg,kwargs[kwarg])

	def setExpOpts(self,debug=False,**kwargs):
		
		"""Sets options for all experiments."""
		
		for exp in self.exps:
			exp.setOpts(debug=debug,**kwargs)
	
	def setChannelOpts(self,name,debug=False,**kwargs):
		
		"""Sets options for all channels of certain name."""
		
		for exp in self.exps:
			exp.getChannel(name).setOpts(debug=debug,**kwargs)
			
	
	
	def checkMinPix(self,angles):
		
		if angles==None:
			return False
		else:
			return True
	
	def filterNaNStacks(self,angles,signals,name=""):
		
		#Align distributions
		anglesFiltered=[]
		signalsFiltered=[]
		for j in range(len(angles)):
			
			if am.filterNaNArray(angles[j]) and am.filterNaNArray(signals[j]):
				anglesFiltered.append(angles)
				signalsFiltered.append(signals)
			else:
				printWarning("Had to exclude stack "+str(j) + " experiment "  + str(name) + " due to NaN.")
			
		return anglesFiltered,signalsFiltered	
		
	def getAlignedSignalStats(self,channel):
		
		#Compute 
		signals=[]
		angles=[]
		
		for exp in self.exps:
			ch=exp.getChannel(channel)
			for i in range(len(ch.anglesAligned)): 
				angles.append(ch.anglesAligned[i])
				signals.append(ch.signalsAligned[i])
					
		#Compute mean/std	
		N=len(signals)
		signals=np.asarray(signals)
		meanSignal=np.mean(signals,axis=0)
		stdSignal=np.std(signals,axis=0)
		sterrSignal=stdSignal/np.sqrt(len(signals))
		
		return angles[0],meanSignal,stdSignal,sterrSignal,N
		
	def plotAlignedSignal(self,channel,ax=None,showErrBar=True,color='r',lw=1.,title="",showN=True):
		
		#Get data
		angles,meanSignal,stdSignal,sterrSignal,N=self.getAlignedSignalStats(channel)
		
		#Make axes if necessary
		if ax==None:
			fig,axes=pm.makeAxes([1,1])
			ax=axes[0]
			
		if showErrBar:
			ax.errorbar(angles,meanSignal,yerr=sterrSignal,color=color,linewidth=lw)
		else:
			ax.plot(angles,meanSignal,color=color,lw=lw)
		
		ax=pm.redraw(ax)
		
		ax=pm.turnAxesForPub(ax,figWidthPt=500)
		ax=pm.turnIntoRadialPlot(ax)
		
		if showN:
			title=title+"(N="+str(N)+")"
		
		ax.set_title(title)
		
		return ax
	
	
	
	def save(self,fn):
		iom.saveToPickle(self,fn=fn)
		
		
		
			
		
	