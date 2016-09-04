#===========================================================================================================================================================================
#Module description
#===========================================================================================================================================================================

"""QuantDorsal module for analysis object. 

"""

#===========================================================================================================================================================================
#Importing necessary modules
#===========================================================================================================================================================================

# Quant dorsal modules
import im_module as im
import analysis_module as am
import io_module as iom
from term_module import *
import plot_module as pm
from qDExp import experiment

# OS
import os

# Numpy 
import numpy as np

#===========================================================================================================================================================================
#Class definitions
#===========================================================================================================================================================================

class analysis:

	"""Analysis object for QD toolbox. 
	
	Basically is a container for all experiments done for one particular sample/phenotype, 
	also saves flags on what to do if analysis is run. Flags are:
	
		* ``align``: Align peaks.
		* ``saveTif``: Save channels to tif.
		* ``ilastik``: Run ilastik.
	
	Also saves method used for aligning peaks, ``fitMethod``.
	
	Experiments are saved in ``exps`` list.
	
	"""

	def __init__(self,name):

		self.name=name
		self.exps=[]
		
		self.ilastik=True
		self.saveTif=True
		self.align=True
		
		self.fitMethod='maxIntensity'
	
		
	def addExp(self,image,fn,series):
		
		"""Creates new experiment and adds it to experiment list.
		
		Args:
			image (numpy.ndarray): Image.
			fn (str): Filename of original file.
			series (int): Number of experiment.
		
		Returns:
			list: Updated list of experiments.
		
		"""
		
		newExp=experiment(fn,image,series)	
		self.exps.append(newExp)
		
		return self.exps
	
	def loadData(self,images,fns):
		
		"""Takes list of image data and assigns it to experiment objects.
		
		Args:
			images (list): List of image data.
			fns: (list): List of filenames.
		
		Returns:
			list: Updated list of experiments.
		
		"""
		
		#Check input
		if len(images)!=len(fns):
			printError("Cannout load data, images and fns need to be the same length.")
			return
		
		#Add experiments
		for i in range(len(images)):
			self.addExp(images[i],fns[i],i)
	
		self.exps
	
	def loadImageData(self,fn,nChannel=3,destChannel=0):
		
		"""Loads image data from either a single file or folder.
		
		Loads image data and then calls :py:func:`loadData` to add 
		loaded data into experiments.
		
		Args:
			fn (str): Source file or folder.
			nChannel(int): Number of channels of files.
			destChannel (int): Index in array where channels should go.
		
		Returns:
			list: Updated list of experiments.
		
		
		"""
		
		
		#Load data
		images,fnsLoaded=im.readImageData(fn,nChannel=nChannel,destChannel=destChannel)
		
		#Stick it into experiment objects
		return self.loadData(images,fnsLoaded)
		
	def saveChannelToTif(self,fnOut,channel,prefix="",axes='ZYX',debug=True):
		
		"""Saves image data of a specific channel to tif files for all 
		experiments.
		
		Args:
			fnOut (str): Path where tif files are supposed to be put.
			channel (str): Name of channel to be saved. 
		
		Keyword Args:
			prefix (str): Prefix string for filename.
			axes (str): Order of axis of image.
			debug (bool): Print debugging messages.
			
		Returns: 
			list: List of written tif files.
		
		"""
		
		tifFiles=[]
		for exp in self.exps:
			tf=exp.saveChannelToTif(fnOut,channel,prefix=prefix,axes=axes,debug=debug)
			tifFiles.append(tf)
		
		return tifFiles
		
	def saveDapiChannelToTif(self,fnOut,prefix="",axes='ZYX',debug=True):
		
		"""Saves image data of Dapi channel to tif files for all 
		experiments.
		
		.. note:: If Dapi channel can't be found, will not do anything.
		
		Args:
			fnOut (str): Path where tif files are supposed to be put.
			channel (str): Name of channel to be saved. 
		
		Keyword Args:
			prefix (str): Prefix string for filename.
			axes (str): Order of axis of image.
			debug (bool): Print debugging messages.
			
		Returns: 
			list: List of written tif files.
		
		"""
		
		tifFiles=[]
		for exp in self.exps:
			tf=exp.saveDapiChannelToTif(fnOut,prefix=prefix,axes=axes,debug=debug)
			tifFiles.append(tf)
			
		return tifFiles
	
	def runIlastik(self,channel):
		
		"""Runs ilastik on specific channel for all experiments.
		
		Args:
			channel (str): Name of channel to be run. 
			
		Returns:
			list: List of written h5 files.
		
		
		"""
		
		h5files=[]
		for exp in self.exps:
			h5fn=exp.runIlastik(channel)
			h5files.append(h5fn)
			
		return h5files
	
	def findCorrespondingH5Files(self,channel):
		
		"""Finds corresponding h5 files to tif files specified in channels
		object for all experiments.
		
		Args:
			channel (str): Name of channel. 
		
		"""
		
		for exp in self.exps:
			exp.getChannel(channel).findCorrespondingH5File()
			
	def createAllSignalProfiles(self,signalChannel,maskChannel,debug=False):
		
		"""Creates signal profiles for all experiments for a given channel.
		
		Args:
			signalChannel (str): Name of channel for which signal profiles should be created. 
			maskChannel (str): Name of channel used for masking. 
		
		Keyword Args:
			debug (bool): Print debugging messages.
		
		"""
		
		for exp in self.exps:
			angles,signals=exp.createChannelSignalProfile(signalChannel,maskChannel,debug=debug)
	
	def checkChannels(self):

		"""Checks if all experiments have same channels.
		"""
		
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
		
		Args:
			signalChannel (str): Name of channel to align.
			
		Keyword Args:
			sideChannels (list): List of channel names that are supposed to be aligned the same way as signalChannel.
		
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
		
		"""Checks if min pixel requirement was fulfilled for a given channel.
		
		Returns:
			bool: ``True`` if fulfilled.
		"""
		
		if angles==None:
			return False
		else:
			return True
	
	def filterNaNStacks(self,angles,signals,name=""):
		
		"""Filter out all signal profiles that contain NaN.
		
		Args:
			angles (list) : List of angle arrays.
			signals (list): List of signal arrays.
			
		Keyword Args:
			name (str): Additionally give name of experiment for debugging.
		
		Returns:
			tuple: Tuple containing:
			
				* angles (list): List of angle arrays.
				* signals (list): List of signal arrays.
					
		"""
	
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
		
		"""Computes mean, standard diviation and error over all experiments 
		for given channel.

		Args:
			channel (str): Name of channel.
			
		Returns:
			tuple: Tuple containing:
			
				* angles (numpy.ndarray): Angle array.
				* meanSignal (numpy.ndarray): Mean array.
				* stdSignal (numpy.ndarray): Standard diviation array.
				* sterrSignal (numpy.ndarray): Standard error array.
				
		"""
		
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
		
		"""Computes and plots mean and standard error over all experiments 
		for given channel.
		
		.. note:: Will create new axes if not specified.
		
		Args:
			channel (str): Name of channel.
			
		Keyword Args:
			ax (matplotlib.axes): Axes used for plotting.
			showErrBar (bool): Flag to control if error bars should be shown.
			color (str): Color of plot.
			lw (float): Linewidth of plot.
			title (str): Title of plot.
			showN (bool): Show number of experiments.
			
		Returns:
			matplotlib.axes: Axes used for plotting.
		"""
		
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
		
		"""Saves analysis object to pickle file. 		
		"""
		
		iom.saveToPickle(self,fn=fn)
		
		