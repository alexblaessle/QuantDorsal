import im_module as im
import ilastik_module as ilm
import analysis_module as am
import plot_module as pm
from term_module import *

import os

import numpy as np


class experiment:
	
	"""Class to store quant dorsal analysis results for a single experiment.
	
	Important properties are:
	
		* image: Image data
		* orgFn: Path to original file
		* series: Number of series of this experiment
		* nStacks: Number of zStacks
		* channels: dictionary assigning channel names with indices.
		* tifFn: dictionary assigning channel names with indices.
		* channels: dictionary assigning channel names with indices.
		* tifFn: dictionary assigning channel names with respective tif file.
		* h5Fn: dictionary assigning channel names with respective h5 file.
		
		
	"""
	
	def __init__(self,orgFn,image,series):

		#Raw data
		self.image=image
		self.orgFn=orgFn
		self.series=series
		
		#List of channels
		self.channels=[]
		
		#Get number of stacks	
		self.nStacks=image.shape[1]
		
		#Generate dictionary for channels
		self.setDefaultChannels()
		
		#Set default analysis options
		#self.setDefaultOpts()
	
	def addChannel(self,name,idx,classifier="",h5Fn="",tifFn=""):
		
		"""Adds channel object to channels list."""
		
		ch=channel(self,name,idx,classifier=classifier,h5Fn=h5Fn,tifFn=tifFn)
		self.channels.append(ch)
		
		return self.channels
	
	def setDefaultChannels(self):
		
		"""Creates default channel dictionary."""
		
		#Default Channels
		self.addChannel("Dapi",0)
		self.addChannel("Myosin",1)
		self.addChannel("Dorsal",2)
		
		if self.image.shape[0]>len(self.channels):
			printWarning("Somehow there are more than 3 channels, will add some unknown ones.")
			
			for i in range(self.image.shape[0]-len(self.channels)):
				self.addChannel("unknown"+str(i+1),len(self.channels))
				
		return self.channels
			
	
		
	def setOpts(self,debug=False,**kwargs):
		
		"""Sets analysis options."""
		
		for kwarg in kwargs:
			if not hasattr(self,kwarg):
				printWarning("Cannot set attribute "+kwarg+" . qDExp has not this property." )
			else:
				if debug:
					printNote("Going to set "+kwarg+" from " + str(getattr(self,kwarg)) + " to "+ kwargs[kwarg])
				setattr(self,kwarg,kwargs[kwarg])
		
	def saveChannelToTif(self,fnOut,channel,prefix="",axes='ZYX',debug=True):
		
		"""Saves channel to tif file.
		"""
		
		#Image into list
		images=[self.image]
		
		#Get Name of channel
		channel=self.getChannel(channel)
		
		#Build prefix string
		if len(prefix)==0:
			prefix=prefix+channel.name+"_"+"series"+str(self.series)
		else:	
			prefix=prefix+"_"+channel.name+"_"+"series"+str(self.series)
		
		#Write tif file	
		tifFiles=im.saveImageSeriesToStacks(images,fnOut,prefix=prefix,axes='ZYX',channel=channel.idx,debug=True)
		
		#Save filename of tif
		channel.tifFn=tifFiles[0]
		
		return channel.tifFn
		
	def saveDapiChannelToTif(self,fnOut,prefix="",axes='ZYX',debug=True):
		
		"""Saves Dapi channel to tif file."""
		
		return self.saveChannelToTif(fnOut,"Dapi",prefix=prefix,axes=axes,debug=debug)
		
	def getChannelByIdx(self,idx):
		
		"""Returns the name of a channel given the channel index.
		
		"""
		
		for channel in self.channels:
			if channel.idx==idx:
				return channel	
			
		printWarning("Cannot find channel with idx "+str(idx)+". Going to return empty None.")
		
		return None
	
	def getChannelByIName(self,name):
		
		"""Returns the name of a channel given the channel index.
		
		"""
		
		for channel in self.channels:
			if channel.name==name:
				return channel	
			
		printWarning("Cannot find channel with name "+str(name)+". Going to return None.")
		
		return None
	
			
	def runIlastik(self,channel):
		
		"""Runs ilastik on specific channel.
		
		.. note:: ``tifFn`` and ``classifier`` needs to be set to work.
		
		.. note:: Channel can be either given via index, name or channel object.
		
		Args:
			channel (str): Some channel.
		
		Returns:
			str: Path to h5 file.
		
		"""
		
		#Get channel
		channel=self.getChannel(channel)
		
		#Check if eveything is ready
		if len(channel.tifFn)==0:
			printError("You have not set a tif filepath for channel "+channel.name+" yet. Run saveChannelToTif first.")
			return
		if len(channel.classifier)==0:
			printError("You have not set a classifier filepath for channel "+channel.name+" yet. ")
			return
		
		#Run ilastik
		probFiles=ilm.runIlastik([channel.tifFn],classFile=channel.classifier)
		channel.h5Fn=probFiles[0]
		
		return probFiles[0]
	
	def getChannel(self,channel):
		
		if type(channel)==str:
			return self.getChannelByIName(channel)
		elif type(channel)==int:
			return self.getChannelNameByIdx(channel)
		else:
			return channel
	
	def createChannelSignalProfile(self,channel,maskChannel,debug=False):
		
		"""Creates signal profile for given channel.
		
		Args:
			channel (str): Name of channel for which signal profiles should be created. 
			maskChannel (str): Name of channel used for masking. 
		
		Keyword Args:
			debug (bool): Print debugging messages.
		
		
		"""
		
		channel=self.getChannel(channel)
		angles,signals=channel.createSignalProfile(maskChannel,debug=False)
		
		return angles,signals
	
	def getChannelNames(self):
		
		"""Returns list of names of all channels.
		"""
		
		names=[]
		for channel in self.channels:
			names.append(channel.name)
			
		return names	
	
	
	
class channel:
	
	def __init__(self,exp,name,idx,classifier="",h5Fn="",tifFn=""):
		
		self.exp=exp
		self.name=name
		self.classifier=classifier
		self.h5Fn=h5Fn
		self.tifFn=tifFn
		self.idx=idx
		
		self.angles=None
		self.anglesAligned=None
		self.signals=None
		self.signalsAligned=None
	
		self.setDefaultOpts()
		
	def setDefaultOpts(self):
		
		"""Sets analysis options back to default."""
		
		#ilastik probabilities
		self.probThresh=0.5
		self.probIdx=0
		
		#Projection
		self.proj='max'
		
		#Binning
		self.bins=50
		
		#Background filtering
		self.bkgd=None
		
		#Minimum mask pixel requirement
		self.minPix=10000
		
		#Denoising via median filter
		self.median=None
		
		#Norm by Dapi channel
		self.norm=True
		
		
		
	def applyMedianFilter(self,img=None):
		
		"""Applies median filter to image data of channel.
		
		If no image data is given, will use ``img=self.exp.image``.
		
		Keyword Args:
			img (numpy.ndarray): Image data. 
		
		Returns:
			numpy.ndarray: Modified image.
		"""
		
		if img==None:
			img=self.exp.image.copy()
		
		img[self.idx]=am.medianFilter(img[self.idx])
		return img
	
	def zProjection(self,img=None,proj=None,axis=1):
		
		"""Performs z-Projection on image data of channel.
		
		If no image data is given, will use ``img=self.exp.image``.
		
		If ``proj=None``, will not perform any projection.
		
		Available projections:
		
			* ``proj=max``: Maximum intensity.
			* ``proj=sum``: Sum intensity.
			* ``proj=mean``: Mean intensity.
			
		Keyword Args:
			img (numpy.ndarray): Image data. 
			proj (str): Type of projection.
			axis (int): Axis to perform projection along.
			
		Returns:
			numpy.ndarray: Modified image.
		"""
		
		
		if img==None:
			img=self.exp.image.copy()
		
		#Perform projections if selected
		if proj!=None:
		
			if proj=="max":
				img=im.maxIntProj(img,axis)
			elif proj=="sum":
				img=im.sumIntProj(mask,axis)
			elif proj=="mean":
				img=im.meanIntProj(img,axis)
				
		#Add another axis so we have a fake zstack
		img= img[np.newaxis,:]		
		
		return img
	
	def maskImg(self,maskChannel,img=None):
		
		"""Masks image data of channel.
		
		If no image data is given, will use ``img=self.exp.image``.
		
		Keyword Args:
			maskChannel (str): Name of channel used for masking.
		
		Returns:
			numpy.ndarray: Masked image.
		
		"""
		
		if img==None:
			img=self.exp.image.copy()
		
		#Get channel
		maskChannel=self.exp.getChannel(maskChannel)
		
		#Mask from h5 file
		mask,maskedImg=am.maskImgFromH5(maskChannel.h5Fn,img,probIdx=maskChannel.probIdx,
			       probThresh=maskChannel.probThresh,channel=self.idx)
				
		return mask, maskedImg
	
	def createSignalProfile(self,maskChannel,debug=False):
		
		"""Creates signal profile for channel.
		
		Args:
			maskChannel (str): Name of channel used for masking. 
		
		Keyword Args:
			debug (bool): Print debugging messages.
			
		Returns:
			tuple: Tuple containing: 
			
				* angles (numpy.ndarray): List of angle arrays.
				* signals (numpy.ndarray): List of signal arrays.
				
		"""
	
		#Get channel used for masking
		maskChannel=self.exp.getChannel(maskChannel)
		
		#Make local copy of original image
		img=self.exp.image.copy()
		
		#Apply median filter
		if self.median!=None:
			img=self.applyMedianFilter(img=img,radius=self.median)
			img=maskChannel.applyMedianFilter(img=img,radius=self.median)
			
		#Norm by mask channel 
		if self.norm:
			img=am.normImg(img,self.idx,normChannel=maskChannel.idx)
		
		#Mask image
		mask, maskedImg=self.maskImg(maskChannel,img=img)
		
		#Minimum pixel in mask that need to be 1.
		if np.nansum(mask)<self.minPix:
			if debug:
				printWarning("Could not create signal profile for channel "+self.name+" of experiment "+ self.exp.name
		 +". Minimum pixels requirement not fullfilled.")
			return None,None
		
		#Projection
		maskedImg=self.zProjection(img=maskedImg,proj=self.proj,axis=0)
		mask=self.zProjection(img=mask,proj=self.proj,axis=0)
		
		#Create profile
		angles,signals=am.createSignalProfile(maskedImg,mask,img,bins=self.bins,bkgd=self.bkgd,debug=debug)
		
		self.angles=angles
		self.signals=signals
		
		return angles,signals
	
	def setOpts(self,debug=False,**kwargs):
		
		"""Sets options."""
		
		for kwarg in kwargs:
			if not hasattr(self,kwarg):
				printWarning("Cannot set attribute "+kwarg+" . Channel has not this property." )
			else:
				if debug:
					printNote("Going to set "+kwarg+" from " + str(getattr(self,kwarg)) + " to "+ kwargs[kwarg])
				setattr(self,kwarg,kwargs[kwarg])

		
	def findCorrespondingH5File(self):
		
		"""Finds corresponding h5 files."""
		
		if not os.path.exists(self.tifFn):
			printError("Cannot find corresponding h5 file for channel "+ self.name+". Tifffile does not exist.")
			return
		
		files=os.listdir(os.path.dirname(self.tifFn))
		
		for f in files:
			if os.path.basename(self.tifFn).replace(".tif","") in f and f.endswith(".h5"):
				self.h5Fn=os.path.dirname(self.tifFn)+"/"+f
				return
			
		printWarning("Cannot find corresponding h5 file to " + self.tifFn)
		return
				
	def plotSignal(self,ax=None,color='r',lw=1.,title=""):
		
		"""Plots signal along radial axis.
		
		.. note:: Will create new axes if not specified.
			
		Keyword Args:
			ax (matplotlib.axes): Axes used for plotting.
			color (str): Color of plot.
			lw (float): Linewidth of plot.
			title (str): Title of plot.
			
		Returns:
			matplotlib.axes: Axes used for plotting.
		"""
		
		if ax==None:
			fig,axes=pm.makeAxes([1,1])
			ax=axes[0]
		
		for i in range(len(self.angles)):
			ax.plot(self.angles[i],self.signals[i],color=color,lw=lw)
		plt.draw()
		
		ax=pm.turnAxesForPub(ax,figWidthPt=500)
		ax=pm.turnIntoRadialPlot(ax)
		
		ax.set_title(title)
		
		return ax	
			
	def plotAlignedSignal(self,ax=None,color='r',lw=1.,title=""):

		"""Plots aligned signal along radial axis.
		
		.. note:: Will create new axes if not specified.
			
		Keyword Args:
			ax (matplotlib.axes): Axes used for plotting.
			color (str): Color of plot.
			lw (float): Linewidth of plot.
			title (str): Title of plot.
			
		Returns:
			matplotlib.axes: Axes used for plotting.
		"""
		

		if ax==None:
			fig,axes=pm.makeAxes([1,1])
			ax=axes[0]
		
		for i in range(len(self.anglesAligned)):
			ax.plot(self.anglesAligned[i],self.signalsAligned[i],color=color,lw=lw)
		plt.draw()
		
		ax=pm.turnAxesForPub(ax,figWidthPt=500)
		ax=pm.turnIntoRadialPlot(ax)
		
		ax.set_title(title)
		
		return ax
	
	def turnDoublePeak(self,anglesDist,debug=False,centerInMiddle=True):
		
		"""Checks if signal has 2 distinct peaks in profiles, if so, flips
		them such that they are aligned.
		
		Args:
			angleDist (float): Angle in radians that peaks need to be apart.
			
		Keyword Args:
			debug (bool): Print debugging messages.
			centerInMiddle (bool): Move valley between peaks to 0.
			
		Returns:
			numpy.ndarray: Newly aligned signal array.
		
		"""
		
		signals=am.turnDoublePeak(self.anglesAligned,self.signalsAligned,angleDist,debug=debug,centerInMiddle=centerInMiddle)
		self.signalsAligned=signals
		
		return self.signalsAligned
	
	def hasDoublePeak(self,angleDist,debug=False):
		
		"""Checks if signals has 2 distinct peaks in profiles.
	
		Args:
			angleDist (float): Angle in radians that peaks need to be apart.
			
		Keyword Args:
			debug (bool): Print debugging messages.
			
		Returns:
			bool: True if double peak.	
		
		"""
		
		return hasDoublePeak(self.anglesAligned,self.signalsAligned,angleDist,debug=debug)
	
	

	