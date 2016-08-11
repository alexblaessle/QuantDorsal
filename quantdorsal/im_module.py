#===========================================================================================================================================================================
#Module description
#===========================================================================================================================================================================

"""QuantDorsal module for image analysis. 

Contains functions for 

	* bioformats images and meta data
	* reading tiff files
	* otsu threshholding
	* ellipse fitting
	* maximum intensity projection

"""

#===========================================================================================================================================================================
#Importing necessary modules
#===========================================================================================================================================================================

#Numpy/Scipy
import numpy as np
from numpy.linalg import eig, inv
from scipy.optimize import curve_fit # for fitting gaussian

#Scikit image
import skimage.io
import tifffile
	
#Bioformats
import javabridge
import bioformats

#System
import sys

#Matplotlib
import matplotlib.pyplot as plt

#QuantDorsal
from term_module import *

#===========================================================================================================================================================================
#Module Functions
#===========================================================================================================================================================================

def loadImg(fn,enc,dtype='float'):
	
	"""Loads image from filename fn with encoding enc and returns it as with given dtype.

	Args:
		fn (str): File path.
		enc (str): Image encoding, e.g. 'uint16'.
		dtype (str): Datatype of pixels of returned image.
	
	Returns:
		numpy.ndarray: Loaded image.
	"""
	
	#Load image
	img = skimage.io.imread(fn).astype(enc)
	
	#Getting img values
	img=img.real
	img=img.astype(dtype)
	
	return img

def saveImg(img,fn,enc="uint16",scale=True,maxVal=None):
	
	"""Saves image as tif file.
	
	``scale`` triggers the image to be scaled to either the maximum
	range of encoding or ``maxVal``. See also :py:func:`scaleToEnc`.
	
	Args:
		img (numpy.ndarray): Image to save.
		fn (str): Filename.
		
	Keyword Args:	
		enc (str): Encoding of image.
		scale (bool): Scale image.
		maxVal (int): Maximum value to which image is scaled.
	
	Returns:
		str: Filename.
	
	"""
	
	#Fill nan pixels with 0
	img=np.nan_to_num(img)
	
	#Scale img
	if scale:
		img=scaleToEnc(img,enc,maxVal=maxVal)
	skimage.io.imsave(fn,img)
	
	return fn

def saveSingleStackFile(img,fn,axes='ZYX'):
	
	"""Saves a stack to a single tiff file.
	
	Uses ``tifffile``.
	
	Args:
		img (numpy.ndarray): Image data.
		fn (str): Filepath.
	
	Keyword Args:
		axes (str): Axes specification.
		
	Returns:
		str: Output filename.
	
	"""
	
	#Define metadata
	metadata={'axes': axes}
	tifffile.imsave(fn, img, metadata=metadata)
	
	return fn

def saveImageSeriesToStacks(images,fnFolder,prefix="",axes='ZYX',channel=0,debug=True):
	
	"""Writes list of images into single stack tiff files.
	
	Args:
		fnFolder (str): Folder to write to.
		images (list): List of stacks.
		
	Keyword Args:
		prefix (str): Prefix to be put in the front of the filename.
		channel (int): Which channel to write.
		debug (bool): Print debugging messages.
	
	Returns:
		list: List of all the filesnames written.
	
	"""
	
	filesWritten=[]
	
	#Loop through all image series
	for i,img in enumerate(images):
		
		#Build save string
		fnSave=fnFolder+prefix+"_series"+getEnumStr(i,len(images))+"_c"+getEnumStr(channel,img.shape[0])+".tif"
		
		#Save to stack
		saveSingleStackFile(img[channel],fnSave,axes=axes)
		
		#Print out message
		if debug:
			print "Successfully wrote file "+ fnSave
		
		#Remeber filename
		filesWritten.append(fnSave)
	
	return filesWritten
			
def saveStackToFiles(fnOut,images,prefix=""):
	
	"""Writes stack to a list of tiff files.
	
	Args:
		fnOut (str): Folder to write to.
		images (list): List of stacks.
		
	Keyword Args:
		prefix (str): Prefix to be put in the front of the filename.
	
	Returns:
		list: List of all the filesnames written.
	
	"""

	filesWritten=[]
	
	#Loop through all datasetsm channels and zstacks
	for i in range(len(images)):
		for j in range(images[i].shape[0]):
			for k in range(images[i].shape[1]):
				
				
				
				fnSave=fnOut+prefix+"_series"+getEnumStr(i,len(images))+"_c"+getEnumStr(j,images[i].shape[0])+"_z"+getEnumStr(k,images[i].shape[1])+".tif"
				saveImg(images[i][j,k,:,:],fnSave,scale=False)
				
				filesWritten.append(fnSave)
				
	return filesWritten
				

def getEnumStr(num,maxNum,L=None):
	
	"""Returns enumeration string.
	
	Example:
	
	>>> getEnumStr(3,125)
	>>> "003"
	>>> getEnumStr(3,125,L=4)
	>>> "0003"
	
	Args:
		num (int): Number.
		maxNum (int): Largest number in enumeration.
	
	Keyword Args:
		L (int): Force length of string to L.
		
	Returns:
		str: Enumeration string.
	
	"""
	
	if L==None:
		L=len(str(maxNum))
	
	enumStr=(L-len(str(num)))*"0"+str(num)
	
	return enumStr
				
			
def readBioFormatsMeta(fn):
	
	"""Reads meta data out of bioformats format.
	
	.. note:: Changes system default encoding to UTF8.
	
	Args:
		fn (str): Path to file.
	
	Returns:
		OMEXML: meta data of all data.
	
	"""
	
	#Change system encoding to UTF 8
	reload(sys)  
	sys.setdefaultencoding('UTF8')

	#Load and convert to utf8
	meta=bioformats.get_omexml_metadata(path=fn)
	meta=meta.decode().encode('utf-8')
	
	meta2 = bioformats.OMEXML(meta)
	
	return meta2
	

def readBioFormats(fn,debug=True):
	
	"""Reads bioformats image file.
	
	Args:
		fn (str): Path to file.
	
	Keyword Args:
		debug (bool): Show debugging output.
	
	Returns:
		tuple: Tuple containing:
		
			* images (list): List of datasets
			* meta (OMEXML): meta data of all data.
	
	"""
	
	javabridge.start_vm(class_path=bioformats.JARS)
	
	meta=readBioFormatsMeta(fn)
	
	#Empty list to put datasets in	
	images=[]
	
	#Open with reader class
	with bioformats.ImageReader(fn) as reader:
		
		#Loop through all images
		for i in range(meta.image_count):
			
			channels=[]
			
			#Check if there is a corrupt zstack for this particular dataset
			problematicStacks=checkProblematicStacks(reader,meta,i,debug=debug)
			
			#Loop through all channels
			for j in range(meta.image(i).Pixels.SizeC):
				
				zStacks=[]
				
				#Loop through all zStacks
				for k in range(meta.image(i).Pixels.SizeZ):
					
					#Check if corrupted
					if k not in problematicStacks:
						img=reader.read(series=i, c=j, z=k, rescale=False)
						zStacks.append(img)
					
				#Append to channels
				channels.append(zStacks)
			
			#Append to list of datasets and converts it to numpy array
			images.append(np.asarray(channels))
	
	#Kill java VM
	javabridge.kill_vm()
	
	return images,meta

def checkProblematicStacks(reader,meta,imageIdx,debug=True):
	
	"""Finds stacks that are somehow corrupted.
	
	Does this by trying to read them via ``bioformats.reader.read``, and in case of 
	exceptions just adds them to a list of problematic stacks.
	
	Args:
		reader (bioformats.reader): A reader object.
		meta (OMEXML): Bioformats meta data object.
		imageIdx (int): Index of series to check.
		
	Keyword Args:
		debug (bool): Print debugging messages.
	
	Returns:
		list: List of indices of zstacks that are corrupted.
	
	"""
		
	problematicStacks=[]
	
	#Loop through all channels and zstacks
	for j in range(meta.image(imageIdx).Pixels.SizeC):
		for k in range(meta.image(imageIdx).Pixels.SizeZ):
			try:
				img=reader.read(series=imageIdx, c=j, z=k, rescale=False)
			except:
				if debug:
					printWarning("Loading failed.")
					print "Cannot load Image ", imageIdx,"/",meta.image_count
					print "channel = ",j, "/",meta.image(imageIdx).Pixels.SizeC
					print "zStack = ",k,"/",meta.image(imageIdx).Pixels.SizeZ
				problematicStacks.append(k)
	
	return list(np.unique(problematicStacks))		
	
def otsuImageJ(img,maxVal,minVal,debug=False,L=256):
	
	"""Python implementation of Fiji's Otsu algorithm. 
	
	See also http://imagej.nih.gov/ij/source/ij/process/AutoThresholder.java.

	Args:
		img (numpy.ndarray): Image as 2D-array.
		maxVal (int): Value assigned to pixels above threshhold.
		minVal (int): Value assigned to pixels below threshhold.
		
	Keyword Args:
		debug (bool): Show debugging outputs and plots.
		L (int): Number of bins used for histogram.
		
	Returns:
		tuple: Tuple containing:
		
			* kStar (int): Optimal threshhold
			* binImg (np.ndarray): Binary image
	"""
	
	#Initialize values
	#L = img.max()
	S = 0 
	N = 0
	
	#Compute histogram
	data,binEdges=np.histogram(img,bins=L)
	binWidth=np.diff(binEdges)[0]
	
	#Debugging plot for histogram
	if debug:
		binVec=arange(L)
		fig=plt.figure()
		fig.show()
		ax=fig.add_subplot(121)
		ax.bar(binVec,data)
		plt.draw()
			
	for k in range(L):
		#Total histogram intensity
		S = S+ k * data[k]
		#Total number of data points
		N = N + data[k]		
	
	#Temporary variables
	Sk = 0
	BCV = 0
	BCVmax=0
	kStar = 0
	
	#The entry for zero intensity
	N1 = data[0] 
	
	#Look at each possible threshold value,
	#calculate the between-class variance, and decide if it's a max
	for k in range (1,L-1): 
		#No need to check endpoints k = 0 or k = L-1
		Sk = Sk + k * data[k]
		N1 = N1 + data[k]

		#The float casting here is to avoid compiler warning about loss of precision and
		#will prevent overflow in the case of large saturated images
		denom = float(N1 * (N - N1)) 

		if denom != 0:
			#Float here is to avoid loss of precision when dividing
			num = ( float(N1) / float(N) ) * S - Sk 
			BCV = (num * num) / denom
		
		else:
			BCV = 0

		if BCV >= BCVmax: 
			#Assign the best threshold found so far
			BCVmax = BCV
			kStar = k
	
	kStar=binEdges[0]+kStar*binWidth
	
	#Now manipulate the image
	binImg=np.zeros(np.shape(img))
	for i in range(np.shape(img)[0]):
		for j in range(np.shape(img)[1]):
			if img[i,j]<=kStar:
				binImg[i,j]=minVal
			else:				
				binImg[i,j]=maxVal
	
	#Some debugging plots and output
	if debug:
		
		print "Optimal threshold = ", kStar
		print "#Pixels above threshold = ", sum(binImg)/float(maxVal)
		print "#Pixels below threshold = ", np.shape(img)[0]**2-sum(binImg)/float(maxVal)
		
		ax2=fig.add_subplot(122)
		ax2.contourf(binImg)
		plt.draw()
		raw_input()
			
	return kStar,binImg	

def maxIntProj(img,axis):
	
	"""Performs maximum intensity projection along 
	axis.
	
	Args:
		img (numpy.ndarray): Some image data.
		axis (int): Axis to perfrom projection along.
	
	Return:
		numpy.ndarray: Projection.
		
	"""

	#Maximum intensity projection
	proj=img.max(axis=axis)
	
	return proj

def sumIntProj(img,axis):
	
	"""Performs sum intensity projection along 
	axis.
	
	Args:
		img (numpy.ndarray): Some image data.
		axis (int): Axis to perfrom projection along.
	
	Return:
		numpy.ndarray: Projection.
		
	"""
	
	#Turn zeros into NaN
	img=maskZeroToNaN(img)
	
	#Sum intensity projection
	proj=np.nansum(img,axis=axis)
	
	#Resubstitute zeros into proj
	proj=maskNaNToZero(proj)
	
	return proj

def meanIntProj(img,axis):
	
	"""Performs mean intensity projection along 
	axis.
	
	Args:
		img (numpy.ndarray): Some image data.
		axis (int): Axis to perfrom projection along.
	
	Return:
		numpy.ndarray: Projection.
		
	"""
	
	#Turn zeros into NaN
	img=maskZeroToNaN(img)
	
	#Mean intensity projection
	proj=np.nanmean(img,axis=axis)
	
	#Resubstitute zeros into proj
	proj=maskNaNToZero(proj)
	
	return proj

def maskZeroToNaN(img):

	"""Replaces all zeros in image with NaN.
	
	"""
	
	img[np.where(img==0)]=np.nan
	
	return img

def maskNaNToZero(img):

	"""Replaces all NaNs in image with zeros.
	
	"""
	
	img[np.where(np.isnan(img))]=0.
	
	return img

def scaleToEnc(img,enc,maxVal=None):
	
	"""Scales image to either the maximum
	range of encoding or ``maxVal``.
	
	Possible encodings: 
	
		* 4bit
		* 8bit
		* 16bit
		* 32bit
	
	Args:
		img (numpy.ndarray): Image to save.
		enc (str): Encoding of image.
		
	Keyword Args:	
		maxVal (int): Maximum value to which image is scaled.
	
	Returns:
		numpy.ndarray: Scaled image.
	
	"""
	
	#Get maximum intensity of encoding
	if "16" in enc:
		maxValEnc=2**16-1
	elif "8" in enc:
		maxValEnc=2**8-1
	elif "4" in enc:
		maxValEnc=2**4-1
	elif "32" in enc:
		maxValEnc=2**32-1
	
	#See if maxVal is greater than maxValEnc
	if maxVal!=None:
		maxVal=min([maxVal,maxValEnc])
	else:
		maxVal=maxValEnc
		
	#Scale image
	factor=maxVal/img.max()
	img=img*factor

	#Convert to encoding
	img=img.astype(enc)
	
	return img