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

#Bioformats
import javabridge
import bioformats

#System
import sys

#Matplotlib
import matplotlib.pyplot as plt



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
	img=scaleToEnc(img,enc,maxVal=maxVal)
	skimage.io.imsave(fn,img)
	
	return fn

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

def saveStack(fnOut,images,prefix=""):
	
	"""Writes stack to tiff files.
	
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
	for i in range(images.shape[0]):
		for j in range(images.shape[1]):
			for k in range(images.shape[2]):
				
				fnSave=prefix+"_series"+getEnumStr(i)+"_c"+getEnumStr(j)+"_z"+getEnumStr(k)+".tif"
				saveImg(images[i,j,k,:,:],fnSave,scale=False)
				


def getEnumStr(num,maxNum,L=None):
	
	"""Returns enumeration string.
	"""
	
	if L==None:
		L=len(maxNum)
	
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
	
	"""Finds stacks that are somehow corrupted."""
		
	problematicStacks=[]
	
	#Loop through all channels and zstacks
	for j in range(meta.image(imageIdx).Pixels.SizeC):
		for k in range(meta.image(imageIdx).Pixels.SizeZ):
			try:
				img=reader.read(series=imageIdx, c=j, z=k, rescale=False)
			except:
				if debug:
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


def fitEllipse(x,y):
	
	"""Fits ellipse to x,y coordinates, similiar to matlabs fitEllipse.
	

	"""
	
	x = x[:,np.newaxis]
	y = y[:,np.newaxis]
	D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
	S = np.dot(D.T,D)
	C = np.zeros([6,6])
	C[0,2] = C[2,0] = 2; C[1,1] = -1
	E, V =  eig(np.dot(inv(S), C))
	n = np.argmax(np.abs(E))
	a = V[:,n]
	
	return a


def ellipseCenter(a):
	
	"""Returns center of ellipse."""
	
	b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
	num = b*b-a*c
	x0=(c*d-b*f)/num
	y0=(a*f-b*d)/num
	return np.array([x0,y0])


def ellipseAngleOfRotation(a):
	
	"""Returns anglular rotation of ellipse."""
	
	b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
	return 0.5*np.arctan(2*b/(a-c))


def ellipseAxisLength(a):
	
	"""Returns length of axis of ellipse."""
	
	b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
	up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
	down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
	down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
	
	print down1,down2
	
	res1=np.sqrt(up/down1)
	res2=np.sqrt(up/down2)
	return np.array([res1, res2])

def decodeEllipse(ell):

	center=ellipseCenter(ell)
	lengths=ellipseAxisLength(ell)
	rot=ellipseAngleOfRotation(ell)
	
	return center,lengths,rot

def ellipseToArray(center,lengths,alpha,steps=200):
	
	"""Return x/y vector given the center,lengths and rotation of 
	an ellipse.
	
	Args:
		center (list): Center of ellipse.
		lengths (list): Lengths of ellipse.
		alpha (float): Rotation angle of ellipse.
	
	Keyword Args:
		steps (int): Number of x/y points returned.
		
	Returns:
		tuple: Tuple containing:
			
			* x (numpy.ndarray): x-coordinates.
			* y (numpy.ndarray): y-coordinates.
			
	"""
	
	t=np.linspace(-np.pi,np.pi,steps)
	
	a,b=lengths
	
	x=center[0]+a*np.cos(t)*np.cos(alpha)-b*np.sin(t)*np.sin(alpha)
	y=center[1]+a*np.cos(t)*np.sin(alpha)+b*np.sin(t)*np.cos(alpha)
	
	return x,y

def multGauss(x, *params):
    
	""" Function of multiple gaussian distribution 
		
	check: http://stackoverflow.com/questions/26902283/fit-multiple-gaussians-to-the-data-in-python
	"""
	
	y = np.zeros_like(x)
	for i in range(0, len(params), 3):
		ctr = params[i]
		amp = params[i+1]
		wid = params[i+2]
		y = y + amp * np.exp( -((x - ctr)/wid)**2)
	return y
	
def alignDorsal(x,intensity,dorsal=0,phase=0,method='maxIntensity',opt=None):

    
	"""Align the dorsal ventral intensity data with the ventral at 'phase'.
	
	Args: 
		x (numpy.array): 1D-array of angles corresponding to the intensity data
		intensity (numpy.array): 2D-array of intensity data for different channels (with the first row the dorsal signal)

	Keyword Args:
		dorsal (int): indicates the row of dorsal signal in 'intensity'
		phase (double): The phase in [-pi,pi] that the ventral center shift to
		method (str): different methods to determine the ventral center
		
			* 'maxIntensity': pick out the ventral center with the point with maximal dorsal signal;
			* 'UI': user indicated point, use 'opt' to indicate the position;
			* 'Illastik': indicated by Illastik, use 'opt' to indicate the position;
			* 'Gaussian': fit the profile with Gaussian distributions, the center is indicated by the mean of the Gaussian fit, use 'opt' to indicates how many number of Gaussian distribution to fit.
		
		opt (double or int): see method for using
		
	Returns:
		tuple: Tuple containing:
		
			* phi (numpy.array): phase from -pi to pi
			* alignInt (numpy.array): aligned intensity
            
	"""

	# size of the data

	nx = x.size
	n1,n2 = intensity.shape # n1 for number of colors, n2 resolution
	dx = x[2]-x[1]  # inteval
	phi = dx*(np.arange(nx)-np.floor(nx/2))  # odd number of data points, 0 at center, even number, 0 at nx/2
	id0 = np.mod(np.floor((nx)/2)+int(round(phase/dx)),nx)  # the position of the peak
	
	# find the dorsal intensity
	
	dosInt = intensity[dorsal,:]
	
	# find the center with different methods
	
	if method=='maxIntensity':
		
		shift = int(id0-np.argmax(dosInt))
    
	# elif method=='UI':  due to Alex
    
	elif method=='Illastik':
		
		if opt==None:
			opt = 0     # not indicated
		
		shift = int(id0-np.argmin(np.absolute(x-opt)))

	elif method=='Gaussian':
    
		if opt==None:
			opt = 1     # fit with one Gaussian peak
        
		guess = [0,np.amax(dosInt),1]
		if opt>1:
			for i in range(1,opt):
				guess += [i*2*np.pi/opt-np.pi,np.amax(dosInt)/opt,1]
        
		popt, pcov = curve_fit(multGauss, x, dosInt, p0=guess)
        
		c0 = np.array(popt[0::3])
		w0 = np.array(popt[1::3])
		cmax = c0[np.argmax(w0)]   # center of the strongest peak
		while cmax>np.pi:    # addjust to -pi to pi
			cmax = cmax-2*np.pi
		while cmax<-np.pi:
			cmax = cmax+2*np.pi
    
		c1 = np.zeros_like(c0)
		for i in range(opt):
		ctemp = np.array([c0[i]-cmax,c0[i]-cmax+2*np.pi,c0[i]-cmax-2*np.pi])
		c1[i] = cmax+ctemp[np.argmin(np.fabs(ctemp))]
        
		shift = int(id0-int(round(np.average(c1, weights=w0)/dx))-np.floor(nx/2)) # weigted average of the Gaussian centers

	# roll the array

	#if n1>1:
	alignInt = np.roll(intensity, shift, axis=1)
	#else:
	#        alignInt = np.roll(intensity, shift)

	# return the variables

	return phi, alignInt
