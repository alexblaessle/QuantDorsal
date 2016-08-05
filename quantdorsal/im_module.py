#===========================================================================================================================================================================
#Importing necessary modules
#===========================================================================================================================================================================

#Numpy/Scipy
import numpy as np
from numpy.linalg import eig, inv

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
	
	t=np.linspace(-np.pi,np.pi,steps)
	
	a,b=lengths
	
	x=center[0]+a*np.cos(t)*np.cos(alpha)-b*np.sin(t)*np.sin(alpha)
	y=center[1]+a*np.cos(t)*np.sin(alpha)+b*np.sin(t)*np.cos(alpha)
	
	return x,y
	
	