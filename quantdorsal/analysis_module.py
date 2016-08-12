#===========================================================================================================================================================================
#Module description
#===========================================================================================================================================================================

"""QuantDorsal module for gradient analysis. 

Contains functions for 

"""

#===========================================================================================================================================================================
#Importing necessary modules
#===========================================================================================================================================================================

#Numpy/Scipy
import numpy as np
from numpy.linalg import eig, inv
from scipy.optimize import curve_fit # for fitting gaussian
import scipy.signal as spsig

#System
import sys

#Matplotlib
import matplotlib.pyplot as plt

#QuantDorsal
from term_module import *
import ilastik_module as ilm
import im_module as im


#===========================================================================================================================================================================
#Module Functions
#===========================================================================================================================================================================

def getCenterOfMass(x,y,masses=None):
	
	r"""Computes center of mass of a given set of points.
	
	.. note:: If ``masses==None``, then all points are assigned :math:`m_i=1`.
	
	Center of mass is computed by:
	
	.. math:: C=\frac{1}{M}\sum\limits_i m_i (x_i,y_i)^T
	
	where 
	
	.. math:: M = \sum\limits_i m_i
	
	Args:
		x (numpy.ndarray): x-coordinates.
		y (numpy.ndarray): y-coordinates.
		
	Keyword Args:
		masses (numpy.ndarray): List of masses.
	
	Returns:
		numpy.ndarray: Center of mass.
	
	"""
	
	if masses==None:
		masses=np.ones(x.shape)
		
	centerOfMassX=1/(sum(massses))*sum(masses*x)
	centerOfMassY=1/(sum(massses))*sum(masses*y)
	
	centerOfMass=np.array([centerOfMassX,centerOfMassY])
	
	return centerOfMass

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

def invEllipse(x,y,center,lengths,alpha):
		
	r"""Inverses ellipse in parametric form.
	
	Centers ellipse, rotates it back using inverse of rotational 
	matrix and then computes angle.
	
	Args:
		x (numpy.ndarray): x-coordinates.
		y (numpy.ndarray): y-coordinates.
		center (list): Center of ellipse.
		lengths (list): Lengths of ellipse.
		alpha (float): Rotation angle of ellipse.
		
	Returns:
		numpy.ndarray: List of angles corresponding to points.
		
	"""
		
	#Move to center
	X=(y-center[1])
	Y=(x-center[0])
	
	#Turn back using inverse rotation matrix
	XRot=(Y*np.cos(alpha)-X*np.sin(alpha))
	YRot=(Y*np.sin(alpha)+X*np.cos(alpha))
	
	#Multiply with a/b
	XRot=XRot*lengths[0]
	YRot=YRot*lengths[1]

	#Compute angle using tan^-1
	t=np.arctan2(XRot,YRot)
	
	return t

def maskImg(img,mask,channel):
	
	"""Applies mask to image in specific channel. 
	
	Args:
		img (numpy.ndarray): Image stack.
		mask (numpy.ndarray): Mask.
	
	Keyword Args:	
		channel (int): Index of channel with signal.
		
	Returns:
		numpy.ndarray: Masked Image.
	
	"""
	
	img=img[channel]
		
	return mask*img

def maskImgFromH5(fn,img,probIdx=0,probThresh=0.8,channel=1):
	
	"""Reads h5 files and produces binary mask, then masks channel of
	image.
	
	Args:
		img (numpy.ndarray): Image stack.
		fn (str): Path to h5 file.
	
	Keyword Args:	
		channel (int): Index of channel with signal.
		probThresh (float): Probability threshhold used.
		probIdx (int): Index of which label in h5 file to be used.
	
	Returns:
		tuple: Tuple containing:
			
			* mask (numpy.ndarray): Mask.
			* maskedImg (numpy.ndarray): Masked Image.
			
	"""
	
	#Load data
	data=ilm.readH5(fn)
	
	#Extract values for right label
	data=data[:,:,:,probIdx]
	
	#Make mask
	mask=ilm.makeProbMask(data,thresh=probThresh)
	
	#Mask img
	maskedImg=maskImg(img,mask,channel)
	
	return mask,maskedImg

def fitEllipseToMask(mask):
	
	"""Fits ellipse to mask.
	
	.. note:: Will consider mask values greater than 0.5 .
	
	Args:
		mask (numpy.ndarray): A Mask.
		
	Returns:
		tuple: Tuple containing:
		
			* center (list): Center of ellipse.
			* lengths (list): Lengths of ellipse.
			* rot (float): Rotation angle of ellipse.
			* x (numpy.ndarray): x-coordinates of mask.
			* y (numpy.ndarray): y-coordinates of mask.
			* xEll (numpy.ndarray): x-coordinates of ellipse.
			* yEll (numpy.ndarray): y-coordinates of ellipse.
			
			
	"""
	
	#Build coordinate grid
	x=np.arange(mask.shape[1])
	y=np.arange(mask.shape[0])
	X,Y=np.meshgrid(x,y)

	#Extract coordinates that are 1
	x=X[np.where(mask>0.5)].flatten()
	y=Y[np.where(mask>0.5)].flatten()
	
	#Fit ellipse
	ell=fitEllipse(x,y)
	center,lengths,rot=decodeEllipse(ell)
	
	xEll,yEll=ellipseToArray(center,lengths,rot,steps=200)
	
	return center, lengths, rot,x,y,xEll,yEll

def normImg(img,signalChannel,normChannel=0,offset=0.01):
	
	"""Norms data in channel signalChannel by
	the one in normChannel."""
	
	img[signalChannel]=(img[signalChannel]+offset)/(img[normChannel]+offset)
	
	return img

def createSignalProfileFromH5(img,fn,signalChannel=2,probThresh=0.8,probIdx=0,proj=None,bins=None,bkgd=None,norm=True,minPix=0,median=None,debug=False):
	
	"""Builds angular signal distribution using h5 file as input.
	
	Does the following:
	
		* Reads in h5 file, creates mask.
		* Masks image.
		* Fits ellipse to mask.
		* Creates angular distribution profile.
	
	There are multiple projections available:
	
		* ``proj=None``: No projection.
		* ``proj="max"``: Maximum intensity projection.
		* ``proj="sum"``: Sum intensity projection.
		* ``proj="mean"``: Mean intensity projection.
	
	If ``bins!=None``, will perform angular binning with ``bins`` bins.
	
	If ``bkgd!=None``, will filter background with threshhold ``bkgd``.
	
	If the mask includes less than ``minPix`` pixels, will abort and turn ``None``.
	
	Args:
		img (numpy.ndarray): Image stack.
		fn (str): Path to h5 file.
		
	Keyword Args:	
		signalChannel (int): Index of channel with signal.
		probThresh (float): Probability threshhold used.
		probIdx (int): Index of which label in h5 file to be used.
		maxInt (bool): Perform maximum intensity projection of stack.
		bins (int): Number of angular bins.
		bkgd (float): Background intensity.
		debug (bool): Show debugging plots.
		norm (bool): Norm by dapi channel.
		minPix (int): Minimum number of pixels that need to be included in mask.
		median (int): Turn on median filter specific radius.		

	Returns:
		tuple: Tuple containing:
		
			* angles (list): List of angle arrays.
			* signals (list): List of signal arrays.
	
	"""
	
	#Median filter
	if median!=None:
		img[0]=medianFilter(img[0])	
		img[signalChannel]=medianFilter(img[signalChannel])

	#Norm 
	if norm:
		img=normImg(img,signalChannel,normChannel=0)
		
	#Make mask
	mask,maskedImg=maskImgFromH5(fn,img,probIdx=probIdx,probThresh=probThresh,channel=signalChannel)
	
	#Minimum pixel in mask that need to be 1.
	if np.nansum(mask)<minPix:
		return None,None
	
	#Perform projections if selected
	if proj!=None:
		
		if proj=="max":
			mask=im.maxIntProj(mask,0)
			maskedImg=im.maxIntProj(maskedImg,0)
		elif proj=="sum":
			mask=im.sumIntProj(mask,0)
			maskedImg=im.sumIntProj(maskedImg,0)	
		elif proj=="mean":
			mask=im.meanIntProj(mask,0)
			maskedImg=im.meanIntProj(maskedImg,0)
		
		#Add another axis so we have a fake zstack
		mask= mask[np.newaxis,:]		
		maskedImg= maskedImg[np.newaxis,:]
	
	#Get signal profile
	angles,signals=createSignalProfile(maskedImg,mask,img,bins=bins,bkgd=bkgd,debug=debug)
	
	return angles,signals
	
def createSignalProfile(maskedImg,mask,img,signalChannel=1,bins=None,bkgd=None,debug=False):
	
	"""Builds angular signal distribution given a masked Image and its mask.
	
	Does the following:
		* Fits ellipse to mask.
		* Creates angular distribution profile.
	
	If ``bins!=None``, will perform angular binning with ``bins`` bins.
	
	If ``bkgd!=None``, will filter background with threshhold ``bkgd``.
	
	Args:
		maskedImg (numpy.ndarray): Masked image stack.
		mask (numpy.ndarray): Mask  stack.
		img (numpy.ndarray): Image stack.
		fn (str): Path to h5 file.
		
	Keyword Args:	
		signalChannel (int): Index of channel with signal.
		probThresh (float): Probability threshhold used.
		probIdx (int): Index of which label in h5 file to be used.
		bins (int): Number of angular bins.
		bkgd (float): Background intensity.
		debug (bool): Show debugging plots.
	
	Returns:
		tuple: Tuple containing:
		
			* angles (list): List of angle arrays.
			* signals (list): List of signal arrays.
	
	"""
	
	angles=[]
	signals=[]
	
	for i in range(mask.shape[0]):
		
		#Fit ellipse
		center,lengths,rot,x,y,xEll,yEll=fitEllipseToMask(mask[i])
		
		#Get values of signal in shape of x,y
		signal=maskedImg[i][np.where(mask[i]>0.5)].flatten()
		
		#Make sure that there is no NaN in img
		signal[np.isnan(signal)]=0

		#Compute angles
		t=invEllipse(x,y,center,lengths,rot)
		
		#Sort by angle
		t,signal=sortByAngle(t,signal)
		
		#If maskZero is selected, pop bkgd intensity 
		if bkgd!=None:
			t=t[np.where(signal>bkgd)[0]]
			signal=signal[np.where(signal>bkgd)[0]]
			
		#Bin if selected
		if bins!=None:
			t,signal=binData(t,signal,bins)
			
		#Append
		signals.append(signal)
		angles.append(t)
		
		if debug:
			showProfileDebugPlots(img,mask,maskedImg,i,signal,t,xEll,yEll,signalChannel,axes=None)
			
	return angles,signals	

def showProfileDebugPlots(img,mask,maskedImg,idx,signal,angle,xEll,yEll,channelIdx,axes=None):
	
	"""Shows some debugging plots."""
	
	fig, axes = plt.subplots(3, 2)
	
	fig.show()
	axes[0,0].imshow(img[0,idx])
	axes[0,0].set_title("Dapi Channel")
	axes[0,1].imshow(img[1,channelIdx])
	axes[1,0].imshow(mask[idx])
	axes[1,1].imshow(maskedImg[idx])
	axes[1,0].plot(xEll,yEll,'g')
	axes[2,0].imshow(maskedImg[idx])
	axes[2,1].plot(angle,signal,'r')
	
	plt.draw()
	raw_input()
	
	return axes
	
def sortByAngle(angle,signal):
	
	"""Sorts angle and signal vector by angle.
	
	Args:
		angle (numpy.ndarray): Angle array.
		signal (numpy.ndarray): Signal array.
	
	Returns:
		tuple: Tuple containing:
		
			* angle (numpy.ndarray): Angle array.
			* signal (numpy.ndarray): Signal array.
		
	"""
	
	idx = angle.argsort()
	angle = angle[idx[::-1]]
	signal = signal[idx[::-1]]

	return angle,signal

def getStats(signals):
	
	"""Computes mean and standard deviation for 
	a list of signal arrays:

	Args:
		signals (list): List of signal arrays.

	Returns:
		tuple: Tuple containing:
		
			* meanSignal (numpy.ndarray): Mean signal.
			* stdSignal (numpy.ndarray): Standard deviation of signal. 

	"""
	
	
	
	signals=np.asarray(signals)
	stdSignal=np.std(signals,axis=0)
	meanSignal=np.mean(signals,axis=0)

	return meanSignal,stdSignal

def medianFilter(img,radius=5):

	"""Applies median filter to image.
	
	.. note:: Radius needs to be odd.

	Args:	
		img (numpy.ndarray): Some image.
	
	Keyword Args:
		radius (int): Radius of filter.

	Returns:
		numpy.ndarray: Filtered image.
	"""
	
	if len(img.shape)>2:

		for i in range(img.shape[0]):
			img[i]=spsig.medfilt(img[i],kernel_size=int(radius))
	else:		
		img=spsig.medfilt(img,kernel_size=int(radius))
	
	return img
	
def binData(x,y,bins):
		
	"""Bins data.
	
	Args:
		x (numpy.ndarray): x-data.
		y (numpy.ndarray): y-data.
		bins (int): Number of bins.
		
	Returns:
		tuple: Tuple containing:
		
			* xBin (numpy.ndarry): Mid-points of bins.
			* binMeans (numpy.ndarray): Binned data.
	
	"""	
	
	#Bin array
	bins=np.linspace(x.min(),x.max(),bins+1)

	#Find idxs
	idx=np.digitize(x,bins)
	binMeans = [y[idx == i].mean() for i in range(1, len(bins))]
	binMeans=np.asarray(binMeans)
	
	#Bin 
	xBin=np.diff(bins)/2.+bins[:-1]

	return xBin,binMeans

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
        
		guess = [0,0,np.amax(dosInt),1]
		if opt>1:
			for i in range(1,opt):
				guess += [i*2*np.pi/opt-np.pi,np.amax(dosInt)/opt,1]
        
		popt, pcov = curve_fit(multGauss, x, dosInt, p0=guess)
        
		c0 = np.array(popt[1::3])
		w0 = np.array(popt[2::3])
		cmax = c0[np.argmax(w0)]   # center of the strongest peak
		while cmax>np.pi:    # addjust to -pi to pi
			cmax = cmax-2*np.pi
		while cmax<-np.pi:
			cmax = cmax+2*np.pi
    
		c1 = np.zeros_like(c0)
		for i in range(opt):
			ctemp = np.array([c0[i]-cmax,c0[i]-cmax+2*np.pi,c0[i]-cmax-2*np.pi])
			c1[i] = cmax+ctemp[np.argmin(np.fabs(ctemp))]
		
		print np.shape(w0), np.shape(c1)
		
		shift = int(id0-int(round(np.average(c1, weights=w0)/dx))-np.floor(nx/2)) # weigted average of the Gaussian centers

	# roll the array

	#if n1>1:
	alignInt = np.roll(intensity, shift, axis=1)
	#else:
	#        alignInt = np.roll(intensity, shift)

	# return the variables

	return phi, alignInt
	
def multGauss(x, *params):
    
	""" Function of multiple gaussian distribution 
		
	check: http://stackoverflow.com/questions/26902283/fit-multiple-gaussians-to-the-data-in-python
	"""
	
	y = np.zeros_like(x)
	for i in range(1, len(params), 3):
		ctr = params[i]
		amp = params[i+1]
		wid = params[i+2]
		y = y + amp * np.exp( -((x - ctr)/wid)**2)

	y=params[0]+y
	return y	
