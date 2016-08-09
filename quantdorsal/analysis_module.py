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

#System
import sys

#Matplotlib
import matplotlib.pyplot as plt

#QuantDorsal
from term_module import *

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

def invEllipse(x,y,center,lengths,alpha):
	
	#Move to center
	X=(y-center[1])
	Y=(x-center[0])
	
	x1=(Y*np.cos(alpha)-X*np.sin(alpha))*lengths[0]
	x2=(Y*np.sin(alpha)+X*np.cos(alpha))*lengths[1]

	t=np.arctan2(x1,x2)
	
	return t

def maskImg(img,mask,channel):
	
	"""Applies mask to image in specific channel. 
	"""
	
	img=img[channel]
	
	return mask*img

