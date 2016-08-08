#===========================================================================================================================================================================
#Module description
#===========================================================================================================================================================================

"""Module for QuantDorsal for interacting with ilastik.

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

#System
import sys
import inspect
import os

#Matplotlib
import matplotlib.pyplot as plt

#QuantDorsal
from term_module import *

#===========================================================================================================================================================================
#Module Functions
#===========================================================================================================================================================================

def runIlastik(fnRawFolder,fnOut,classFile="classifiers/quantDorsalDefault.ilp",channel=0,exportImg=False):
	
	"""Runs ilastik on all files in specific folder.
	
	Calls ilastik in headless mode using ``os.system``.
	
	Args:
		fnRawFolder (str): Folder containing raw tiff files.
		fnOut (str): Folder where to put output.
	
	Keyword Args:
		classFile (str): Path to classifier file.
		channel (int): Which channel to mask.
		exportImg (bool): Export prediction images.
		
	Returns:
		bool: True if success.
		
	"""
	
	ilastikPath=getIlastikBin()
	
	run_ilastik.sh --headless --project=classifiers/Dorsal_Dapi_alex2.ilp "../data/tifs/*_c0*.tif"

	regExData='"'+fnRawFolder+"*_c"+str(channel) +"*.tif"+'"'
	#+ ' --raw_data '

	cmd = ilastikPath + " --headless" + " --project=" +classFile 
	
	if exportImg:
		cmd = cmd + " --export_object_prediction_img --export_object_probability_img  "
	
	cmd = cmd + " --output_internal_path " + fnOut
	
	cmd=cmd+" " + regExData
	
	printNote("About to execute:")
	print cmd
	
	ret=os.system(cmd)
	
	return ret
	
def getConfDir():
	
	"""Returns path to configurations directory."""
	
	modulePath=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
	path=modulePath+"/configurations"+"/"
	return path

def getPathFile():
	
	"""Returns path to paths file."""
	
	return getConfDir()+'paths'

def getIlastikBin(fnPath=None):
	
	"""Returns path to ilastik binary."""
	
	return getPath('ilastikBin',fnPath=fnPath)


def getPath(identifier,fnPath=None):
	
	"""Extracts path with identifier from path definition file.
	
	If not defined diferently, will first look in configurations/paths,
	then configurations/paths.default.
	
	Args:
		identifier (str): Identifier of path
		
	Keyword Args:
		fnPath (str): Path to path definition file
			
	Returns:
		str: Path

	"""
	
	if fnPath==None:
		fnPath=getPathFile()
	else:
		if not os.path.isfile(fnPath):
			printWarning(fnPath+" does not exist. Will continue with paths defined in default paths files.")
			fnPath=getPathFile()
		
	path=None
	
	with  open (fnPath,'rb') as f:
		for line in f:
			if line.strip().startswith(identifier):
				ident,path=line.split('=')
				path=path.strip()
				break
		
	if path==None:
		printWarning("There is no line starting with ", identifier+"= in ", fnPath, ".")
		fnPath=getPathFile()+'.default'
		path=getPath(identifier,fnPath=fnPath)
		
	path=os.path.expanduser(path)
	
	return path

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
	
def readH5(fn):	
	
	"""Reads h5fs data file.
	
	Args:
		fn (str): Path to h5 file.
	
	Returns:
		
	
	"""
	
	
	with h5py.File('data.h5','r') as hf:
		print('List of arrays in this file: \n', hf.keys())
		data = hf.get('dataset_1')
		np_data = np.array(data)
		print('Shape of the array dataset_1: \n', np_data.shape)
		
	return data	