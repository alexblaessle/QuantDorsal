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


def runIlastik(fnImg,fnOut,classFile="classifiers/quantDorsalDefault.ilp"):
	
	ilastikPath=getIlastikBin()
	
	
	cmd = ""

def getConfDir():
	
	"""Returns path to configurations directory."""
	
	modulePath=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
	path=modulePath+"configurations"+"/"
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

