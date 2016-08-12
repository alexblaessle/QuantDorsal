#===========================================================================================================================================================================
#Module description
#===========================================================================================================================================================================

"""Module for QuantDorsal for interacting with ilastik.

Contains functions for 

	* Launching ilastik.
	* Path management.
	* h5 I/O.
	
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
import subprocess
import shlex

#H5
import h5py

#QuantDorsal
from term_module import *

#===========================================================================================================================================================================
#Module Functions
#===========================================================================================================================================================================

def runIlastik(files,fnOut=None,classFile="classifiers/quantDorsalDefault.ilp",channel=0,exportImg=False):
	
	"""Runs ilastik on all files in specified in files.
	
	Calls ilastik in headless mode using ``os.system``.
	
	Args:
		files (str): List of files to run ilastik on.
		
	Keyword Args:
		classFile (str): Path to classifier file.
		channel (int): Which channel to mask.
		exportImg (bool): Export prediction images.
		fnOut (str): Output filepath
		
	Returns:
		list: List of paths to output files.
		
	"""
	
	#Get ilastik binary
	ilastikPath=getIlastikBin()
	
	outFiles=[]
	
	for fn in files:
	
		#Build input data string
		regExData=" " + fn + " " 
		
		#Build basic command
		cmd = ilastikPath + " --headless" + " --project=" +classFile 
		
		#Some extra options for output
		if exportImg:
			cmd = cmd + " --export_object_prediction_img --export_object_probability_img  "
		
		if fnOut!=None:
			cmd = cmd + " --output_internal_path " + fnOut
			outFiles.append(fnOut)
		else: 
			outFiles.append(fn.replace(".tif","_Probabilities.h5"))
		
		#Add input data to command string
		cmd=cmd+" " + regExData
		
		#Print what we are about to do
		printNote("About to execute:")
		print cmd
		
		#Run command
		runCommand(cmd,fnStout=fn.replace(".tif",".stout"),fnSterr=fn.replace(".tif",".sterr"),redirect=True)
	
	return outFiles

def runCommand(cmd,fnStout="stout.txt",fnSterr="sterr",redirect=True):
	
	#Split command in list for subprocess
	args = shlex.split(cmd)
	
	#redirect stdout and stderr if selected
	if redirect:
		stoutFile = open(fnStout,'wb')
		sterrFile = open(fnSterr,'wb')
	else:	
		stoutFile = None
		sterrFile = None
		
	#Call via subprocess and wait till its done
	p = subprocess.Popen(args,stdout=stoutFile,stderr=sterrFile)
	p.wait()
	
	return 
	
	
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

def readH5(fn):
    
	""" Reads HDF5 data file
	
	Args:
		fn (str): Path to h5 file
		
	Returns:
		np_data: numpy array containing trained probabilities
		[z-stack, y-coord, x-coord,(0=background, 1=foreground)]
	"""
    
	with h5py.File(fn,'r') as hf:
		data = hf.get(hf.keys()[0])
		np_data = np.array(data)
		
	return np_data

def makeProbMask(array,thresh=0.8,debug=False):
    
	""" Makes a mask from the trained probabilities
	
	Args:
		array (nparray): probability data
		
	Keyword Args:
		thresh (float): Threshhold.
		
	Returns:
		mask (nparray): mask
	"""
	prob = np.copy(array)
	mask=np.zeros(prob.shape)
	mask[np.where(prob>thresh)]=1
	
	return mask

def getH5FilesFromFolder(fn):
	
	"""Returns paths to all h5 files given in a certain folder.
	
	Files will be sorted.
	
	Args:
		fn (str): Path to folder.
		
	Returns:
		list: List of files.
	
	"""
	
	files=os.listdir(fn)
	
	newFiles=[]
	for f in files:
		if f.endswith(".h5"):
			newFiles.append(fn+f)
	newFiles.sort()
	
	return newFiles  	

def filterBrokenH5(probFiles,tifFiles):
	
	"""Checks if a probFile exist for each tifFile, if not
	remembers the index."""

	brokenIdx=[]
		
	for i,tfile in enumerate(tifFiles):
		base=tfile.replace(".tif","")
		found=False

		
		for pfile in probFiles:
			if base in pfile:
				
				found=True
				break
		if not found:
			printWarning(os.path.basename(tfile) + " has not corresponding probFile.")
			brokenIdx.append(i)
	
	return brokenIdx		
