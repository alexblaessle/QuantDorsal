#===========================================================================================================================================================================
#Module description
#===========================================================================================================================================================================

"""Module for misc. functions.

Contains functions for 

	* Reading path files.
	* Running commands
	
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
import shutil
from tempfile import mkstemp

#QuantDorsal
from term_module import *


#===========================================================================================================================================================================
#Module Functions
#===========================================================================================================================================================================


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

def setPath(identifier,val,fnPath=None):
	
	if fnPath==None:
		fnPath=getPathFile()
	else:
		if not os.path.isfile(fnPath):
			printWarning(fnPath+" does not exist. Will continue with paths defined in default paths files.")
			fnPath=getPathFile()
		
	txtLineReplace(fnPath,identifier,identifier+"="+str(val))
			
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

			
def txtLineReplace(filePath, pattern, subst):
		
	"""Replaces line in file that starts with ``pattern`` and substitutes it 
	with ``subst``.
	
	.. note:: Will create temporary file using ``tempfile.mkstemp()``. You should have 
	   read/write access to whereever ``mkstemp`` is putting files.
	
	Args:
		filePath (str): Filename.
		pattern (str): Pattern to be looked for.
		subst (str): String used as a replacement.
			
	"""
	
	
	#Create temp file
	fh, absPath = mkstemp()
	newFile = open(absPath,'w')
	oldFile = open(filePath)
	
	#Loop through file and replace line 
	for line in oldFile:
		
		if line.startswith(pattern):
			newFile.write(line.replace(line, subst))
			newFile.write('\n')
		else:
			newFile.write(line)
			
	#close temp file
	newFile.close()
	os.close(fh)
	oldFile.close()
		
	#Remove original file
	os.remove(filePath)
	
	#Move new file
	shutil.move(absPath, filePath)
	return	

def copyObjAttr(obj1,obj2,filterAttr=[],debug=False):
	
	"""Copies all attributes in obj1 into obj2, leaving
	out all attributes with names defined in filterAttr.
	
	Args:
		obj1 (object): Source object.
		obj2 (object): Destination object.
		
	Keyword Args:
		filterAttr (list): List of attributes to be left out of copying.
		debug (bool): Print debugging messages.
	
	Returns:
		object: Updated obj2.
	
	"""
	
	
	#Going through all attributes blank object
	for item in vars(obj1):
		if item not in filterAttr:
			setattr(obj2, str(item), vars(obj1)[str(item)])
		
	return obj2
