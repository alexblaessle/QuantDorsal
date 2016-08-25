import pickle
import platform
import gc
import sys
import os
import csv
	
def saveToPickle(obj,fn=None):
	
	"""Saves obj into pickled format.
	
	.. note:: If ``fn==Non``, will try to save to ``obj.name``, otherwise unnamed.pk
	
	Keyword Args:
		fn (str): Output file name.	
	
	Returns: 
		str: Output filename.
	
	"""
	
	cleanUp()
        if fn==None:
                if hasattr(obj,"name"):
                        fn=obj.name+".pk"
                else:
                        fn="unnamed"+".pk"
                
        with open(fn, 'wb') as output:
                pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
        return fn

def loadFromPickle(fn):
	
	"""Loads obj from pickled format.
	
	Args:
		fn (str): Filename.	
	
	Returns: 
		str: Output filename.
	
	"""
	
	cleanUp()
	
        if platform.system() in ["Darwin","Linux"]:
                filehandler=open(fn, 'r')
        elif platform.system() in ["Windows"]:
                filehandler=open(fn, 'rb')
                
        loadedFile=pickle.load(filehandler)
        
        return loadedFile

def cleanUp():
	"""Calls garbage collector to clean up.
	"""
	
	gc.collect()
	return None

     
