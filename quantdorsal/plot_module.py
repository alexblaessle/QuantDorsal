#===========================================================================================================================================================================
#Module description
#===========================================================================================================================================================================

"""QuantDorsal module for plotting. 

Contains functions for 

	* Showing images.
	* Quick creation of figures.
	* Publication ready figures.
	
"""

#===========================================================================================================================================================================
#Importing necessary modules
#===========================================================================================================================================================================

#Numpy/Scipy
import numpy as np

#System
import sys
import os

#Matplotlib
import matplotlib.pyplot as plt

#QuantDorsal
from term_module import *

#===========================================================================================================================================================================
#Module Functions
#===========================================================================================================================================================================

def imshow(img):
	
	"""Shows image data.

	If image data is multichannel, will show channels seperately.
	
	Args:
		img (numpy.ndarray): Some image.
		
	Returns:
		list: List of matplotlib.axes.
	
	"""	

	fig=plt.figure()
	
	if len(img.shape)>2:
		fig,axes=makeAxes([np.ceil(img.shape[0]/3.),3])
		
		for i in range(img.shape[0]):
			axes[i].imshow(img[i])
			plt.draw()
	else:
		fig,axes=makeAxes([1,1])

	return axes

		
		
def makeAxes(size):
	
	"""Makes figure with size subplots.
	
	Args:
		size (list): (x,y)-size of axes.
		
	Returns:
		tuple: Tuple containing:
		
			fig (matplotlib.figure): Figure.
			axes (list): List of matplotlib.axes.
	"""

	#Create figure
	fig=plt.figure()
	fig.show()
	
	#Add axis
	axes=[]	
	for i in range(int(size[0])*int(size[1])):
		axes.append(fig.add_subplot(size[0],size[1],i))

	return fig,axes


def getPubParms():
	
	"""Returns dictionary with good parameters for nice
	publication figures.

	Resulting ``dict`` can be loaded via ``plt.rcParams.update()``.
	
	.. note:: Use this if you want to include LaTeX symbols in the figure.
	
	Taken from the PyFRAP project. See also https://github.com/alexblaessle/PyFRAP. 
	
	Returns:	
		dict: Parameter dictionary.
	"""

	params = {'backend': 'ps',
	'axes.labelsize': 10,
	'text.fontsize': 10,
	'legend.fontsize': 10,
	'xtick.labelsize': 10,
	'ytick.labelsize': 10,
	'text.usetex': True,
	'font.family': 'sans-serif',
	#'font.sans-serif': 'Bitstream Vera Sans, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif',
	#'ytick.direction': 'out',
	'text.latex.preamble': [r'\usepackage{helvet}', r'\usepackage{sansmath}'] , #r'\sansmath',
	}	
	return params


def turnAxesForPub(ax,adjustFigSize=True,figWidthPt=180.4,ptPerInches=72.27):
	
	"""Turns axes nice for publication.
	
	If ``adjustFigSize=True``, will also adjust the size the figure.
	
	Taken from the PyFRAP project. See also https://github.com/alexblaessle/PyFRAP. 
	
	Args:
		ax (matplotlib.axes): A matplotlib axes.
	Keyword Args:
		adjustFigSize (bool): Adjust the size of the figure.
		figWidthPt (float): Width of the figure in pt.
		ptPerInches (float): Resolution in pt/inches.
	
	Returns:
		matplotlib.axes: Modified matplotlib axes.
	"""

	params=getPubParms()
	plt.rcParams.update(params)
	ax=setPubAxis(ax)
	setPubFigSize(ax.get_figure(),figWidthPt=figWidthPt)
	ax=closerLabels(ax,padx=3,pady=1)
	
	return ax
	

def setPubAxis(ax):

	"""Gets rid of top and right axis.
	
	Taken from the PyFRAP project. See also https://github.com/alexblaessle/PyFRAP. 
	
	Args:
		ax (matplotlib.axes): A matplotlib axes.
	
	Returns:
		matplotlib.axes: Modified matplotlib axes.
	"""

	ax.spines['top'].set_color('none')
	ax.spines['right'].set_color('none')
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')
	return ax

def setPubFigSize(fig,figWidthPt=180.4,ptPerInches=72.27):
	
	"""Adjusts figure size/aspect into golden ratio.
	
	Taken from the PyFRAP project. See also https://github.com/alexblaessle/PyFRAP. 
	
	Args:
		fig (matplotlib.figure): Some figure.
	
	Keyword Args:
		figWidthPt (float): Width of the figure in pt.
		ptPerInches (float): Resolution in pt/inches.
		
	Returns:
		matplotlib.figure: Adjusted figure.
	"""

	inchesPerPt = 1.0/ptPerInches
	goldenMean = (np.sqrt(5)-1.0)/2.0 # Aesthetic ratio
	figWidth = figWidthPt*inchesPerPt # width in inches
	figHeight = figWidth*goldenMean # height in inches
	figSize = [figWidth,figHeight]
	fig.set_size_inches(figSize[0],figSize[1])
	fig.subplots_adjust(bottom=0.25)
	fig.subplots_adjust(left=0.2)
	fig.subplots_adjust(top=0.9)
	return fig

def closerLabels(ax,padx=10,pady=10):
	
	"""Moves x/y labels closer to axis.
	
	Taken from the PyFRAP project. See also https://github.com/alexblaessle/PyFRAP. 
	
	Args:
		ax (matplotlib.axes): Some axes.
		
	Keyword Args:
		padx (int): Number of inches of padding between x-axis and label.
		pady (int): Number of inches of padding between y-axis and label.
	
	Returns:
		matplotlib.axes. Adjusted axes.
			
	
	"""
	
	ax.xaxis.labelpad = padx
	ax.yaxis.labelpad = pady
	return ax

def turnIntoRadialPlot(ax):
	
	ax=setRadiansTicks(ax)
	ax=setRadianLabels(ax)
	
	return ax
	

def setRadiansTicks(ax):
	
	"""Sets x-labels to radians from -pi to pi."""
	
	xtick = np.linspace(-np.pi, np.pi, 5)
	xlabel = [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$+\frac{\pi}{2}$",   r"$+\pi$"]
	ax.set_xticks(xtick)
	ax.set_xticklabels(xlabel)
	
	return ax

def setRadianLabels(ax):
	
	"""Sets default labels for radial plots."""

	ax.set_ylabel("Intensity (AU)")
	ax.set_xlabel("Embryo Circumfarence (radians)")

	return ax

def redraw(ax):
	
	"""Redraws axes's figure's canvas.
	
	Makes sure that current axes content is visible.
	
	Args:
		ax (matplotlib.axes): Matplotlib axes.
		
	Returns:
		matplotlib.axes: Matplotlib axes 
	
	"""
			
	ax.get_figure().canvas.draw()
	
	return ax