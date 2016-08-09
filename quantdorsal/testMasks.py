"""Script to test h5 files"""

import ilastik_module as ilm
import matplotlib.pyplot as plt

import os


files=['../data/tifs/160804_toll10B_dapi_series0_c0_Probabilities.h5', '../data/tifs/160804_toll10B_dapi_series1_c0_Probabilities.h5']



for fn in files:
	
	data=ilm.readH5(fn)
	data=data[:,:,:,0]
	mask=ilm.makeProbMask(data)
	
	
	
	plt.imshow(mask[0])
	plt.show()

	
	

	




