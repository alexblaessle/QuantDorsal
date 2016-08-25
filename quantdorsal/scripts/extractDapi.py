"""Takes a bunch of tif files and saves only the dapi channel 
into one combined tif file."""

import sys
import im_module as im
import numpy as np
import tifffile

fn=sys.argv[1]

images=im.readImageData(fn)

#images=images[:10]

newImages=[]
for i,img in enumerate(images):
	
		

	if img.shape[2]==792:
		newImages.append(img[0])
		print img[0].shape
#newImages=np.concatenate(newImages,axis=0)

	tifffile.imsave("trainingDataSet"+str(i)+".tif",img[0])






	


