#for img in images:
	
	##Maximum intensity projection
	#proj=im.maxIntProj(img,1)
	
	##Otsu
	#thresh,mask=im.otsuImageJ(proj[0],1,0,debug=False,L=256)
	#maskedImg=mask*proj[0]
	
	##Dilate threshhold
	#maskDilated=skimage.morphology.binary_dilation(mask)
	
	#np.save("maskDilated.npy",maskDilated)
	
	
	##raw_input()
	
	##Get contours
	#contours=skimage.measure.find_contours(maskDilated,0.5)
	
	##Filter by lengths
	#newContours=[]
	#for contour in contours:
		#if contour.shape[0]>200:
			#newContours.append(contour)
	#contours=newContours
	
	###Fit ellipse
	##ellipses=[]
	
	##x=np.arange(maskDilated.shape[0])
	##y=np.arange(maskDilated.shape[1])	
	##X,Y=np.meshgrid(x,y)
	
	##x2=X[np.where(maskDilated==1)[0]].flatten()
	##y2=Y[np.where(maskDilated==1)[1]].flatten()
	
	##raw_input()
	
	##for contour in contours:
		
		
		
		##ell=im.fitEllipse(x,y)
		##center=im.ellipseCenter(ell)
		##angle=im.ellipseAngleOfRotation(ell)
		##length=im.ellipseAxisLength(ell)
		
		##ellipses.append([center,length,angle])	
	
	##Show result of threshholding
	#fig=plt.figure()
	#fig.show()
	#ax=fig.add_subplot(131)
	#ax.imshow(proj[0].T)
	#ax=fig.add_subplot(132)
	#ax.imshow(mask.T)
	#ax=fig.add_subplot(133)
	#ax.imshow(maskedImg.T)
	
	#for contour in contours:
		#ax.plot(contour[:,0],contour[:,1],'r-')
		
	##for ell in ellipses:	
		
		##print ell[0]
		##print ell[1]
		##print ell[2]
		
		##p=ptc.Ellipse(xy=ell[0], width=ell[1][0], height=ell[1][1], angle=ell[2],fill=False,color='g')
		##ax.add_patch(p)
	
	#plt.draw()
	#raw_input()
	
	
	
	

##Otsu threshhold

