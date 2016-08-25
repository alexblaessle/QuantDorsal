#QT
from PyQt4 import QtGui, QtCore

import sys
import os

class channelWidget(QtGui.QDialog):
	
	"""Dialog to modify all settings about channel.
	"""
	
	def __init__(self,channel, parent = None):
	
		QtGui.QDialog.__init__(self, parent)
		
		self.channel=channel
		self.parent=parent
		
		#Labels
		self.lblName=QtGui.QLabel("Name:", self)
		self.lblIdx=QtGui.QLabel("Idx:", self)
		self.lblClassifier=QtGui.QLabel("Classifier:", self)
		self.lblH5Fn=QtGui.QLabel("H5Fn:", self)
		self.lblTifFn=QtGui.QLabel("TifFn:", self)
		self.lblProbThresh=QtGui.QLabel("ProbThresh:", self)
		self.lblProbIdx=QtGui.QLabel("ProbIdx:", self)
		self.lblProj=QtGui.QLabel("Projection:", self)
		self.lblBins=QtGui.QLabel("Bins:", self)
		self.lblMinPix=QtGui.QLabel("MinPix:", self)
		self.lblMedian=QtGui.QLabel("Median:", self)
		self.lblNorm=QtGui.QLabel("Norm:", self)
		self.lblBkgd=QtGui.QLabel("Bkgd:", self)
		
		self.lblClassifierVal=QtGui.QLabel("", self)
		self.lblH5FnVal=QtGui.QLabel("", self)
		self.lblTifFnVal=QtGui.QLabel("", self)
		
		self.updateLblClassifierVal()
		self.updateLblH5Val()
		self.updateLblTifVal()
		
		#LineEdits
		self.qleName = QtGui.QLineEdit(str(self.channel.name))
		self.qleIdx = QtGui.QLineEdit(str(self.channel.idx))
		self.qleProbThresh = QtGui.QLineEdit(str(self.channel.probThresh))
		self.qleProbIdx = QtGui.QLineEdit(str(self.channel.probIdx))
		self.qleBins = QtGui.QLineEdit(str(self.channel.bins))
		self.qleMinPix = QtGui.QLineEdit(str(self.channel.minPix))
		self.qleMedian = QtGui.QLineEdit(str(self.channel.median))
		self.qleBkgd = QtGui.QLineEdit(str(self.channel.bkgd))
		
		self.doubleValid=QtGui.QDoubleValidator()
		self.intValid=QtGui.QIntValidator()
		self.qleProbThresh.setValidator(self.doubleValid)
		self.qleProbIdx.setValidator(self.intValid)
		self.qleBins.setValidator(self.intValid)
		self.qleMinPix.setValidator(self.intValid)
		
		self.qleName.editingFinished.connect(self.setName)
		self.qleIdx.editingFinished.connect(self.setIdx)
		self.qleProbThresh.editingFinished.connect(self.setProbThresh)
		self.qleProbIdx.editingFinished.connect(self.setProbIdx)
		self.qleBins.editingFinished.connect(self.setBins)
		self.qleMinPix.editingFinished.connect(self.setMinPix)
		self.qleMedian.editingFinished.connect(self.setMedian)
		self.qleBkgd.editingFinished.connect(self.setBkgd)
		
		
		#ComboBox
		self.comboProj = QtGui.QComboBox(self)
		self.comboProj.addItem("Max")
		self.comboProj.addItem("Mean")
		self.comboProj.addItem("Sum")
		self.comboProj.addItem("None")
		
		self.initComboProj()
		
		self.comboProj.activated[str].connect(self.setProj)   
		
		#Buttons
		self.btnDone=QtGui.QPushButton('Done')
		self.btnDone.connect(self.btnDone, QtCore.SIGNAL('clicked()'), self.donePressed)
		
		self.btnClassifier=QtGui.QPushButton('Change')
		self.btnClassifier.connect(self.btnClassifier, QtCore.SIGNAL('clicked()'), self.setClassifier)
		
		self.btnH5=QtGui.QPushButton('Change')
		self.btnH5.connect(self.btnH5, QtCore.SIGNAL('clicked()'), self.setH5)
		
		self.btnTif=QtGui.QPushButton('Change')
		self.btnTif.connect(self.btnTif, QtCore.SIGNAL('clicked()'), self.setTif)
		
		#Checkboxes
		self.cbNorm = QtGui.QCheckBox('', self)
		self.updateCBs()
		self.connect(self.cbNorm, QtCore.SIGNAL('stateChanged(int)'), self.checkNorm)
		
		#Layout
		self.grid = QtGui.QGridLayout()		
		self.grid.setColumnMinimumWidth(2,200) 
		
		self.grid.addWidget(self.lblName,1,1)
		self.grid.addWidget(self.lblIdx,2,1)
		self.grid.addWidget(self.lblClassifier,3,1)
		self.grid.addWidget(self.lblTifFn,4,1)
		self.grid.addWidget(self.lblH5Fn,5,1)
		self.grid.addWidget(self.lblProbThresh,6,1)
		self.grid.addWidget(self.lblProbIdx,7,1)
		self.grid.addWidget(self.lblProj,8,1)
		self.grid.addWidget(self.lblBins,9,1)
		self.grid.addWidget(self.lblMinPix,10,1)
		self.grid.addWidget(self.lblMedian,11,1)
		self.grid.addWidget(self.lblBkgd,12,1)
		self.grid.addWidget(self.lblNorm,13,1)
		
		self.grid.addWidget(self.qleName,1,2)
		self.grid.addWidget(self.qleIdx,2,2)
		self.grid.addWidget(self.lblClassifierVal,3,2)
		self.grid.addWidget(self.lblTifFnVal,4,2)
		self.grid.addWidget(self.lblH5FnVal,5,2)
		self.grid.addWidget(self.qleProbThresh,6,2)
		self.grid.addWidget(self.qleProbIdx,7,2)
		self.grid.addWidget(self.comboProj,8,2)
		self.grid.addWidget(self.qleBins,9,2)
		self.grid.addWidget(self.qleMinPix,10,2)
		self.grid.addWidget(self.qleMedian,11,2)
		self.grid.addWidget(self.qleBkgd,12,2)
		self.grid.addWidget(self.cbNorm,13,2)
		
		self.grid.addWidget(self.btnClassifier,3,3)
		self.grid.addWidget(self.btnTif,4,3)
		self.grid.addWidget(self.btnH5,5,3)
		
		self.grid.addWidget(self.btnDone,15,3)
		
		self.setLayout(self.grid)    
			
		self.setWindowTitle('Channel Dialog')   
		
		self.show()
	
	def getChannel(self):
		return self.channel
	
	def setName(self):
		self.channel.name=str(self.qleName.text())
	
	def setIdx(self):
		self.channel.idx=int(str(self.qleIdx.text()))
		
	def setProbThresh(self):
		p=float(str(self.qleProbThresh.text()))
		if 0<=p and p<=1:
			self.channel.probThresh=float(str(self.qleProbThresh.text()))
		else:
			QtGui.QMessageBox.critical(None, "Error","Probability needs to be between 0 and 1.",QtGui.QMessageBox.Ok | QtGui.QMessageBox.Default)
	
	def setProbIdx(self):
		self.channel.probIdx=int(str(self.qleProbIdx.text()))
		
	def setBins(self):
		self.channel.bins=int(str(self.qleBins.text()))
			
	def setMinPix(self):
		self.channel.minPix=int(str(self.qleMinPix.text()))
		
	def setMedian(self):
		s=str(self.qleMedian.text())
		if s=='None':
			self.channel.median=None
		else:
			self.channel.median=float(str(self.qleMedian.text()))
		
	def setBkgd(self):
		s=str(self.qleBkgd.text())
		if s=='None':
			self.channel.bkgd=None
		else:
			self.channel.bkgd=float(str(self.qleBkgd.text()))
	
	def setProj(self,text):
		if str(text)=='None':
			self.channel.proj=None
		else:
			self.channel.proj=str(text)
	
	def initComboProj(self):
		idx=self.comboProj.findText(str(self.channel.proj),QtCore.Qt.MatchExactly)
		self.comboProj.setCurrentIndex(idx)
		
	def setTif(self):
		
		fn = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file',self.parent.lastOpen,"*.tif",))
		if fn=='':
			return
		
		self.channel.tifFn=fn
		self.updateLblTifVal()
		
	def setH5(self):
		
		fn = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file',self.parent.lastOpen,"*.h5",))
		if fn=='':
			return
		
		self.channel.h5Fn=fn
		self.updateLblH5Val()
	
	def setClassifier(self):
		
		fn = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file',self.parent.lastOpen,"*.ilp",))
		if fn=='':
			return
		
		self.channel.classifier=fn
		self.updateLblClassifierVal()
	
	def updateLblTifVal(self,n=50):	
		self.lblTifFnVal.setText("..."+self.channel.tifFn[-n:])	
		
	def updateLblH5Val(self,n=50):	
		self.lblH5FnVal.setText("..."+self.channel.h5Fn[-n:])	
	
	def updateLblClassifierVal(self,n=50):	
		self.lblClassifierVal.setText("..."+self.channel.classifier[-n:])	
	
		
	def checkNorm(self,val):
		self.channel.norm=bool(2*val)
		
	def updateCBs(self):	
		self.cbNorm.setCheckState(2*int(self.channel.norm))	
	
	def donePressed(self):
		self.done(1)
		return	