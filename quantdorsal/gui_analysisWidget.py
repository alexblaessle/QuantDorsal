#QT
from PyQt4 import QtGui, QtCore

import sys
import os

class analysisWidget(QtGui.QDialog):
	
	"""Dialog to modify all settings about exp.
	"""
	
	def __init__(self,analysis, parent = None):
	
		QtGui.QDialog.__init__(self, parent)
		
		self.analysis=analysis
		self.fitMethod=self.analysis.fitMethod
		
		#Labels
		self.lblIlastik=QtGui.QLabel("Ilastik:", self)
		self.lblTif=QtGui.QLabel("Tif:", self)
		self.lblAlign=QtGui.QLabel("Align:", self)
		self.lblFitMethod=QtGui.QLabel("Fit Method:", self)
		
		#Checkboxes
		self.cbIlastik = QtGui.QCheckBox('', self)
		self.cbTif = QtGui.QCheckBox('', self)
		self.cbAlign = QtGui.QCheckBox('', self)
		
		self.updateCBs()
		
		self.connect(self.cbIlastik, QtCore.SIGNAL('stateChanged(int)'), self.checkIlastik)
		self.connect(self.cbTif, QtCore.SIGNAL('stateChanged(int)'), self.checkTif)
		self.connect(self.cbAlign, QtCore.SIGNAL('stateChanged(int)'), self.checkAlign)
	
		#Combo
		self.comboMeth = QtGui.QComboBox(self)
		self.comboMeth.addItem("maxIntensity")
		self.comboMeth.addItem("Gaussian")
		self.initComboMeth()
		
		self.comboMeth.activated[str].connect(self.setMeth)   
		
		#Buttons
		self.btnDone=QtGui.QPushButton('Done')
		self.btnDone.connect(self.btnDone, QtCore.SIGNAL('clicked()'), self.donePressed)
		
		#Layout
		self.grid = QtGui.QGridLayout()		
		
		self.grid.addWidget(self.lblIlastik,1,1)
		self.grid.addWidget(self.lblTif,2,1)
		self.grid.addWidget(self.lblAlign,3,1)
		self.grid.addWidget(self.lblFitMethod,4,1)
		
		self.grid.addWidget(self.cbIlastik,1,2)
		self.grid.addWidget(self.cbTif,2,2)
		self.grid.addWidget(self.cbAlign,3,2)
		self.grid.addWidget(self.comboMeth,4,2)
		
		self.grid.addWidget(self.btnDone,5,2)
		
		self.setLayout(self.grid)    
			
		self.setWindowTitle('Analysis Dialog')   
		
		self.show()
	
	def checkAlign(self,val):
		self.analysis.align=bool(2*val)
	
	def checkIlastik(self,val):
		self.analysis.ilastik=bool(2*val)
	
	def checkTif(self,val):
		self.analysis.saveTif=bool(2*val)
	
	def updateCBs(self):
		self.cbIlastik.setCheckState(2*int(self.analysis.ilastik))
		self.cbTif.setCheckState(2*int(self.analysis.saveTif))
		self.cbIlastik.setCheckState(2*int(self.analysis.ilastik))
	
	def initComboMeth(self):
		idx=self.comboMeth.findText(str(self.analysis.fitMethod),QtCore.Qt.MatchExactly)
		self.comboMeth.setCurrentIndex(idx)
	
	def setMeth(self,text):
		self.analysis.fitMethod=str(text)
	
	def getAnalysis(self):
		return self.analysis
	
	def donePressed(self):
		self.done(1)
		return		