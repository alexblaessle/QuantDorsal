#QT
from PyQt4 import QtGui, QtCore

import sys
import os

class pathWidget(QtGui.QDialog):
	
	"""Dialog to modify all settings about channel.
	"""
	
	def __init__(self,identifier,path, parent = None):
	
		QtGui.QDialog.__init__(self, parent)

		self.parent=parent
		self.identifier=identifier
		self.path=path
		
		#Labels
		self.lblIdentifier=QtGui.QLabel("Identifier:", self)
		self.lblPath=QtGui.QLabel("Path:", self)
		self.lblPathVal=QtGui.QLabel(self.path, self)
		
		#QLEs
		self.qleIdentifier = QtGui.QLineEdit(str(self.identifier))
		self.qleIdentifier.editingFinished.connect(self.setIdentifier)
		
		#Button
		self.btnPath=QtGui.QPushButton('Change')
		self.btnPath.connect(self.btnPath, QtCore.SIGNAL('clicked()'), self.setPath)
		
		#Layout
		self.grid = QtGui.QGridLayout()		
		self.grid.setColumnMinimumWidth(2,200) 
		
		self.grid.addWidget(self.lblIdentifier,1,1)
		self.grid.addWidget(self.lblPath,2,1)
		self.grid.addWidget(self.lblPathVal,2,2)
		self.grid.addWidget(self.qleIdentifier,1,2)
		self.grid.addWidget(self.btnPath,2,3)
		
		self.setLayout(self.grid)    
			
		self.setWindowTitle('Path Dialog')   
		
		self.show()
			
	def setIdentifier(self):
		self.identifier=str(self.qleIdentifier.text())
	
	def updateLblPathVal(self,n=50):	
		self.lblPathVal.setText("..."+self.path[-n:])		
		
	def getPath(self):
		return self.identifier,self.path
	
	def setPath(self):
		
		fn = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file',self.parent.lastOpen,))
		if fn=='':
			return
		
		self.path=fn
		self.updateLblPathVal()
		
	def donePressed(self):
		self.done(1)
		return			
		
		
		
		
		
		
		
		
	