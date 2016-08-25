#QT
from PyQt4 import QtGui, QtCore

import sys
import os

class expWidget(QtGui.QDialog):
	
	"""Dialog to modify all settings about exp.
	"""
	
	def __init__(self,exp, parent = None):
	
		QtGui.QDialog.__init__(self, parent)
		
		self.exp=exp
		
		#Labels
		self.lblSeries=QtGui.QLabel("Series:", self)
		self.lblOrgFn=QtGui.QLabel("OrgFn:", self)
		
		self.lblOrgFnVal=QtGui.QLabel("", self)
		self.updateLblOrgVal()
		
		#QLEs
		self.qleSeries = QtGui.QLineEdit(str(self.exp.series))
		self.qleSeries.editingFinished.connect(self.setSeries)
		
		#Buttons
		self.btnDone=QtGui.QPushButton('Done')
		self.btnDone.connect(self.btnDone, QtCore.SIGNAL('clicked()'), self.donePressed)
		
		#Layout
		self.grid = QtGui.QGridLayout()		
		self.grid.setColumnMinimumWidth(2,200) 
		
		self.grid.addWidget(self.lblSeries,1,1)
		self.grid.addWidget(self.lblOrgFn,2,1)
		
		self.grid.addWidget(self.qleSeries,1,2)
		self.grid.addWidget(self.lblOrgFnVal,2,2)
		
		self.setLayout(self.grid)    
			
		self.setWindowTitle('Exp Dialog')   
		
		self.show()
	
	def setSeries(self):
		self.exp.series=str(self.qleSeries.text())
	
	def updateLblOrgVal(self,n=50):	
		self.lblOrgFnVal.setText("..."+self.exp.orgFn[-n:])		
	
	def getExp(self):
		return self.exp
	
	def donePressed(self):
		self.done(1)
		return		
	
	