#QT
from PyQt4 import QtGui, QtCore

import sys
import os

#QD
import qDAnalysis
import io_module as iom
import misc_module as mm

#QD GUI
from gui_channelWidget import channelWidget
from gui_expWidget import expWidget
from gui_pathWidget import pathWidget
from gui_analysisWidget import analysisWidget

#Main Window
class MainWindow(QtGui.QMainWindow):

	def __init__(self, parent = None):
	
		QtGui.QWidget.__init__(self, parent)
		
		#Some global parms
		self.analysis=None
		self.loadPaths()
		
		#Setup GUI
		self.setupUI()
		
		#Menus
		self.setupMenus()
		
		#Layout
		self.mainWidget=QtGui.QWidget(self)
		self.hbox = QtGui.QHBoxLayout()
		self.hbox.addWidget(self.expList)
		self.hbox.addWidget(self.channelList)
		
		self.mainWidget.setLayout(self.hbox)
		
		self.setCentralWidget(self.mainWidget)
		self.show()
	
	def loadPaths(self):
		self.lastOpen=mm.getPath('lastOpen')
		print self.lastOpen
	
	def saveConf(self):
		mm.setPath("lastOpen",self.lastOpen)
		
	def setupUI(self):
		
		#Exp List
		self.expList=QtGui.QTreeWidget()
		self.expList.setHeaderLabels(["Exps"])
		self.expList.setColumnWidth(0,100)
		self.expList.itemClicked.connect(self.expClicked)
		self.expList.itemDoubleClicked.connect(self.expDoubleClicked)
		
		#Channels List
		self.channelList=QtGui.QTreeWidget()
		self.channelList.setHeaderLabels(["Channels"])
		self.channelList.setColumnWidth(0,100)
		self.channelList.itemDoubleClicked.connect(self.channelClicked)
	
	def setupMenus(self):
		
		self.menubar = self.menuBar()
		self.mbFile = self.menubar.addMenu('&File')
		
		self.mbEdit = self.menubar.addMenu('&Edit')
		self.mbAnalysis = self.menubar.addMenu('&Analysis')
		
		self.mbSettings = self.menubar.addMenu('&Settings')
		
		self.setupFileMenu()
		self.setupEditMenu()
		self.setupAnalysisMenu()
		self.setupSettingsMenu()
		
	def setupFileMenu(self):
		
		newAnalysisButton = QtGui.QAction('New Analysis', self)
		self.connect(newAnalysisButton, QtCore.SIGNAL('triggered()'), self.newAnalysis)
		
		saveAnalysisButton = QtGui.QAction('Save Analysis', self)
		self.connect(saveAnalysisButton, QtCore.SIGNAL('triggered()'), self.saveAnalysis)
		
		loadAnalysisButton = QtGui.QAction('Load Analysis', self)
		self.connect(loadAnalysisButton, QtCore.SIGNAL('triggered()'), self.loadAnalysis)
		
		exitButton = QtGui.QAction('Exit', self)
		exitButton.setShortcut('Ctrl+Q')	
		self.connect(exitButton, QtCore.SIGNAL('triggered()'), QtCore.SLOT('close()'))
		
		self.mbFile.addAction(newAnalysisButton)
		self.mbFile.addAction(saveAnalysisButton)
		self.mbFile.addAction(loadAnalysisButton)
		self.mbFile.addAction(exitButton)
		
	def setupEditMenu(self):
		
		copyChannelButton = QtGui.QAction('Globalize channel settings', self)
		self.connect(copyChannelButton, QtCore.SIGNAL('triggered()'), self.copyOptsToAllChannels)
		

		self.mbEdit.addAction(copyChannelButton)
		
	def setupAnalysisMenu(self):
		
		editAnalysisButton = QtGui.QAction('Edit Analysis', self)
		self.connect(editAnalysisButton, QtCore.SIGNAL('triggered()'), self.editAnalysis)
		
		createAllSignalProfilesButton = QtGui.QAction('Create signal profiles', self)
		self.connect(createAllSignalProfilesButton, QtCore.SIGNAL('triggered()'), self.createAllSignalProfiles)
		
		plotAlignedSignalButton = QtGui.QAction('Plot Aligned Signal', self)
		self.connect(plotAlignedSignalButton, QtCore.SIGNAL('triggered()'), self.plotAlignedSignal)
		
		runIlastikButton = QtGui.QAction('Run ilastik', self)
		self.connect(runIlastikButton, QtCore.SIGNAL('triggered()'), self.runIlastik)
		
		saveChannelToTifButton = QtGui.QAction('Save Channel to Tif', self)
		self.connect(saveChannelToTifButton, QtCore.SIGNAL('triggered()'), self.saveChannelToTif)
		
		alignSignalsButton = QtGui.QAction('Align signal profiles', self)
		self.connect(alignSignalsButton, QtCore.SIGNAL('triggered()'), self.alignSignalProfiles)
		
		runAnalysisButton = QtGui.QAction('Edit Analysis', self)
		self.connect(runAnalysisButton, QtCore.SIGNAL('triggered()'), self.runAnalysis)
		
		self.mbAnalysis.addAction(editAnalysisButton)
		self.mbAnalysis.addAction(saveChannelToTifButton)
		self.mbAnalysis.addAction(runIlastikButton)
		self.mbAnalysis.addAction(createAllSignalProfilesButton)
		self.mbAnalysis.addAction(alignSignalsButton)
		self.mbAnalysis.addAction(plotAlignedSignalButton)
		self.mbAnalysis.addAction(runAnalysisButton)
		
		
	def setupSettingsMenu(self):
		
		setPathButton = QtGui.QAction('Set Path', self)
		self.connect(setPathButton, QtCore.SIGNAL('triggered()'), self.setPath)
		
		setIlastikPathButton = QtGui.QAction('Set Ilastik Path', self)
		self.connect(setIlastikPathButton, QtCore.SIGNAL('triggered()'), self.setIlastikPath)
		
		self.mbSettings.addAction(setPathButton)
		self.mbSettings.addAction(setIlastikPathButton)
		
	def closeEvent(self, event):
		
		"""Closes GUI.
		
		Args:
			event (QtCore.closeEvent): Close event triggered by close signal.
		
		"""
		
		self.saveConf()
		
		reply = QtGui.QMessageBox.question(self, 'Message',"Are you sure you want to quit?", QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
	
		if reply == QtGui.QMessageBox.Yes:
			
			if self.analysis!=None:
				replySave = QtGui.QMessageBox.question(self, 'Message',"Do you want to save your analysis?", QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
				if replySave == QtGui.QMessageBox.Yes:
					self.saveAnalysis()
		else:
			event.ignore()
		
		return
	
	def saveAnalysis(self):
		
		#Filename dialog
		fn=QtGui.QFileDialog.getSaveFileName(self, 'Save file',self.lastOpen,"*.qd",)
		fn=str(fn)
		
		
		if not os.path.exists(os.path.dirname(fn)):
			return
		
		self.lastOpen=os.path.dirname(fn)
		self.analysis.save(fn)
		
	def loadAnalysis(self):
		
		fnLoad=str(QtGui.QFileDialog.getOpenFileName(self, 'Open file', self.lastOpen,"*.qd",))
		if fnLoad=='':
			return
		
		self.lastOpen=os.path.dirname(fnLoad)
		
		self.analysis=iom.loadFromPickle(fnLoad)
		
		self.updateExpList()
		
		return self.analysis	

	def newAnalysis(self):
		
		if self.analysis!=None:
			replySave = QtGui.QMessageBox.question(self, 'Message',"Do you want to save your current analysis first?", QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
			if replySave == QtGui.QMessageBox.Yes:
				self.saveAnalysis()
			
		text, ok = QtGui.QInputDialog.getText(self, 'Input Dialog', 'Enter analysis name:')
        
		if ok:
			self.analysis=qDAnalysis.analysis(str(text))
		else:
			return
		
		fn = str(QtGui.QFileDialog.getExistingDirectory(self, 'Select data directory',))
		
		if os.path.exists(fn):
			self.analysis.loadImageData(fn,nChannel=3,destChannel=0)
			self.updateExpList()
			
	def updateExpList(self):
		
		self.expList.clear()
		for r in self.analysis.exps:
			QtGui.QTreeWidgetItem(self.expList,[str(r.series)])
		
	def expClicked(self):
		idx=self.expList.indexFromItem(self.expList.currentItem()).row()
		self.exp=self.analysis.exps[idx]
		self.updateChannelList()
		return idx
		
	def updateChannelList(self):
		
		self.channelList.clear()
		for r in self.exp.channels:
			QtGui.QTreeWidgetItem(self.channelList,[r.name])
		return
	
	def channelClicked(self):
		idx=self.channelList.indexFromItem(self.channelList.currentItem()).row()
		self.channel=self.exp.channels[idx]
		
		channelDialog = channelWidget(self.channel,self)
		if channelDialog.exec_():
			self.channel = channelDialog.getChannel()
			self.exp.channels[idx]=self.channel	
			
	def expDoubleClicked(self):
		idx=self.expClicked()
		
		expDialog = expWidget(self.exp,self)
		if expDialog.exec_():
			self.exp = expDialog.getExp()
			self.analysis.exps[idx]=self.exp	
	
	def copyOptsToAllChannels(self):
		
		filterAttr=["name","exp","angles","anglesAligned","signals","signalsAligned"]
		
		name, ok = QtGui.QInputDialog.getText(self, 'Input Dialog',
					'Name of channels to copy attributes to (if left blank, will copy to all channels.):')
		name=str(name)
		
		
		for exp in self.analysis.exps:
			for channel in exp.channels:
				if name in channel.name:
					mm.copyObjAttr(self.channel,channel,filterAttr=filterAttr)
		
	def setPath(self,identifier="",path=""):
		
		pathDialog = pathWidget(identifier,path,self)
		identifier,path = pathDialog.getPath()
		mm.setPath(identifier,path)

	def setIlastikPath(self):
		path=mm.getPath('ilastikBin')
		self.setPath(identifier='ilastikBin',path=path)
		
	def saveChannelToTif(self,name=None,fnOut=None):
		
		if name==None:
			name, ok = QtGui.QInputDialog.getText(self, 'Input Dialog',
						'Name of channel to save tifs (if left blank, will copy to all channels.):')
			
			if not ok:
				return
			
		name=str(name)
		if fnOut==None:
			fnOut=str(QtGui.QFileDialog.getExistingDirectory(self, 'Select output directory',))
		
		for channel in self.analysis.exps[0].channels:
			if name in channel.name:
				self.analysis.saveChannelToTif(fnOut,name,prefix=self.analysis.name,axes='ZYX',debug=True)
	
	def findCorrespondingH5Files(self,name=None):
		
		if name==None:
			name, ok = QtGui.QInputDialog.getText(self, 'Input Dialog',
					'Name of channel (if left blank, will try to find for all channels.):')
		
			if not ok:
				return
		
		name=str(name)
				
		for channel in self.analysis.exps[0].channels:
			if name in channel.name:
				self.analysis.findCorrespondingH5Files(name)

	def runIlastik(self,name=None):
		
		if name==None:
		
			name, ok = QtGui.QInputDialog.getText(self, 'Input Dialog',
						'Name of channel (if left blank, will try to run for all channels.):')
			
			if not ok:
				return
		
		name=str(name)
		
		for channel in self.analysis.exps[0].channels:
			if name in channel.name:		
				self.analysis.runIlastik(name)
				
	def createAllSignalProfiles(self,nameMask=None,nameSignal=None):
		
		if nameMask==None:
		
			nameMask, ok = QtGui.QInputDialog.getText(self, 'Input Dialog',
						'Name of mask channel (if left blank, will try to use Dapi.):')
			
			if not ok:
				return
				
		nameMask=str(nameMask)
		
		if nameMask=="":
			nameMask='Dapi'
		
		if nameSignal==None:
		
			nameSignal, ok = QtGui.QInputDialog.getText(self, 'Input Dialog',
						'Name of signal channel (if left blank, will try to use Dorsal.):')
			
			if not ok:
				return
			
		nameSignal=str(nameSignal)
		
		if nameMask=="":
			nameMask='Dorsal'
		
		self.analysis.createAllSignalProfiles(nameSignal,nameMask)
		
	def alignSignalProfiles(self,nameSignal=None):
		
		if nameSignal==None:
		
			nameSignal, ok = QtGui.QInputDialog.getText(self, 'Input Dialog',
						'Name of signal channel (if left blank, will try to use Dorsal.):')
			
			if not ok:
				return
		
		self.analysis.alignSignalProfiles(nameSignal)
		
	def plotAlignedSignal(self,name=None):
		if name==None:
			name=self.channel.name
			
		self.analysis.plotAlignedSignal(name,title=self.analysis.name)
	
	def editAnalysis(self):
		analysisDialog = analysisWidget(self.analysis,self)
		if analysisDialog.exec_():
			self.analysis = analysisDialog.getAnalysis()
	
	def runAnalysis(self):
		
		#Making analysis settings
		self.editAnalysis()
		
		#Choosing channels
		nameSignal, ok = QtGui.QInputDialog.getText(self, 'Input Dialog',
					'Name of signal channel:')
		nameSignal=str(nameSignal)
		if not ok:
			return
		
		nameMask, ok = QtGui.QInputDialog.getText(self, 'Input Dialog',
					'Name of signal channel:')
		nameMask=str(nameMask)
		if not ok:
			return
		
		#Choosing output folder
		fnOut=str(QtGui.QFileDialog.getExistingDirectory(self, 'Select output directory',))
		
		#Run
		if self.analysis.saveTif:
			self.saveChannelToTif(name=nameMask,fnOut=fnOut)
			
		if self.analysis.ilastik:
			self.runIlastik(name=nameMask)
		else:
			self.findCorrespondingH5Files(name=nameMask)
			
		self.createAllSignalProfiles(nameMask=nameMask,nameSignal=nameSignal)	
		
		if self.analysis.align:
			self.alignSignalProfiles(nameSignal=nameSignal)
				
#Main
def main():
		    
	app = QtGui.QApplication(sys.argv)
	font=app.font()
	font.setPointSize(12)
	app.setFont(font)
	
	mainWin = MainWindow()
	mainWin.show()
	
	sys.exit(app.exec_())
	
if __name__ == '__main__':
	main()	