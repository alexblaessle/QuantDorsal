#===========================================================================================================================================================================
#Module Description
#===========================================================================================================================================================================

"""Terminal module for QuantDorsal.
Provides custom output inside a Python/bash terminal.
"""

#===========================================================================================================================================================================
#Improting necessary modules
#===========================================================================================================================================================================

import colorama

#===========================================================================================================================================================================
#Module Functions
#===========================================================================================================================================================================


def printWarning(txt):
	
	"""Prints Warning of the form "WARNING: txt", while warning is rendered yellow.	
	"""

	print(colorama.Fore.YELLOW + "WARNING:") + colorama.Fore.RESET + txt

def printError(txt):
	
	"""Prints Error of the form "ERROR: txt", while error is rendered red.	
	"""
	
	print(colorama.Fore.RED + "ERROR:") + colorama.Fore.RESET + txt

def printNote(txt):
	
	"""Prints note of the form "NOTE: txt", while note is rendered green.	
	"""

	print(colorama.Fore.GREEN + "NOTE:") + colorama.Fore.RESET + txt