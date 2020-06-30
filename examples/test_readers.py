#This script tests the PySonde reader for each sounding format in turn.
#Chris Phillips
#June, 2020

#First import PySonde
from pysonde.pysonde import PySonde

#Import other modules
from datetime import datetime
import matplotlib.pyplot as pp

#Test the Wyoming sounding
sonde = PySonde("example_soundings/wyoming_sounding.txt", "wyo")
print("Lines in Wyoming sounding: {}".format(sonde.sounding["temp"].size))
print("Wyoming CAPE: {:.03f}\n".format(sonde.sfc_cape))

#Test the NWS sounding
sonde = PySonde("example_soundings/nws_sounding.txt", "nws")
print("Lines in NWS sounding: {}".format(sonde.sounding["temp"].size))
print("NWS CAPE: {:.03f}\n".format(sonde.sfc_cape))

#Test the Web reader
sonde = PySonde("bna", "web", date=datetime(2011, 4, 27, 12))
print("Lines in Web sounding: {}".format(sonde.sounding["temp"].size))
print("WEB CAPE: {:.03f}\n".format(sonde.sfc_cape))

#Test the WRF reader
sonde = PySonde("example_soundings/wrfscm_sounding.txt", "wrf")
print("Lines in WRF sounding: {}".format(sonde.sounding["temp"].size))
print("WRF CAPE: {:.03f} (should be similar to NWS)".format(sonde.sfc_cape))

