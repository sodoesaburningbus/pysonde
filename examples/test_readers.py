#This script tests the PySonde reader for each sounding format in turn.
#Chris Phillips
#June, 2020

#First import PySonde
from pysonde.pysonde import PySonde

#Import other modules
from datetime import datetime

#Test the Wyoming sounding
sonde = PySonde("example_soundings/wyoming_sounding.txt", "wyo")
print("Lines in Wyoming sounding: {}\n".format(sonde.sounding["temp"].size))

#Test the NWS sounding
sonde = PySonde("example_soundings/nws_sounding.txt", "nws")
print("Lines in NWS sounding: {}\n".format(sonde.sounding["temp"].size))

#Test the Web reader
sonde = PySonde("bna", "web", date=datetime(2011, 4, 27, 12))
print("Lines in Web sounding: {}\n".format(sonde.sounding["temp"].size))
fig, skewt = sonde.basic_skewt()
pp.show()
print(sonde.sfc_cape)