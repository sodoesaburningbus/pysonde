#Script to test PySonde
#Chris Phillips

#Load PySonde
from pysonde.pysonde import PySonde

#Import other necessary modules
import matplotlib.pyplot as pp

#Location of test sounding
sounding_path = "example_soundings/wyoming_sounding.txt"

#Create sounding object
sonde = PySonde(sounding_path, "wyo")

#Create the figure and SkewT object
fig, skewt = sonde.basic_skewt()

#Display the SkewT
pp.show()
