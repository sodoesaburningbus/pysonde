#Script to test PySonde
#Chris Phillips

#Load PySonde
from pysonde.pysonde import PySonde

#Import other necessary modules
import matplotlib.pyplot as pp

#Location of test sounding
sounding_path = "./example_sounding_cswr.txt"

#Create sounding object
sonde = PySonde(sounding_path, "NWS")

#Print out the available variables
print("Sounding variables are:")
for k in sorted(sonde.sounding.keys()):
    print(k)

#Create the figure and SkewT object
fig, skewt = sonde.basic_skewt(nbarbs=50, pblh=True)

#Display the SkewT
pp.show()
