#Script to test PySonde
#Chris Phillips

#Load PySonde
from pysonde.pysonde import PySonde

#Import other necessary modules
import matplotlib.pyplot as pp

#Location of test sounding
sounding_path = "./example_sounding_pysonde.txt"

#Create sounding object
sonde = PySonde(sounding_path, "csv")

#Print out the available variables
print("Sounding variables are:")
for k in sorted(sonde.sounding.keys()):
    print(k)

print('Thermo variables are:')
print('CAPE, ', sonde.sfc_cape)
print('CIN, ', sonde.sfc_cin)

#Create the figure and SkewT object
fig, skewt = sonde.basic_skewt(nbarbs=50, pblh=True)

#Display the SkewT
pp.show()
