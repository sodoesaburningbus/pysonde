#This script shows an example for calculating and plotting Cn2
#using the PySonde package
#Christopher Phillips

#Import PySonde
from pysonde.pysonde import PySonde

#Import pyplot for plotting
import matplotlib.pyplot as pp

#Location of test sounding
sounding_path = "./example_sounding_nws.txt"

#Create sounding object
sonde = PySonde(sounding_path, "NWS")

#Compute Cn2
cn2 = sonde.calculate_Cn2()

#Plot Cn2
fig, ax = pp.subplots()
ax.plot(cn2, sonde.sounding["alt"], color="black")
ax.set_xlabel("Cn$^2$ [log$_{10}$(m$^{-2/3}$)]", fontsize=14, fontweight="bold")
ax.set_ylabel("Height [m]", fontsize=14, fontweight="bold")
pp.show()