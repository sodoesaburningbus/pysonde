### This script tests the advanced skewt functionality.
### Christopher Phillips

# Imports
import matplotlib.pyplot as pp
from pysonde.pysonde import PySonde

sonde = PySonde('sounding_BMX_2011042718.csv', 'CSV')

fig, skewt = sonde.advanced_skewt(nbarbs=3)
fig.savefig('test.png')
pp.close()