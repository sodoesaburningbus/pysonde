### This script tests the advanced skewt functionality.
### Christopher Phillips

# Imports
from datetime import datetime
import matplotlib.pyplot as pp
from pysonde.pysonde import PySonde

sonde = PySonde('BMX', 'WEB', date=datetime(2011,4,27,18))
#sonde = PySonde('sounding_BMX_2011042718.csv', 'CSV')

fig, skewt = sonde.advanced_skewt(nbarbs=3)
fig.savefig('test.png')
pp.close()