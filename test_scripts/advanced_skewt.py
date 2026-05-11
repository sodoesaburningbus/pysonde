### This script tests the advanced skewt functionality.
### Christopher Phillips

# Imports
from datetime import datetime
import matplotlib.pyplot as pp
from pysonde.pysonde import PySonde

#sonde = PySonde('JAN', 'WEB', date=datetime(2023,3,25,0))
sonde = PySonde('FWD', 'WEB', date=datetime(2026,5,10,12))
#sonde = PySonde('sounding_BMX_2011042718.csv', 'CSV')

fig, skewt = sonde.advanced_skewt(nbarbs=3)
fig.savefig('test.png')
pp.close()