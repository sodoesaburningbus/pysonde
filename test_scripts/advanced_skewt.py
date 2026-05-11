### This script tests the advanced skewt functionality.
### Christopher Phillips

# Imports
from datetime import datetime
import matplotlib.pyplot as pp
from pysonde.pysonde import PySonde

sonde = PySonde('FWD', 'WEB', date=datetime(2026,5,10,12))
#sonde = PySonde('sounding_BMX_2011042718.csv', 'CSV')
print(sonde.mu_parcel)

fig, skewt = sonde.advanced_skewt(nbarbs=3)
fig.savefig('test.png')
pp.close()