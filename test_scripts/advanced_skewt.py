### This script tests the advanced skewt functionality.
### Christopher Phillips

# Imports
import matplotlib.pyplot as pp
from pysonde.pysonde import PySonde

sonde = PySonde('../examples/example_sounding_nws.txt', 'NWS')

fig, skewt = sonde.advanced_skewt(nbarbs=20)
fig.savefig('test.png')
pp.close()