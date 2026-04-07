# Import packages
from pysonde.pysonde import PySonde
from datetime import datetime
import numpy as np

# Download sounding from Univ. of Wyoming archive
sonde = PySonde('BNA', 'WEB', date=datetime(2011,4,27,12))

# Perform an interpolation to pressure levels
plevs = np.arange(950,200,-50)
sonde.write_interpolated_csv(plevs, 'demo_interp_sounding.csv', coords='P')
