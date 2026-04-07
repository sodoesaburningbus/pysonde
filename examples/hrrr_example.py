# Import modules
from pysonde.pysonde import PySonde
from datetime import datetime
import numpy as np

# Settings
spath = 'hrrr_demo_sounding.csv'
latlon = (-97, 33.0)

# Grab the sounding
sonde = PySonde('hrrr.grib2', 'HRRR', date=datetime(2026,4,6,18), point=latlon)
sonde.write_csv(spath)