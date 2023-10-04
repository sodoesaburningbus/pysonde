### This script demonstrates the Locator module
### Note: requires internet connection

### Import modules
from pysonde.pysonde import PySonde
from pysonde.Locator import locate_IGRA2
from pysonde import severewx as pwx
from datetime import datetime

### Locate a station
### Inputs are LON, LAT, date
date = datetime(2011, 4, 27, 18)
station_id = locate_IGRA2(-86.7, 33, date)

print(station_id)

# Download the sounding
sonde = PySonde(station_id, 'IGRA2', date=date)
sonde.write_csv('test.csv')

print('WBI', pwx.get_wbi(sonde))
print('SCP', pwx.get_scp(sonde))
print('STP', pwx.get_stp(sonde))
print('LCL', sonde.lcl_pres)
print('PW', sonde.pw)
print('SCAPE', sonde.sfc_cape)
print('MUCAPE', sonde.mu_cape)

exit()