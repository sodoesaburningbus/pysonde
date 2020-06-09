This package contains contains the PySonde module
A PySonde class provides utilities for interfacing with several
common sounding formats. It also provideas utilities for common
applications such as plotting and CAPE calculations.
Further, once a sounding is loaded, it can be written out for
other applications such as input to the WRF SCM.
By default, the object supports unit aware computing, but the option exists to strip
units from the sounding.

The object only supports one sounding per file. If more are present, only the first will be read in
if the reader doesn't break.

Written by Christopher Phillips
University of Alabama in Huntsville, June 2020

Module requirements
Python 3+
Matplotlib
MetPy
NetCDF4
Numpy

------------------------------------CURRENT AND FUTURE FEATURES------------------------------------

Currently supports:
 NWS high density soundings
 Center for Severe Weather Research L2 soundings
 National Center for Atmospheric Research - Earth Observing Laboratory soundings

Future updates plan to include support for the following sounding sources
  University of Wyoming
  WRF SCM input soundings

Future updates also plan to add the following features
  Options to calculate mixed layer and most unstable CAPE
  Siphon compatibility to automatically download soundings from the web
  Plotting of detailed soundings

---------------------------------------------EXAMPLES---------------------------------------------

For full examples, please see the example/ directory. The creation of the PySonde object is clarified here.

Example creation of the PySonde object

    from pysonde.pysonde import PySonde
    sonde = PySonde(sounding_file_path, format_code, units=True)

sounding_file_path, string, path to file containing a single sounding
format_code, string, format of the sounding, see below for the options
units, boolean, optional, default=True, whether to attach units to variables via MetPy.

The available sounding formats at this time are:

"NWS" - National Weather Service soundings
"CSWR" - Center for Severe Research L2 soundings
"EOL" - National Center for Atmospheric Research, Earth Oberving Laboratory soundings
