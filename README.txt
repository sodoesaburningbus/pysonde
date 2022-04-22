This package contains contains the PySonde module
A PySonde class provides utilities for interfacing with several
common sounding formats, including cloud-based data from the university of Wyoming.
It also provides utilities for common applications such as plotting, CAPE calculations, and LLJ identification.
Further, once a sounding is loaded, it can be written out for
other applications such as input to the WRF SCM.
By default, the object supports unit aware computing, but the option exists to strip
units from the sounding.

The object only supports one sounding per file. If more are present, the reader will likely break. If it does not, use
the output at your own risk.

If this code is useful in a research project, please contact 
Christopher Phillips and provide an acknolwedgement in any publications.
If the software is a key part of the analysis, please consider offering co-authorship.

Any questions or suggestions should be directed to Christopher Phillips at chris.phillips@uah.edu

Written by Christopher Phillips
Atmopsheric and Earth Science Department
University of Alabama in Huntsville, June 2020

Module requirements
Python 3+
Matplotlib
MetPy 1+
NetCDF4
Numpy

------------------------------------INSTALLATION------------------------------------

The PySonde package may be installed with pip:

    pip install git+https://github.com/sodoesaburningbus/pysonde


Or manually:

  Get PySonde:
    git clone https://github.com/sodoesaburningbus/pysonde
    cd pysonde
    python3 setup.py install
    cd ..
  

------------------------------------CURRENT AND FUTURE SOUNDING FORMATS------------------------------------

Currently supports:
  NWS - NWS high density soundings
  CSWR - Center for Severe Weather Research L2 soundings
  EOL - National Center for Atmospheric Research - Earth Observing Laboratory soundings
  UAH - University of Alabama in Huntsville UPSTORM group soundings
  WRF - Weather Research and Forecasting Single Column Model Input Sounding
  WYO - University of Wyoming sounding file
  WYOWEB - University of Wyoming sounding online archive. (These are pulled from online.)
  IGRA2 - The IGRA2 online sounding archive. (These are pulled from online.)
  PSF - PySonde Sounding Format, a netCDF4 format more fully described in the documentation.
  CSV - CSV files originally outputted by PySonde

Future updates also plan to add support for the following sounding formats
  SPC sounding archive
  
------------------------------------CURRENT FEATURES------------------------------------

All methods and attributes are fully documented in the PySonde.py module file.
Summaries are presented here.
Currently, the PySonde object has the following attributes and methods:

  Attributes:
  
    General:
      fpath - The location of the sounding file
      sformat - The sounding format
      sounding - A dictionary containing the sounding variables with Pint (MetPy) units attached.
      sounding_units - A dictionary containing the units used by PySonde
      units - The MetPy unit object
    
    Thermo:
      lcl_alt - The height of the Lifting Condensation Level
      lcl_pres - The pressure of the Lifting Condensation Level
      lfc_alt - The height of the Level of Free Convection
      lfc_pres - The pressure of the Level of Free Convection
      lfc_temp - The environmental temperature at the Level of Free Convection
      parcel_path - The temeprature profile of a parcel lifted from the surface
      pw - Precipitable Water
      sfc_cape - The sounding's CAPE using a near-surface parcel
      sfc_cin - The sounding's CIN using a near-surface parcel
      
    PBLH:
      sbi - Whether a Surface-Based Inversion exists
      sbih - The depth of the Surface-Based Inversion
      sbih_ind - The sounding data index of the Surface-Based Inversion top
      sbih_pres - The pressure of the top of the Surface-Based Inversion
      pblh - The depth of the Planetary Boundary Layer
      pblh_ind - The sounding data index of the top of the Planetary Boundary Layer
      pblh_pres - The pressure at the top of the Planetary Boundary Layer
      


  Methods (For the user; others exist that are used internally by the object.):
  
    basic_skewt(nbarbs=None, llj=False, pblh=False, sbi=False) - Returns a MetPy SkewT figure object. options exist for
      highlighting the PBL top, LLJ, and any surface-based inversion. Wind barb spacing can also be passed in.
  
    calculate_Cn2() - Compute the index of refraction structure parameter Cn^2 using the methodology
      found in Fiorino and Meier 2016 "Improving the Fidelity of the Tatarskii Cn2 Calculation with Inclusion
      of Pressure Perturbations"
    
    calculate_gph(units=True) - Compute geopotential height from the sounding by integrating the hypsometric equation.
      By default returns geopotential height with untis attached.
    
    calculate_pres(units=True) - Compute the hydrostatic pressure profile by integrating hypsometric equation.
      By default returns the hydrostatic pressure with untis attached.
      
    calculate_layer_thickness(level1, level2) - Calculate the thickness between 2 pressure levels. Requires units on input.
    
    extract_level(level) - Extract observations for a single pressure level from the sounding. If units are not used in input, then pressure must be hPa.
    
    find_llj() - Outputs a dictionary contianing the altitude, pressure, speed, and direction of the low-level jet
      if present in a sounding. Also returns the category of the jet. Methodology follows that of Yang et al. 2020
      "Understanding irrigation impacts on low-level jets over the Great Plains" in Climate Dynamics"
    
    strip_units() - Return a dictionary containing the sounding without MetPy units attached.
    
    write_csv(save_path) - Converts sounding to an easily accessed CSV format at the specified file location.
    
    write_interpolated_csv(levs, save_path, coords="Z", write=True) - Interpolates a sounding to the desired coordinates
      in either height or pressure. Can optionally write to a file or output a dicionary with the inteprolated sounding.
      
    write_rams(save_path, maxz=None) - Write a RAMS input sounding.
    
    write_wrfscm(save_path) - Write a WRF single-column model input sounding
    
  sonde.

------------------------------------FUTURE UPDATES------------------------------------

Add installation via pip

Options to calculate mixed layer and most unstable CAPE

---------------------------------------------EXAMPLES---------------------------------------------

For full examples, please see the example/ directory. The creation of the PySonde object is clarified here.

Example creation of the PySonde object

    from pysonde.pysonde import PySonde
    sonde = PySonde(sounding_file_path, format_code, units=True)
    
Accessing the sounding data (e.g. temperature) may then be accomplished via:
    
    sonde.sounding["temp"]

sounding_file_path, string, path to file containing a single sounding
format_code, string, format of the sounding, see below for the options
units, boolean, optional, default=True, whether to attach units to variables via MetPy.
