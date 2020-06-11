### This module contains the PySonde class
### A PySonde class provides utilities for interfacing with several
### common sounding formats. It also provideas utilities for common
### applications such as plotting and CAPE calculations.
### Further, once a sounding is loaded, it can be written out for
### other applications such as input to the WRF SCM.
### By default, the object supports unit aware computing, but the option exists to strip
### units from the sounding.
###
### The object only supports one sounding per file. If more are present, only the first will be read in
### if the reader doesn't break.
###
### Written by Christopher Phillips
### University of Alabama in Huntsville, June 2020
###
### Currently supports:
###  NWS high density soundings
###  Center for Severe Weather Research L2 soundings
###  National Center for Atmospheric Research - Earth Observing Laboratory soundings
###
### Future updates plan to include support for the following sounding sources
###   University of Wyoming
###   WRF SCM input soundings
###
### Future updates also plan to add the following features
###   Options to calculate mixed layer and most unstable CAPE
###   Siphon compatibility to automatically download soundings from the web
###
### Module requirements
### Python 3+
### Matplotlib
### MetPy
### NetCDF4
### Numpy

### Importing required modules
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as pp
from metpy.units import units as mu
import metpy.calc as mc
from metpy.plots import SkewT
import netCDF4 as nc
import numpy

############################################################
#---------------------     PYSONDE     ---------------------#
############################################################
class PySonde:

    ### Constructor Method
    ### Inputs:
    ###  fpath, string, file path to sounding
    ###  sformat, string, options are: "NWS", "CSWR", "EOL" see ReadMe for full breakdown
    ###  units, optional, boolean, default=True, whether to attach units to sounding
    ###    note, if unit flag is false, units should be metric mks, but this is not guaranteed.
    ###
    ### Outputs:
    ###  None
    def __init__(self, fpath, sformat, units=True):

        #First attach file path, format, and units flag to object
        self.fpath = fpath
        self.sformat = sformat.upper()
        self.units = units

        #Attach correct units to sounding object as a dictionary
        self.sounding_units = {"time":mu.second, "pres":mu.hPa, "temp":mu.degC, "dewp":mu.degC,
            "uwind":mu.meter/mu.second, "vwind":mu.meter/mu.second, "lon":mu.deg, "lat":mu.deg, "alt":mu.meter}

        #Now read in the sounding based on it's format
        if (self.sformat == "NWS"): #NWS sounding
            self.read_nws()
        
        elif (self.sformat == "EOL"): #NCAR-EOL sounding
            self.read_eol()

        elif (self.sformat == "CSWR"): #CSWR sounding
            self.read_cswr()
        
        else: #Unrecognized format
            print("Unrecognized sounding format: ()".format(self.sformat))
            raise ValueError

        #Calculate the basic thermo propertie, (SFC CAPE/CIN, LCL, and Precipitable Water (PW)
        self.calculate_basic_thermo()

        #Returning
        return

    #####-----------METHODS TO CALCULATE SOUNDING PROPERTIES-----------#####

    #Method to calculate basic thermodynamic properties of the sounding.
    #Units are attached unless the user specifies otherwise
    def calculate_basic_thermo(self):

        #Calculate surface-based CAPE and CIN, LCL, and Precipitable Water (PW) from sounding
        if self.units: #If units attached

            #Precipitable Water
            self.pw = mc.precipitable_water(self.sounding["dewp"], self.sounding["pres"])

            #Lifting condensation level
            self.lcl_pres, self.lcl_temp = mc.lcl(self.sounding["pres"][0], self.sounding["temp"][0],
                self.sounding["dewp"][0])
            
            #Surface-based CAPE and CIN
            self.parcel_path = mc.parcel_profile(self.sounding["pres"], self.sounding["temp"][0],
                self.sounding["dewp"][0])
            self.sfc_cape, self.sfc_cin = mc.cape_cin(self.sounding["pres"], self.sounding["temp"],
                self.sounding["dewp"], self.parcel_path)

        else: #If no units
            #Precipitable Water
            pw = mc.precipitable_water(numpy.array(self.sounding["dewp"])*self.sounding_units["dewp"],
                numpy.array(self.sounding["pres"])*self.sounding_units["pres"])

            #Lifting condensation level
            lcl_pres, lcl_temp = mc.lcl(self.sounding["pres"][0]*self.sounding_units["pres"],
                self.sounding["temp"][0]*self.sounding_units["temp"],
                self.sounding["dewp"][0]*self.sounding_units["dewp"])
            
            #Surface-based CAPE and CIN
            parcel_path = mc.parcel_profile(numpy.array(self.sounding["pres"])*self.sounding_units["pres"],
                self.sounding["temp"][0]*self.sounding_units["temp"],
                self.sounding["dewp"][0]*self.sounding_units["dewp"])
            sfc_cape, sfc_cin = mc.cape_cin(numpy.array(self.sounding["pres"])*self.sounding_units["pres"],
                numpy.array(self.sounding["temp"])*self.sounding_units["temp"],
                numpy.array(self.sounding["dewp"])*self.sounding_units["dewp"], parcel_path)

            #Strip units from final quantities
            self.parcel_path = numpy.array(parcel_path, dtype="float")
            self.pw = numpy.array(pw, dtype="float")
            self.lcl_pres = numpy.array(lcl_pres, dtype="float")
            self.lcl_temp = numpy.array(lcl_temp, dtype="float")
            self.sfc_cape = numpy.array(sfc_cape, dtype="float")
            self.sfc_cin = numpy.array(sfc_cin, dtype="float")

        #Returning
        return

    #####-------------------METHODS TO PLOT SOUNDING-------------------#####

    ### Method to create an empty SkewT diagram
    ### Outputs:
    ###  fig, the pyplot figure object
    ###  skewt, the MetPy SkewT axis object
    def empty_skewt(self):

        #First create the figure and SkewT objects
        fig = pp.figure(figsize=(9,9))
        skewt = SkewT(fig, rotation=45)

        #Now set the limits
        skewt.ax.set_xlim(-40, 60)
        skewt.ax.set_ylim(1000, 100)

        #Add the adiabats, etc
        skewt.plot_dry_adiabats(t0=numpy.arange(-40, 200, 10)*self.sounding_units["temp"])
        skewt.plot_moist_adiabats()
        skewt.plot_mixing_lines(p=self.sounding["pres"])

        #Adjust the axis labels
        skewt.ax.set_xlabel("Temperature ('C)", fontsize=14, fontweight="bold")
        skewt.ax.set_ylabel("Pressure (hPa)", fontsize=14, fontweight="bold")

        #Returning
        return fig, skewt

    ### Method to plot a basic sounding
    ### Includes only temperature, dewpoint, and the parcel path
    ### Outputs:
    ###  fig, the pyplot figure object
    ###  skewt, the MetPy SkewT axis object
    def basic_skewt(self):

        #Create the empty SkewT
        fig, skewt = self.empty_skewt()

        #Plot the sounding
        skewt.plot(self.sounding["pres"], self.sounding["temp"], color="red")
        skewt.plot(self.sounding["pres"], self.sounding["dewp"], color="green")
        skewt.plot(self.sounding["pres"], self.parcel_path, color="black")
        skewt.plot_barbs(self.sounding["pres"], self.sounding["uwind"], self.sounding["vwind"])

        #Add the Release time and location to plot
        skewt.ax.set_title("Date: {}; Station: {}\nLon: {:.2f}; Lat: {:.2f}".format(
            self.release_time.strftime("%Y-%m-%d %H%M"), self.release_site, self.release_lon, self.release_lat),
            fontsize=14, fontweight="bold", horizontalalignment="left", x=0)

        #Returning
        return fig, skewt

    #####---------------METHODS TO READ SOUNDING FORMATS---------------#####
    
    ### Method to read CSWR soundings
    def read_cswr(self):
    
        #Set a flag to indicate still inside the file header
        header = True

        #Create a dictionary to hold the sounding
        keys = ["time", "pres", "temp", "dewp", "uwind", "vwind", "lon", "lat", "alt"]
        self.sounding = {}
        for k in keys:
            self.sounding[k] = []

        #Extra lists for wind before u and v calculation
        wdir = []
        wspd = []

        #Read the file
        fn = open(self.fpath)
        for line in fn:

            #Split line into columns
            dummy = line.split()

            #Check if still in header
            if header:

                #Skip unnecessary header lines
                if ((dummy[0] == "STN") or (dummy[0] == "NLVL")):
                    continue
                elif (dummy[0] == "P"):
                    header = False
                    continue

                #Pull metadata
                self.release_site = dummy[0]
                self.release_time = datetime.strptime("{}{}".format(dummy[1], dummy[2]), "%Y%m%d%H%M")
                if self.units: #Test for unit flag
                    self.release_lon = float(dummy[5])*mu.deg
                    self.release_lat = float(dummy[4])*mu.deg
                    self.release_elv = float(dummy[3])*mu.meter
                else:
                    self.release_lon = float(dummy[5])*mu.deg
                    self.release_lat = float(dummy[4])*mu.deg
                    self.release_elv = float(dummy[3])*mu.meter

            #Now for the data rows
            else:
                
                self.sounding["time"].append(numpy.nan)
                self.sounding["pres"].append(dummy[0])
                self.sounding["alt"].append(dummy[1])
                self.sounding["temp"].append(dummy[2])
                self.sounding["dewp"].append(dummy[3])
                wdir.append(dummy[4])
                wspd.append(dummy[5])
                self.sounding["lon"].append(dummy[11])
                self.sounding["lat"].append(dummy[12])

        #Close file
        fn.close()

        #Now calculate uwind and vwind
        wspd = numpy.array(wspd, dtype="float")
        wdir = numpy.array(wdir, dtype="float")
        self.sounding["uwind"] = wspd*numpy.cos((270-wdir)*numpy.pi/180.0)
        self.sounding["vwind"] = wspd*numpy.sin((270-wdir)*numpy.pi/180.0)

        #Now convert the other variables to arrays and attach units
        units = [mu.second, mu.hPa, mu.degC, mu.degC, mu.meter/mu.second,
            mu.meter/mu.second, mu.deg, mu.deg, mu.meter]
        if self.units: #Default case, units are attached
            for k in keys:
                self.sounding[k] = numpy.array(self.sounding[k], dtype="float")*self.sounding_units[k]
        else: #No units
            for k in keys:
                self.sounding[k] = numpy.array(self.sounding[k], dtype="float")

        #Returning
        return

    ### Method to read NCAR-EOL soundings
    def read_eol(self):
        
        #Load the NetCDF file for reading
        fn = nc.Dataset(self.fpath)

        #Grab the launch location and time
        try:
            self.release_site = fn.StationName
        except:
            self.release_site = fn.site_id
        
        try:
            self.release_time = datetime.strptime(fn.BalloonReleaseDateAndTime, "%Y-%m-%dT%H:%M:%S")
        except:
            self.release_time = datetime.strptime(fn.launch_status.split("\n")[3], "%y%m%d %H%M\r")
            
        if self.units: #Attach units
            self.release_lon = fn.variables["lon"][:][0]*mu.deg
            self.release_lat = fn.variables["lat"][:][0]*mu.deg
            self.release_elv = fn.variables["alt"][:][0]*mu.meter
        else: #no units

            self.release_lon = fn.variables["lon"][:][0]
            self.release_lat = fn.variables["lat"][:][0]
            self.release_elv = fn.variables["alt"][:][0]

        #Create a dictionary to hold the sounding
        self.sounding = {}

        #Read in the sounding data
        skeys = ["time", "pres", "temp", "dewp", "uwind", "vwind", "lon", "lat", "alt"]
        fkeys = ["time", "pres", "tdry", "dp", "u_wind", "v_wind", "lon", "lat", "alt"]
        for [sk, fk] in zip(skeys, fkeys):
            if self.units: #Attach units
                self.sounding[sk] = numpy.array(fn.variables[fk][:])*self.sounding_units[sk]
            
            else: #No units
                self.sounding[sk] = fn.variables[fk][:]

        #Close the netcdf file
        fn.close()

        #Returning
        return

    ### Method to read NWS soundings
    def read_nws(self):
        
        #Set a flag to indicate still inside the file header
        header = True

        #Create a dictionary to hold the sounding
        keys = ["time", "pres", "temp", "dewp", "uwind", "vwind", "lon", "lat", "alt"]
        self.sounding = {}
        for k in keys:
            self.sounding[k] = []

        #Read the file
        fn = open(self.fpath)
        for line in fn:

            #First check if still inside the header
            #to pull metadata
            if header:
                if ("Release Site" in line):
                    self.release_site = line.split()[4]

                elif ("Release Location" in line):
                    dummy = line.split()
                    if self.units: #Test for unit flag
                        self.release_lon = float(dummy[7].strip(","))*mu.deg
                        self.release_lat = float(dummy[8].strip(","))*mu.deg
                        self.release_elv = float(dummy[9].strip(","))*mu.meter
                    else:
                        self.release_lon = float(dummy[7].strip(","))
                        self.release_lat = float(dummy[8].strip(","))
                        self.release_elv = float(dummy[9].strip(","))
                
                elif ("UTC Release Time" in line):
                    dummy = line[line.find(":")+1:]
                    self.release_time = datetime.strptime(dummy.strip(), "%Y, %m, %d, %H:%M:%S")
                
                elif ("------ ------" in line):
                    header = False

            #Read in data if outside header
            else:
                #Check that we're still reading dataq and not a second sounding header
                try:
                    test = float(line.split()[0])
                except:
                    break                

                #Parse the line                
                dummy = line.split()
                self.sounding["time"].append(dummy[0])
                self.sounding["pres"].append(dummy[1])
                self.sounding["temp"].append(dummy[2])
                self.sounding["dewp"].append(dummy[3])
                self.sounding["uwind"].append(dummy[5])
                self.sounding["vwind"].append(dummy[6])
                self.sounding["lon"].append(dummy[10])
                self.sounding["lat"].append(dummy[11])
                self.sounding["alt"].append(dummy[14])

        #Close file
        fn.close()

        #Once the data has been read in, convert everything to numpy arrays and attach units
        if self.units: #Default case, units are attached
            for k in keys:
                self.sounding[k] = numpy.array(self.sounding[k], dtype="float")*self.sounding_units[k]
        else: #No units
            for k in keys:
                self.sounding[k] = numpy.array(self.sounding[k], dtype="float")

        #Returning
        return















