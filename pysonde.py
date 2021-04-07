### This module contains the PySonde class
### A PySonde object provides utilities for interfacing with several
### common sounding formats. It also provideas utilities for common
### applications such as plotting and CAPE calculations.
### Further, once a sounding is loaded, it can be written out for
### other applications such as input to the WRF SCM.
### By default, the object supports unit aware computing, but the option exists to strip
### units from the sounding.
###
### The object only supports one sounding per file. If more are present, only the first
### will be read in, if the reader doesn't break.
###
### Written by Christopher Phillips
### University of Alabama in Huntsville, June 2020
###
### Currently supports:
###  NWS - NWS high density soundings
###  CSWR - Center for Severe Weather Research L2 soundings
###  EOL - National Center for Atmospheric Research - Earth Observing Laboratory soundings
###  WEB - University of Wyoming sounding online archive (these are pulled from online)
###  WRF - Weather Research and Forecasting Single Column Model Input Sounding
###  WYO - University of Wyoming sounding file
###
### Future updates also plan to add the following features
###   Options to calculate mixed layer and most unstable CAPE
###
### Module requirements
### Python 3+
### Matplotlib
### MetPy 1.0+
### NetCDF4
### Numpy

### Importing required modules
import atmos.math as am
import atmos.thermo as at
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as pp
from metpy.units import units as mu
import metpy.calc as mc
from metpy.plots import SkewT
import netCDF4 as nc
import numpy
from siphon.simplewebservice.wyoming import WyomingUpperAir

############################################################
#---------------------     PYSONDE     ---------------------#
############################################################
class PySonde:

    ### Constructor Method
    ### Inputs:
    ###  fpath, string, file path to sounding. In the case of the web service, this is the station identifier
    ###  sformat, string, options are: "NWS", "CSWR", "EOL", "WEB" see ReadMe for full breakdown
    ###  date, optional, required datetime object for Siphon web service sounding. (UTC)
    ###
    ### Outputs:
    ###  None
    def __init__(self, fpath, sformat, date=None):

        #First attach file path, format, and units flag to object
        self.fpath = fpath
        self.sformat = sformat.upper()

        #Attach correct units to sounding object as a dictionary
        self.sounding_units = {"time":mu.second, "pres":mu.hPa, "temp":mu.degC, "dewp":mu.degC, "mixr":mu.g/mu.kg,
            "uwind":mu.meter/mu.second, "vwind":mu.meter/mu.second, "lon":mu.deg, "lat":mu.deg, "alt":mu.meter}

        #Attach MetPy unit object so users have access
        self.units = mu

        #Initialize sounding dictionaries for both unitless and with units
        self.sounding = {} #Sounding with units attached
        self.sounding_uf = {} #Unit free sounding
        for k in self.sounding_units.keys():
            self.sounding[k] = []

        #Now read in the sounding based on it's format
        if (self.sformat == "NWS"): #NWS sounding
            self.read_nws()
        
        elif (self.sformat == "EOL"): #NCAR-EOL sounding
            self.read_eol()

        elif (self.sformat == "CSWR"): #CSWR sounding
            self.read_cswr()
            
        elif (self.sformat == "WYO"): #University of Wyoming sounding
            self.read_wyoming()
            
        elif (self.sformat == "WEB"): #Pull sounding from internet
            self.read_web(date)
        
        elif (self.sformat == "WRF"): #WRF SCM input sounding
            self.read_wrfscm()
        
        else: #Unrecognized format
            raise ValueError("Unrecognized sounding format: ()".format(self.sformat))

        #Add mixing ratio to the sounding
        unitless = self.strip_units()
        e = at.sat_vaporpres(unitless["dewp"]+273.15)
        p = unitless["pres"]
        self.sounding["mixr"] = at.etow(p*100.0,e)*1000.0*mu.g/mu.kg

        #Calculate the basic thermo properties, (SFC CAPE/CIN, LCL, and Precipitable Water (PW)
        self.calculate_basic_thermo()
        
        #Calculate planetary boundary layer height
        self.calculate_pblh()

        #Returning
        return

    #####-----------METHODS TO CALCULATE SOUNDING PROPERTIES-----------#####
    
    #Method to calculate basic thermodynamic properties of the sounding.
    def calculate_basic_thermo(self):

        #Enclose in try, except because not every sounding will have a converging parcel path or CAPE.
        try:
        
            #Precipitable Water
            self.pw = mc.precipitable_water(self.sounding["pres"], self.sounding["dewp"])

            #Lifting condensation level
            self.lcl_pres, self.lcl_temp = mc.lcl(self.sounding["pres"][0], self.sounding["temp"][0],
                self.sounding["dewp"][0])

            #Surface-based CAPE and CIN
            self.parcel_path = mc.parcel_profile(self.sounding["pres"], self.sounding["temp"][0],
                self.sounding["dewp"][0])
            self.sfc_cape, self.sfc_cin = mc.cape_cin(self.sounding["pres"], self.sounding["temp"],
                self.sounding["dewp"], self.parcel_path)

        #Do this when parcel path fails to converge
        except Exception as e:
            print("WARNING: No LCL, CAPE, or PW stats because:\n{}.".format(e))
            self.parcel_path = numpy.nan
            self.pw = numpy.nan
            self.lcl_pres = numpy.nan
            self.lcl_temp = numpy.nan
            self.sfc_cape = numpy.nan
            self.sfc_cin = numpy.nan
                
        #Returning
        return

    #Method to calculate geopotential height from sounding
    #Integrates the hypsomteric equation using a 3rd order
    #Adams-Bashforth scheme.
    #Inputs,
    # units, optional, boolean, whether to return geopotential height profile with units attached
    #        defaults to True
    def calculate_gph(self, units=True):
    
        #Strip units from sounding while integrating
        sounding = self.strip_units()
        
        #Convert units
        pres = sounding["pres"]*100.0 #hPa -> Pa
        temp = sounding["temp"]+273.15 #'C -> K
        mixr = sounding["mixr"]/1000.0 #g/kg -> kg/kg
                
        #Calculate virtual temperature
        vtemp = at.virt_temp(temp, mixr)
        
        #If first element isn't finite, then replace with temperature
        if (not numpy.isfinite(vtemp[0])):
            vtemp[0] = temp[0]
                
        #Loop across the soudning profile while summing layer thickness
        z = [0]
        j = 1
        for i in range(1, pres.size):
        
            tvbar = am.layer_interp(pres[i-j], pres[i], (pres[i-j]+pres[i])/2.0, vtemp[i-j], vtemp[i])
            dz = at.hypsometric(pres[i-j], pres[i], tvbar)
            z.append(z[-j]+dz)
            
            #For handling layers with missing data
            if (not numpy.isfinite(dz)):
                j += 1
            else:
                j = 1

        #Convert to numpy array and add release elevation
        if units:
            gph = numpy.array(z)*self.sounding_units["alt"]+self.release_elv
        else:
            gph = numpy.array(z)+numpy.array(self.release_elv/self.sounding_units["alt"])
        
        return gph

    #Method to calculate planetary boundary layer height (PBLH) from the sounding
    #The algorithm finds the first location where the environmental virtual potential temperature is greater
    #than the surface. If a surface inversion is present, than the 
    #If the PBLH cannot be calculated, it is set to -1
    def calculate_pblh(self):
    
        try:
            #Set Surbace based inversion flag to false initially
            self.sbi = False
            self.sbih = numpy.nan
            self.sbih_pres = numpy.nan
            self.sbih_ind = None
        
            #Strip units from sounding for use with atmos package
            sounding = self.strip_units()
            
            #Calculate sounding potential temperature, temperature gradient, and height AGL
            height = (sounding["alt"]-self.release_elv/self.sounding_units["alt"]) #Height AGL
            tgrad = numpy.gradient(sounding["temp"])
            theta = at.pot_temp(sounding["pres"]*100.0, sounding["temp"]+273.15)                        
                        
            #Locate elevated temperature inversion, call that the PBLH
            ind_pbl = numpy.where(tgrad > 0.0)[0][0]

            #If a surface based inversion exists, then look for the top of the inversion
            #Re-calculate the PBLH for the overlying remnant layer
            while (height[ind_pbl] < 100.0):
            
                #Handle SBI
                self.sbi = True
                ind_sbi = numpy.where(tgrad[ind_pbl:] <= 0.0)[0][0]+ind_pbl
                self.sbih = height[ind_sbi]*self.sounding_units["alt"]
                self.sbih_pres = sounding["pres"][ind_sbi]*self.sounding_units["pres"]
                self.sbih_ind = ind_sbi
                
                #Re-locate PBL top
                ind_pbl = numpy.where(tgrad[ind_sbi:] > 0.0)[0][0]+ind_sbi                

            #If no inversion exists below 600 hPa, then use the mixing method
            if (sounding["pres"][ind_pbl] < 600.0):
            
                #Locate first location where surface parcel potential temp is less than environment
                ind_pbl = 0
                ind = 0
                while (ind_pbl < 2): #To avoid getting stuck at the surface
                    ind_pbl = numpy.where(theta[ind] < theta)[0][0]
                    ind += 1
            
            #Retreive PBL top height and pressure
            pblh = height[ind_pbl]*self.sounding_units["alt"]
            pblh_pres = sounding["pres"][ind_pbl]*self.sounding_units["pres"]
                    
            #Store PBLH as attribute of sounding
            self.pblh = pblh
            self.pblh_pres = pblh_pres
            self.pblh_ind = ind_pbl
           
        except Exception as err:
            print("WARNING: PBL top could not be found due to:\n{}".format(err))
            self.pblh = numpy.nan
            self.pblh_pres = numpy.nan
            self.pblh_ind = None
    
        #Returning
        return
        
    #Method to calculate thickness between two
    #atmospheric levels
    #Inputs:
    # layer1, float (with units), pressure of lower level
    # layer2, float (with units), pressure of upper level
    #Outputs:
    # thickness, float (with units), thickness between levels
    def calculate_layer_thickness(self, layer1, layer2):
    
        #First convert units to be consistent with the sounding
        layer1 = layer1.to(self.sounding_units["pres"])
        layer2 = layer2.to(self.sounding_units["pres"])
        
        #Locate level in sounding closest to each level
        ind1 = numpy.argmin((self.sounding["pres"]-layer1)**2)
        ind2 = numpy.argmin((self.sounding["pres"]-layer2)**2)
        
        #Bracket the desired levels in the sounding
        if (self.sounding["pres"][ind1] > layer1):
            indb1 = ind1
            indt1 = ind1+1
        else:
            indb1 = ind1-1
            indt1 = ind1
        if (self.sounding["pres"][ind2] > layer2):
            indb2 = ind2
            indt2 = ind2+1
        else:
            indb2 = ind2-1
            indt2 = ind2
            
        #Calculate the layer thickness and return
        z1 = am.layer_interp(self.sounding["pres"][indb1], self.sounding["pres"][indt1], layer1,
            self.sounding["alt"][indb1], self.sounding["alt"][indt1])
        z2 = am.layer_interp(self.sounding["pres"][indb2], self.sounding["pres"][indt2], layer2,
            self.sounding["alt"][indb2], self.sounding["alt"][indt2])
        return z2-z1
    
    #Method to extract sounding variables at a single level
    #This method will interpolate between the two nearest levels.
    #Inputs:
    # level, float, pressure level (hPa if no unit attached) for whcih to pull values
    #Outputs
    # data, dictionary keyed with sounding variables, contains vallues at
    #       requested level
    def extract_level(self, level):
        
        #Force level unit to same as sounding
        try:
            level.to(self.sounding_units["pres"])
        except:
            level = (level*100.0*mu.Pa).to(self.sounding_units["pres"])
    
        #Create dictionary to hold data
        data = {}
    
        #Locate nearest levels that bracket the desired level
        ind = numpy.argmin(abs(self.sounding["pres"]-level))
        if (self.sounding["pres"][ind] >= level):
            tind = ind+1
            bind = ind
        elif (self.sounding["pres"][ind] < level):
            tind = ind
            bind = ind-1
            
        {"time":mu.second, "pres":mu.hPa, "temp":mu.degC, "dewp":mu.degC,
            "uwind":mu.meter/mu.second, "vwind":mu.meter/mu.second, "lon":mu.deg,
            "lat":mu.deg, "alt":mu.meter}
            
        #Perfrom the interpolations
        for k in self.sounding.keys():
        
            data[k] = am.layer_interp(self.sounding["pres"][bind], self.sounding["pres"][tind],
                level, self.sounding[k][bind], self.sounding[k][tind])
                
        #Return the extracted level    
        return data

    ### Method to find low-level jet
    #This method locates any Low-Level Jet (LLJ) and
    #determines its strength and location.
    #The criteria used in this method follows that of Yang et al. 2020
    #in their study "Understanding irrigation impacts on low-level jets
    #over the Great Plains" in Climate Dynamics.    
    #Inputs:
    # None
    #Outputs:
    # jet, dictionary containing the jet characteristics with the following keys:
    #  "alt" - float, height (m) of the jet.
    #  "pres" - float, height (hPa) of the jet.
    #  "wspd" - speed (m/s) of the jet.
    #  "wdir" - direction (met. deg.) of the jet.
    #  "category" - integer, type of LLJ following the methodology of Yang et al. 2020.
    #    [-1, 3], with -1 being no LLJ and 3 being the strongest LLJ.
    #  "falloff" - float, difference between the LLJ speed maximum and the above minimum.
    # If no LLJ is present, the dictionary contains NaNs and the category is -1.
    def find_llj(self):

        #Create the jet dictionary
        jet = {}

        #Calculate total wind profile
        wspd = numpy.sqrt(self.sounding["uwind"]**2+self.sounding["vwind"]**2)
        wdir = (270.0*self.units.degrees-
            numpy.arctan2(self.sounding["vwind"]/self.sounding_units["vwind"],
            self.sounding["uwind"]/self.sounding_units["uwind"])*180.0/numpy.pi*
            self.units.degrees)
            
        #Eliminate all values above 700 hPa
        mask = self.sounding["pres"] >= 700*self.units.hPa
        wspd = wspd[mask]
        wdir = wdir[mask]
        
        #Locate the maximum wind speed
        maxind = numpy.argmax(wspd)
        wmax = wspd[maxind]
        
        #Locate the minimum above the jet core
        minind = numpy.argmin(wspd[maxind:])
        wmin = wspd[maxind:][minind]
        
        #Assign the catgeory
        ms = self.units.meter/self.units.second #Speed units
        if ((wmax >= 20*ms) and (wmax-wmin >= 10*ms)): #Mighty LLJ
            jet["category"] = 3
        
        elif ((wmax >= 16*ms) and (wmax-wmin >= 8*ms)): #Strong LLJ
            jet["category"] = 2
        
        elif ((wmax >= 12*ms) and (wmax-wmin >= 6*ms)): #Moderate LLJ
            jet["category"] = 1
        
        elif ((wmax >= 10*ms) and (wmax-wmin >= 5*ms)): #Weak LLJ
            jet["category"] = 0
        
        else: #No LLJ
            jet["category"] = -1
            jet["alt"] = numpy.nan*self.sounding_units["alt"]
            jet["pres"] = numpy.nan*self.sounding_units["pres"]
            jet["wspd"] = numpy.nan*self.sounding_units["uwind"]
            jet["wdeg"] = numpy.nan*self.units.degrees
            jet["falloff"] = numpy.nan*self.sounding_units["uwind"]
            
            #Returning if no jet
            return jet
            
        #Fill the dictionary and return
        jet["alt"] = self.sounding["alt"][mask][maxind]
        jet["pres"] = self.sounding["pres"][mask][maxind]
        jet["wspd"] = wmax
        jet["wdir"] = wdir
        jet["falloff"] = wmax-wmin
        
        #Return jet characteristics
        return jet

    #####-------------------METHODS TO PLOT OR OUTPUT SOUNDING-------------------#####

    ### Method to plot a basic sounding
    ### Includes only temperature, dewpoint, and the parcel path
    ### Inputs:
    ###  nbarbs, optional, integer, spacing between wind barbs
    ###  llj, optional, boolean, whether to check for a Low-level jet and highlight it.
    ###       Defaults to False.
    ###  pblh, optional, boolean, whether to plot the PBLH height. Defaults to False.
    ### Outputs:
    ###  fig, the pyplot figure object
    ###  skewt, the MetPy SkewT axis object
    def basic_skewt(self, nbarbs=None, llj=False, pblh=False, sbi=False):

        #Create the empty SkewT
        fig, skewt = self.empty_skewt()

        #Plot the sounding
        skewt.plot(self.sounding["pres"], self.sounding["temp"], color="red")
        skewt.plot(self.sounding["pres"], self.sounding["dewp"], color="green")
        try:
            skewt.plot(self.sounding["pres"], self.parcel_path, color="black")
        except:
            pass
            
        if (nbarbs == None):
            skewt.plot_barbs(self.sounding["pres"], self.sounding["uwind"], self.sounding["vwind"])
        else:
            skewt.plot_barbs(self.sounding["pres"][::nbarbs], self.sounding["uwind"][::nbarbs], self.sounding["vwind"][::nbarbs])

        #Add the Release time and location to plot
        skewt.ax.set_title("Date: {}; Station: {}\nLon: {:.2f}; Lat: {:.2f}".format(
            self.release_time.strftime("%Y-%m-%d %H%M"), self.release_site, self.release_lon, self.release_lat),
            fontsize=14, fontweight="bold", horizontalalignment="left", x=0)

        #Add jet highlight
        if llj:
            jet = self.find_llj()
            if (jet["category"] != -1):
                skewt.ax.axhline(jet["pres"], color="blue")
        
        #Add PBLH
        if pblh:
            skewt.ax.axhline(self.pblh_pres, color="black", linestyle="--")
            
        #Add SBI
        if (sbi and self.sbi):
            skewt.ax.axhline(self.sbih_pres, color="black", linestyle=":")

        #Returning
        return fig, skewt

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
        try:
            skewt.plot_mixing_lines(pressure=self.sounding["pres"])
        except:
            skewt.plot_mixing_lines(p=self.sounding["pres"])

        #Adjust the axis labels
        skewt.ax.set_xlabel("Temperature ('C)", fontsize=14, fontweight="bold")
        skewt.ax.set_ylabel("Pressure (hPa)", fontsize=14, fontweight="bold")

        #Returning
        return fig, skewt
    
    ### Method to output sounding to a text file
    ### Inputs:
    ###  spath, string, full path on which to save file
    def write_csv(self, spath):
    
        #First strip all units from sounding
        unitless = self.strip_units()
        
        {"time":mu.second, "pres":mu.hPa, "temp":mu.degC, "dewp":mu.degC, "mixr":mu.g/mu.kg,
            "uwind":mu.meter/mu.second, "vwind":mu.meter/mu.second, "lon":mu.deg, "lat":mu.deg, "alt":mu.meter}
        
        #Open file and write header
        fn = open(spath, "w")
        header = "Time (s)\tGPH (m)\tPrs (hPa)\tTemp ('C)\tMixR (g/kg)\tDewp ('C)\tUspd (m/s)\tVspd (m/s)\tLon\tLat".split("\t")
        fn.write(("{:>12}"+",{:>12}"*(len(header)-1)).format(*header))
        
        #Now loop through the data
        for i in range(unitless["pres"].size):
            data = (unitless["time"][i], unitless["alt"][i], unitless["pres"][i], unitless["temp"][i],
                unitless["mixr"][i], unitless["dewp"][i], unitless["uwind"][i], unitless["vwind"][i],
                unitless["lon"][i], unitless["lat"][i])
            fn.write(("\n{:>12.2f}"+",{:>12.2f}"*(len(header)-1)).format(*data))
            
        #Close the file
        fn.close()
    
        #Returning
        return

    ### Method to output WRF SCM input sounding
    ### Inputs:
    ###  spath, string, location to save file to
    def write_wrfscm(self, spath):
    
        #First strip all units from sounding
        unitless = self.strip_units()
        
        #Do necessary unit conversions
        heights = unitless["alt"]
        temp = unitless["temp"]+273.15 #C -> K
        pres = unitless["pres"]*100.0 #hPa -> Pa
        
        #Calculate necessary surface variables
        #First compute 10m winds from the sounding using linear interpolation
        ind2 = (numpy.arange(heights.size, dtype="int")[(heights-10)>0])[0]
        ind1 = ind2-1
        weight = (10.0-heights[ind1])/(heights[ind2]-heights[ind1])
        u10 = unitless["uwind"][ind1]*(1-weight)+unitless["uwind"][ind2]*weight
        v10 = unitless["vwind"][ind1]*(1-weight)+unitless["vwind"][ind2]*weight
        
        #Calculate potential temperature and mixing ratio at each level of the sounding
        theta = at.pot_temp(pres, temp)
        qvapor = at.etow(pres, at.sat_vaporpres(unitless["dewp"]+273.15))
        
        #Check that sounding isn't longer than what WRF allows (1000 lines max)
        if (pres.size > 999): #Downsample to a reasonable number
            #Determine pressure levels to keep
            pmid = numpy.linspace(pres[1], pres[-2], 900) #Only using 900 levels, to give WRF plenty of space
            bind = numpy.array(list((numpy.arange(0, pres.size, dtype="int")[(pres-pm)>0][-1] for pm in pmid)))
            tind = numpy.array(list((numpy.arange(0, pres.size, dtype="int")[(pres-pm)<=0][0] for pm in pmid)))
            new_pres = am.layer_interp(pres[bind], pres[tind], pmid, pres[bind], pres[tind])
            new_theta = am.layer_interp(pres[bind], pres[tind], pmid, theta[bind], theta[tind])
            new_uwind = am.layer_interp(pres[bind], pres[tind], pmid, unitless["uwind"][bind], unitless["uwind"][tind])
            new_vwind = am.layer_interp(pres[bind], pres[tind], pmid, unitless["vwind"][bind], unitless["vwind"][tind])
            new_qvapor = am.layer_interp(pres[bind], pres[tind], pmid, qvapor[bind], qvapor[tind])
            new_height = am.layer_interp(pres[bind], pres[tind], pmid, heights[bind], heights[tind])
        else:
            new_pres = pres
            new_theta = theta
            new_uwind = unitless["uwind"]
            new_vwind = unitless["vwind"]
            new_qvapor = qvapor
            new_height = heights
        
        #Write everything to the output file
        fn = open(spath, "w")
        fn.write("{} {} {} {} {} {}".format(unitless["release_elv"], u10, v10, theta[0], qvapor[0], pres[0]))
        for i, h in enumerate(new_height[1:]):
            fn.write("\n{} {} {} {} {}".format(h, new_uwind[i], new_vwind[i], new_theta[i], new_qvapor[i]))
        fn.close()
        
        #Returning
        return

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
                try: #header isn't always in the same order
                    self.release_site = dummy[0]
                    self.release_time = datetime.strptime("{}{}".format(dummy[1], dummy[2]), "%Y%m%d%H%M")
                    self.release_lon = float(dummy[5])*mu.deg
                    self.release_lat = float(dummy[4])*mu.deg
                    self.release_elv = float(dummy[3])*mu.meter
                except:
                    self.release_site = dummy[-1]
                    try:
                        self.release_time = datetime.strptime("{}{}".format(dummy[0], dummy[1]), "%y%m%d%H%M")
                    except:
                        self.release_time = datetime.strptime("{}{}".format(dummy[1], dummy[2]), "%y%m%d%H%M")
                    self.release_lon = float(dummy[-2])*mu.deg
                    self.release_lat = float(dummy[-3])*mu.deg
                    self.release_elv = float(dummy[-4])*mu.meter

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
        
        #Replace missing wind values with Nans
        self.sounding["uwind"][wspd == -999] = numpy.nan
        self.sounding["vwind"][wspd == -999] = numpy.nan

        #Now convert the other variables to arrays and attach units
        #And eliminate missing values
        for k in keys:
            self.sounding[k] = numpy.array(self.sounding[k], dtype="float")
            self.sounding[k][self.sounding[k] == -999] = numpy.nan
            self.sounding[k] *= self.sounding_units[k]

        #Ensure that heights are AMSL and not AGL
        if (self.sounding["alt"][0] < self.release_elv):
            self.sounding["alt"] += self.release_elv

        #Returning
        return

    ### Method to read NCAR-EOL soundings
    def read_eol(self):
        
        #Load the NetCDF file for reading
        fn = nc.Dataset(self.fpath)
        
        #Grab the missing value
        missing_value = fn.variables["dp"].missing_value

        #Grab the launch location and time
        try:
            self.release_site = fn.StationName
        except:
            self.release_site = fn.site_id
        
        try:
            self.release_time = datetime.strptime(fn.BalloonReleaseDateAndTime, "%Y-%m-%dT%H:%M:%S")
        except:
            self.release_time = datetime.strptime(fn.launch_status.split("\n")[3], "%y%m%d %H%M\r") 

        self.release_lon = fn.variables["lon"][:][0]*mu.deg
        self.release_lat = fn.variables["lat"][:][0]*mu.deg
        self.release_elv = fn.variables["alt"][:][0]*mu.meter

        #Create a dictionary to hold the sounding
        self.sounding = {}

        #Read in the sounding data and replace missing data with NaNs
        skeys = ["time", "pres", "temp", "dewp", "uwind", "vwind", "lon", "lat", "alt"]
        fkeys = ["time", "pres", "tdry", "dp", "u_wind", "v_wind", "lon", "lat", "alt"]
        for [sk, fk] in zip(skeys, fkeys):
            self.sounding[sk] = numpy.array(fn.variables[fk][:])*self.sounding_units[sk]
            self.sounding[sk][self.sounding[sk] == missing_value*self.sounding_units[sk]] = numpy.nan*self.sounding_units[sk]
            
        #Close the netcdf file
        fn.close()

        #Ensure that heights are AMSL and not AGL
        if (self.sounding["alt"][0] < self.release_elv):
            self.sounding["alt"] += self.release_elv

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

            #First check if w
            if header:
                if ("Release Site" in line):
                    self.release_site = line.split()[4]

                elif ("Release Location" in line):
                    dummy = line.split()
                    
                    self.release_lon = float(dummy[7].strip(","))*mu.deg
                    self.release_lat = float(dummy[8].strip(","))*mu.deg
                    self.release_elv = float(dummy[9].strip(","))*mu.meter
                
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
        for k in keys:
            self.sounding[k] = numpy.array(self.sounding[k], dtype="float")*self.sounding_units[k]

        #Ensure that heights are AMSL and not AGL
        if (self.sounding["alt"][0] < self.release_elv):
            self.sounding["alt"] += self.release_elv

        #Returning
        return
    
    ### Method to pull Univ. of Wyoming Sounding from internet with Siphon
    ### Input, date for which to pull sounding
    def read_web(self, date):
    
        #Pull down the sounding
        sounding = WyomingUpperAir.request_data(date, self.fpath)
        
        #Convert sounding to proper data format and attach to PySonde object
        self.release_time = datetime.utcfromtimestamp(sounding["time"].values[0].tolist()/1e9)
        self.release_site = sounding["station"].values[0]
        self.release_lat = sounding["latitude"].values[0]*mu(sounding.units["latitude"]).to(mu.deg)
        self.release_lon = sounding["longitude"].values[0]*mu(sounding.units["longitude"]).to(mu.deg)
        self.release_elv = sounding["elevation"].values[0]*mu(sounding.units["elevation"]).to(mu.meter)
        
        skeys = ["pres", "temp", "dewp", "uwind", "vwind", "lon", "lat", "alt"]
        wkeys = ["pressure", "temperature", "dewpoint", "u_wind", "v_wind",
            "longitude", "latitude", "height"]
        for sk, wk in zip(skeys, wkeys):
            self.sounding[sk] = sounding[wk].values*mu(sounding.units[wk]).to(self.sounding_units[sk])
        
        #Fill in time array with Nans
        self.sounding["time"] = numpy.ones(self.sounding["pres"].shape)*numpy.nan
        
        #Ensure that heights are AMSL and not AGL
        if (self.sounding["alt"][0] < self.release_elv):
            self.sounding["alt"] += self.release_elv
        
        #Returning
        return
    
    ### Method to read WRF SCM input sounding
    def read_wrfscm(self):
    
        #Open the sounding and read line by line
        fn = open(self.fpath)
        first = True #Flag for first line
        theta = []   #List to hold potential temperature (K)
        qvapor = []  #List to hold mixing ratio (kg/kg)
        for line in fn:
            
            #Split the line into columns
            dummy = line.split()
            
            #Handle the first line
            if first:
                first = False
                self.release_elv = float(dummy[0])
                theta_sfc = float(dummy[3])
                qvapor_sfc = float(dummy[4])
                pres_sfc = float(dummy[5])
                continue
                
            #Read in the data
            self.sounding["alt"].append(dummy[0])
            self.sounding["uwind"].append(dummy[1])
            self.sounding["vwind"].append(dummy[2])
            theta.append(dummy[3])
            qvapor.append(dummy[4])
                
        #Close the file
        fn.close()
        
        #Convert data to arrays
        self.sounding["alt"] = numpy.array(self.sounding["alt"], dtype="float")
        self.sounding["uwind"] = numpy.array(self.sounding["uwind"], dtype="float")*self.sounding_units["uwind"]
        self.sounding["vwind"] = numpy.array(self.sounding["vwind"], dtype="float")*self.sounding_units["vwind"]
        theta = numpy.array(theta, dtype="float")
        qvapor = numpy.array(qvapor, dtype="float")
        
        #Calculate surface density
        tv = at.virt_temp(theta_sfc*(pres_sfc/100000.0)**(at.RD/at.CP), qvapor_sfc)
        rho_sfc = pres_sfc/(at.RD*tv)
        
        #Calculate pressure levels that correspond to sounding heights
        #Use the method present in module_initialize_scm_xy in WRF/dyn_em
        #Create arrays to hold values
        rho = numpy.zeros(theta.shape)
        pres = numpy.zeros(theta.shape)
        
        #Setup some supporting values
        rvord = at.RV/at.RD
        qvf = 1 + rvord*qvapor[0]
        qvf1 = 1 + qvapor[0]
        dz = [self.sounding["alt"][0]-self.release_elv]+list(self.sounding["alt"][1:]-self.sounding["alt"][:-1])
        
        #Iteratively solve for pressure and density
        rho[0] = rho_sfc
        for i in range(10):
            pres[0] = pres_sfc-0.5*dz[0]*(rho_sfc+rho[0])*at.G*qvf1
            tv = at.virt_temp(theta[0]*(pres[0]/100000.0)**(at.RD/at.CP), qvapor[0])
            rho[0] = pres[0]/(at.RD*tv)

        for i in range(1, theta.size):
            rho[i] = rho[i-1]
            qvf1 = 0.5*(2.0+(qvapor[i]+qvapor[i-1]))
            qvf = 1+rvord*qvapor[i]
            
            for j in range(10):
                pres[i] = pres[i-1]-0.5*dz[i]*(rho[i]+rho[i-1])*at.G*qvf1
                tv = at.virt_temp(theta[i]*(pres[i]/100000.0)**(at.RD/at.CP), qvapor[i])
                rho[i] = pres[i]/(at.RD*tv)
        
        #Now calculate temperature from the pressure levels
        temp = at.poisson(100000.0, pres, theta)
        
        #Fill rest of sounding
        skeys = ["time", "pres", "temp", "dewp", "uwind", "vwind", "lon", "lat", "alt"]
        self.sounding["time"] = numpy.ones(temp.shape)*numpy.nan*self.sounding_units["time"]
        self.sounding["alt"] = self.sounding["alt"]*self.sounding_units["alt"]
        self.sounding["pres"] = pres/100.0*self.sounding_units["pres"] #Pa -> hPa
        self.sounding["temp"] = (temp-273.15)*self.sounding_units["temp"] #K -> 'C
        self.sounding["dewp"] = (at.dewpoint(at.wtoe(pres, qvapor))-273.15)*self.sounding_units["dewp"]
        self.sounding["lon"] = numpy.nan*self.sounding_units["lon"]
        self.sounding["lat"] = numpy.nan*self.sounding_units["lat"]
        
        #Attach meta data
        self.release_time = datetime(2000, 1, 1)
        self.release_site = "WRF SCM"
        self.release_lat = self.sounding["lat"]
        self.release_lon = self.sounding["lon"]
        self.release_elv = self.release_elv*self.sounding_units["alt"]
        
        #Returning
        return
    
    ### Method to read in University of Wyoming sounding
    def read_wyoming(self):
        
        #Open the sounding and read line-by-line
        fn = open(self.fpath)
        wdir = [] #Lists to hold temporary wind variables
        wspd = []
        header = True #Flag for the header rows
        for line in fn:

            #Split the data row into columns
            dummy = line.split()
        
            #Testing if within data sections
            if ((len(dummy) == 11) and ("." in dummy[0])):
            
                #Read the data
                self.sounding["time"].append(numpy.nan)
                self.sounding["pres"].append(dummy[0])
                self.sounding["temp"].append(dummy[2])
                self.sounding["dewp"].append(dummy[3])
                wdir.append(dummy[6])
                wspd.append(dummy[7])
                self.sounding["lon"].append(numpy.nan)
                self.sounding["lat"].append(numpy.nan)
                self.sounding["alt"].append(dummy[1])
                
            else: #Go into metadata          
                if ("Station identifier" in line):
                    self.release_site = line.split(":")[1].strip()
                elif ("Station latitude" in line):
                    self.release_lat = float(line.split(":")[1].strip())*mu.deg
                elif ("Station longitude" in line):
                    self.release_lon = float(line.split(":")[1].strip())*mu.deg
                elif ("Station elevation" in line):
                    self.release_elv = float(line.split(":")[1].strip())*self.sounding_units["alt"]
                elif ("Observation time" in line):
                    self.release_time = datetime.strptime(line.split(":")[1].strip(), "%y%m%d/%H%M")
                
                continue
                        
        #Close the file
        fn.close()
                
        #Convert lists to arrays, attach units, and calculat wind components
        for k in self.sounding.keys():
            self.sounding[k] = numpy.array(self.sounding[k], dtype="float")*self.sounding_units[k]            
        
        #Calculate wind components
        wspd = numpy.array(wspd, dtype="float")
        wdir = numpy.array(wdir, dtype="float")
        self.sounding["uwind"] = (numpy.array(wspd*numpy.cos((270-wdir)*numpy.pi/180.0), dtype="float")*mu.knot).to(self.sounding_units["uwind"])
        self.sounding["vwind"] = (numpy.array(wspd*numpy.sin((270-wdir)*numpy.pi/180.0), dtype="float")*mu.knot).to(self.sounding_units["vwind"])
    
        #Ensure that heights are AMSL and not AGL
        if (self.sounding["alt"][0] < self.release_elv):
            self.sounding["alt"] += self.release_elv
    
        #Returning
        return

    #####---------------OTHER METHODS---------------#####
    
    ###Method to strip sounding of units
    ###Output
    ### unitless, dictionary containing sounding without units.
    ###  keys inlcude time, pres, temp, dewp, uwind, vwind, lon, lat, alt,
    ###  release_site, release_lat, release_lon, release_elv, release_time
    def strip_units(self):
    
        #Create dictionary for unitless sounding, starting with metadata
        unitless = {"release_site":self.release_site, "release_lat":numpy.array(self.release_lat),
            "release_lon":numpy.array(self.release_lon), "release_elv":numpy.array(self.release_elv),
            "release_time":self.release_time}
            
        #Now handle the other arrays
        for k in self.sounding.keys():
            unitless[k] = numpy.array(self.sounding[k])
    
        #Return unitless sounding
        return unitless













