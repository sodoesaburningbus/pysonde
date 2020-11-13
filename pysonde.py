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
### MetPy
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
    ###  date, optional, datetime object for Siphon web service sounding
    ###
    ### Outputs:
    ###  None
    def __init__(self, fpath, sformat, date=None):

        #First attach file path, format, and units flag to object
        self.fpath = fpath
        self.sformat = sformat.upper()

        #Attach correct units to sounding object as a dictionary
        self.sounding_units = {"time":mu.second, "pres":mu.hPa, "temp":mu.degC, "dewp":mu.degC,
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
            print("Unrecognized sounding format: ()".format(self.sformat))
            raise ValueError

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
            self.pw = mc.precipitable_water(self.sounding["dewp"], self.sounding["pres"])

            #Lifting condensation level
            self.lcl_pres, self.lcl_temp = mc.lcl(self.sounding["pres"][0], self.sounding["temp"][0],
                self.sounding["dewp"][0])

            #Surface-based CAPE and CIN
            self.parcel_path = mc.parcel_profile(self.sounding["pres"], self.sounding["temp"][0],
                self.sounding["dewp"][0])
            self.sfc_cape, self.sfc_cin = mc.cape_cin(self.sounding["pres"], self.sounding["temp"],
                self.sounding["dewp"], self.parcel_path)

        #Do this when parcel path fails to converge
        except:
            print("WARNING: Parcel path did not converge. No LCL, CAPE, or PW stats.")
            self.parcel_path = numpy.nan
            self.pw = numpy.nan
            self.lcl_pres = numpy.nan
            self.lcl_temp = numpy.nan
            self.sfc_cape = numpy.nan
            self.sfc_cin = numpy.nan
                
        #Returning
        return

    #Method to calculate planetary boundary layer height (PBLH) from the sounding
    #The algorithm takes the surface parcel and finds the first height where that parcel
    #has negative buoyancy. If that location is the surface, we move up one level and try again.
    #If the PBLH cannot be calculated, it is set to -1
    def calculate_pblh(self):
    
        try:
            #Strip units from sounding for use with atmos package
            sounding = self.strip_units()
            
            #Calculate sounding potential temeprature and height AGL
            height = (sounding["alt"]-self.release_elv/self.sounding_units["alt"])
            theta = at.pot_temp(sounding["pres"]*100.0, sounding["temp"]+273.15)
                    
            #Locate first location where surface parcel temp is less than environment
            #But still above the surface
            ind_pbl = 0
            ind = 0
            while (ind_pbl < 2):
                ind_pbl = numpy.where(theta[ind] < theta)[0][0]
                ind += 1
            
            pblh = height[ind_pbl]*self.sounding_units["alt"]
            pblh_pres = sounding["pres"][ind_pbl]*self.sounding_units["pres"]
            
            ### Old method using refractive index gradient
            ### This tended to get hung up on remnant layers
            #Calculate refracitive index
            #refr = (77.6*(sounding["pres"]*100.0)/(sounding["temp"]+273.15)-
            #                5.6*(at.sat_vaporpres(sounding["dewp"]+273.15))/(sounding["temp"]+273.15)+
            #                3.75e5*at.sat_vaporpres(sounding["dewp"]+273.15)/(sounding["temp"]+273.15)**2)
            #
            #Calculate the gradient of refractivity
            #grad_refr = numpy.gradient(refr, sounding["alt"])
            
            #Locate the height of maximum grad_theta and call that the PBLH
            #(and re-attach proper units)
            #height = (sounding["alt"]-self.release_elv/self.sounding_units["alt"])
            #grad_refr = grad_refr[(height<5000)&(height>50)]
            #pres = (sounding["pres"])[(height<5000)&(height>50)]
            #height = height[(height<5000)&(height>50)]
            #pblh = height[numpy.argmin(grad_refr)]*self.sounding_units["alt"]
            #pblh_pres = pres[numpy.argmin(grad_refr)]*self.sounding_units["pres"]"""
            
            #Store PBLH as attribute of sounding
            self.pblh = pblh
            self.pblh_pres = pblh_pres
           
        except:
            self.pblh = -1
            self.pblh_pres = -1
    
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

    #####-------------------METHODS TO PLOT OR OUTPUT SOUNDING-------------------#####

    ### Method to plot a basic sounding
    ### Includes only temperature, dewpoint, and the parcel path
    ### Inputs:
    ###  nbarbs, optional, integer, spacing between wind barbs
    ### Outputs:
    ###  fig, the pyplot figure object
    ###  skewt, the MetPy SkewT axis object
    def basic_skewt(self, nbarbs=None):

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
        skewt.plot_mixing_lines(p=self.sounding["pres"])

        #Adjust the axis labels
        skewt.ax.set_xlabel("Temperature ('C)", fontsize=14, fontweight="bold")
        skewt.ax.set_ylabel("Pressure (hPa)", fontsize=14, fontweight="bold")

        #Returning
        return fig, skewt

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
        
        skeys = ["time", "pres", "temp", "dewp", "uwind", "vwind", "lon", "lat", "alt"]
        wkeys = ["time", "pressure", "temperature", "dewpoint", "u_wind", "v_wind",
            "longitude", "latitude", "height"]
        for sk, wk in zip(skeys, wkeys):
            if (sk == "time"):
                self.sounding[sk] = numpy.nan
            else:
                self.sounding[sk] = sounding[wk].values*mu(sounding.units[wk]).to(self.sounding_units[sk])
        
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













