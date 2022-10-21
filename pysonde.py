### This module contains the PySonde class
### A PySonde object provides utilities for interfacing with several
### common sounding formats. It also provideas utilities for common
### applications such as plotting and CAPE calculations.
### Further, once a sounding is loaded, it can be written out for
### other applications such as input to the WRF SCM or to the
### PySonde Sounding Format which is NetCDF.
### By default, the object supports unit aware computing, but the option exists to strip
### units from the sounding.
###
### The object only supports one sounding per file. If more are present, only the first
### will be read in, if the reader doesn't break.
###
### Written by Christopher Phillips, Therese Parkes
### University of Alabama in Huntsville, June 2020
###
### Currently supports:
###  NWS - NWS high density soundings
###  CSWR - Center for Severe Weather Research L2 soundings
###  EOL - National Center for Atmospheric Research - Earth Observing Laboratory soundings
###  HRRR - High Resolution Rapid Refresh analysis from AWS
###  UAH - University of Alabama in Huntsville UPSTORM group soundings
###  WRF - Weather Research and Forecasting Single Column Model Input Sounding
###  WYO - University of Wyoming sounding file
###  WYOWEB - University of Wyoming sounding online archive. (These are pulled from online.)
###  IGRA2 - The IGRA2 online sounding archive. (These are pulled from online.)
###  PSF - PySonde Sounding Format, a netCDF4 format more fully described in the documentation.
###  CSV - CSV files originally outputted by PySonde
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
import pysonde.atmos_math as am
import pysonde.atmos_thermo as at
import pysonde.hrrr_funcs as hf
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as pp
from metpy.units import units as mu
import metpy.calc as mc
from metpy.plots import SkewT
import netCDF4 as nc
import numpy
import os
import pygrib
from siphon.simplewebservice.wyoming import WyomingUpperAir
from siphon.simplewebservice.igra2 import IGRAUpperAir
import siphon.http_util as shu
import urllib.request as ur

############################################################
#---------------------     PYSONDE     ---------------------#
############################################################
class PySonde:

    ### Constructor Method
    ### Inputs:
    ###  fpath, string, file path to sounding. In the case of the web service, this is the station identifier
    ###  sformat, string, some options are: "NWS", "CSWR", "EOL", "IGRA2" see ReadMe for full breakdown
    ###  date, optional, required datetime object for web service sounding (WYOWEB, IGRA2, HRRR). (UTC)
    ###  point, optional, tuple of floats, (lon, lat) for use with model analysis soundings such as HRRR.
    ###
    ### Outputs:
    ###  None
    def __init__(self, fpath, sformat, date=None, point=None):

        #First attach file path, format, and units flag to object
        self.fpath = fpath
        self.sformat = sformat.upper()

        #Attach correct units to sounding object as a dictionary
        self.sounding_units = {"time":mu.second, "pres":mu.hPa, "temp":mu.degC, "dewp":mu.degC, "mixr":mu.g/mu.kg,
            "uwind":mu.meter/mu.second, "vwind":mu.meter/mu.second, "lon":mu.deg, "lat":mu.deg, "alt":mu.meter,
            "pot_temp":mu.degC}

        #Attach MetPy unit object so users have access
        self.units = mu

        #Initialize sounding dictionaries for both unitless and with units
        self.sounding = {} #Sounding with units attached
        self.sounding_uf = {} #Unit free sounding
        for k in self.sounding_units.keys():
            self.sounding[k] = []

        #Now read in the sounding based on it's format
        if (self.sformat == "NWS"): # NWS sounding
            self.read_nws()

        elif (self.sformat == "EOL"): # NCAR-EOL sounding
            self.read_eol()

        elif (self.sformat == "HRRR"): # HRRR Analysis
            self.read_hrrr(date, point, fpath=fpath)

        elif (self.sformat == "CSWR"): # CSWR sounding
            self.read_cswr()

        elif (self.sformat == "UAH"): # UAH UPSTORM format
            self.read_uah()

        elif (self.sformat == "WYO"): # University of Wyoming sounding
            self.read_wyoming()

        elif (self.sformat == "WEB"): # Pull sounding from internet
            self.read_web(date)

        elif (self.sformat == "WRF"): # WRF SCM input sounding
            self.read_wrfscm()

        elif (self.sformat == "PSF"):
            self.read_psf()

        elif (self.sformat == "CSV"):
            self.read_csv()
            
        elif (self.sformat == "IGRA2"): # IGRA2 online archive
            self.read_igra2(date)

        else: #Unrecognized format
            raise ValueError("Unrecognized sounding format: ()".format(self.sformat))

        # Add potential temperature to the sounding
        unitless = self.strip_units()
        self.sounding["pot_temp"] = (at.pot_temp(unitless["pres"]*100.0,
            unitless["temp"]+273.15)-273.15)*self.sounding_units["pot_temp"]

        # Add mixing ratio to the sounding
        if ((self.sformat != "PSF") and (self.sformat != "CSV")): #These already contains mixing ratio
            e = at.sat_vaporpres(unitless["dewp"]+273.15)
            p = unitless["pres"]
            self.sounding["mixr"] = at.etow(p*100.0,e)*1000.0*self.sounding_units["mixr"]

        # Calculate planetary boundary layer height
        self.calculate_pblh()

        # Calculate the basic thermo properties, (SFC CAPE/CIN, LCL, and Precipitable Water (PW)
        self.calculate_basic_thermo()

        #Returning
        return

    #####-----------METHODS TO CALCULATE SOUNDING PROPERTIES-----------#####

    #Method to calculate basic thermodynamic properties of the sounding.
    def calculate_basic_thermo(self):

        #Find first non-Nan level
        for ind, t in enumerate(self.sounding["dewp"]):
            if numpy.isfinite(t):
                break

        #Precipitable Water
        try: # backwards compatibility for older versions of metpy
            self.pw = mc.precipitable_water(self.sounding["pres"], self.sounding["dewp"])
        except:
            self.pw = mc.precipitable_water(self.sounding["dewp"], self.sounding["pres"])
            
        #Lifting condensation level
        self.lcl_pres, self.lcl_temp = mc.lcl(self.sounding["pres"][ind], self.sounding["temp"][ind],
            self.sounding["dewp"][ind])
        self.lcl_alt = self.sounding["alt"][numpy.nanargmin((self.lcl_pres-self.sounding["pres"])**2)]

        #Level of free convection
        try:
            inds = numpy.isfinite(self.sounding["dewp"])
            self.parcel_path = mc.parcel_profile(self.sounding["pres"][inds], self.sounding["temp"][inds][0],
                self.sounding["dewp"][inds][0])
            pos = (self.parcel_path > self.sounding["temp"][inds])
            lfc = (self.sounding["pres"][inds][pos])[0]
            i = 0
            while (lfc > self.lcl_pres):
                i += 1
                lfc = (self.sounding["pres"][inds][pos])[i]
            self.lfc_pres = lfc
            self.lfc_temp = (self.sounding["temp"][inds][pos])[i]
            self.lfc_alt = (self.sounding["alt"][inds][pos])[i]
        except Exception as err:
            print("WARNING: No LFC because:\n{}".format(err))
            self.lfc_pres = numpy.nan
            self.lfc_temp = numpy.nan
            self.lfc_alt = numpy.nan

        #Enclose in try, except because not every sounding will have a converging parcel path or CAPE.
        try:

            #Surface-based CAPE and CIN
            self.sfc_cape, self.sfc_cin = mc.cape_cin(self.sounding["pres"][inds], self.sounding["temp"][inds],
                self.sounding["dewp"][inds], self.parcel_path)

        #Do this when parcel path fails to converge
        except Exception as e:
            print("WARNING: No CAPE because:\n{}.".format(e))
            self.sfc_cape = numpy.nan
            self.sfc_cin = numpy.nan

        #Returning
        return

    # Method to calculate Cn2 along the sounding profile
    # Cn2 is commonly used to quantify the effects of atmospheric turbulence
    # on scintillation of light.
    #  Two methods are offered:
    #    "fiorino"
    #    Using the equations in Fiorino and Meier 2016
    #    "Improving the Fiedelity of the Tatarskii Cn2 Calculation with Inclusion
    #    of Pressure Perturbations"
    #  and
    #    "direct"
    #    Computing n directly from the sounding variables to calculate
    #    the structure constant
    # The direct method requires high resolution sounding data, dz <~ 25 m
    # and may not be applicable for conditions were the sounding is not representative
    # of the surrounding volume. I.e., it is assumed that the local mean environment
    # is equivalent to the measurement taken by the sounding.
    # Note that Cn2 is returned as log10(Cn2) so no units are associated natively.
    # Units are log10(m^(-2/3))
    # 
    # Outputs,
    #  logCn2, array of floats, log10 of Cn2
    def calculate_Cn2(self, method="fiorino"):

        #Assume a value of 150 m for the outer-scale length, L0.
        #This is from Fiorino but will be updated to real calculations.
        L0 = 150

        #Strip units from sounding
        sounding = self.strip_units()

        #Adjust units
        sounding["temp"] += 273.15
        sounding["pres"] *= 100.0
        sounding["dewp"] += 273.15
        sounding["mixr"] /= 1000.0
        sounding["pot_temp"] += 273.15

        # Choose method
        if (method.lower() == "fiorino"):

            #Compute virtual temperature and virtual potential temperature
            vtemp = at.virt_temp(sounding["temp"], sounding["mixr"])
            thetaV = sounding["pot_temp"]*(1.0+0.61*sounding["mixr"])

            #Compute vapor pressure (in hPa)
            ev = sounding["mixr"]*sounding["pres"]/(at.RD/at.RV+sounding["mixr"])/100.0

            #Compute perturabtion pressure (in hPa)
            #Perturbations are departures from hydrostaty
            pp = sounding["pres"]/100.0-self.calculate_pres(units=False)

            #Compute necessary atmospheric gradients
            dTdz = numpy.gradient(sounding["temp"], sounding["alt"])
            dTHvdz = numpy.gradient(thetaV, sounding["alt"])
            dTHdz = numpy.gradient(sounding["pot_temp"], sounding["alt"])
            dUdz = numpy.gradient(sounding["uwind"], sounding["alt"])
            dVdz = numpy.gradient(sounding["vwind"], sounding["alt"])
            dEvdz = numpy.gradient(ev, sounding["alt"])
            dPdz = numpy.gradient(pp, sounding["alt"])

            #Compute Richardson number
            Ri = at.G/vtemp*dTHvdz/(dUdz**2+dVdz**2)

            #Compute eddy diffusivity ratio (based on Fiorino and Meier 2016)
            khkm = numpy.where(Ri <= 1.0, 1.0/(6.873*Ri+(1.0/(1.0+6.873*Ri))), 1/(7.0*Ri))
            khkm = numpy.where(Ri < 0.01, 1.0, khkm)

            #Cn2 expanding dn/dz in terms of potental temperature and vapor pressure, and pressure pert.
            #Fiorino and Meier 2016 (Optics InfoBase Conference Papers)
            dndT = -1*(10**(-6)) * ((79.0*sounding["pres"]/100.0/sounding["temp"]**2) + 9600*ev/sounding["temp"]**3)
            dndev = 10**(-6)*4800.0/sounding["temp"]**2
            dndp = 10**(-6)*79.0/sounding["temp"]
            Cn2 = 2.8*khkm*(L0**(4.0/3.0))*(dndT*dTHdz+dndp*dPdz+dndev*dEvdz)**2

        elif (method.lower() == "direct"): # Follows methods of Muchinski et al. (1999; Radio Science)
        
            # Compute the water vapor pressure contribution
            q = sounding["mixr"]/(1+sounding["mixr"])
            pw = 1.608*q*sounding["pres"]
        
            # Compute index of refraction
            n = (7.76e-7)*sounding["pres"]/sounding["pot_temp"] + (3.73e-3)*pw/(sounding["pot_temp"]**2)
        
            # Compute Cn2
            Cn2 = ((n[1:]-n[:-1])**2)/((sounding["alt"][1:]-sounding["alt"][:-1])**(2.0/3.0))
            
            # Interpolate back to sounding levels
            Cn2_z = (sounding["alt"][1:]+sounding["alt"][:-1])/2.0
            Cn2 = numpy.interp(sounding["alt"], Cn2_z, Cn2)
        
        else:
        
            raise ValueError("{} is not valid value for 'method'. Use 'fiorino' or 'direct'.".format(method))

        #Replace infinities with NaNs
        Cn2[~numpy.isfinite(Cn2)] = numpy.nan

        #Compute logarithmic value
        logCn2 = numpy.log10(Cn2)

        #Remove unrealistically large values
        logCn2[logCn2 > 0] = numpy.nan

        #Return Cn2
        return logCn2

    #Method to calculate geopotential height from sounding
    #Integrates the hypsomteric equation.
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

        #Loop across the sounding profile while summing layer thickness
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

    #Method to calculate the hydrostatic pressure profile
    #The hypsometric equation is integrated
    #Inputs,
    # units=True, optional, whether to attach units to output
    #
    #Outputs,
    # hpres, the hydrostatic pressure in hPa
    def calculate_pres(self, units=True):

        #Strip units from sounding while integrating
        sounding = self.strip_units()

        #Convert units
        pres = sounding["pres"]*100.0  #hPa -> Pa
        temp = sounding["temp"]+273.15 #'C -> K
        mixr = sounding["mixr"]/1000.0 #g/kg -> kg/kg

        #Calculate virtual temperature
        vtemp = at.virt_temp(temp, mixr)

        #Identify the first non-Nan level in the sounding
        for i, ps in enumerate(pres):
            if numpy.isfinite(ps):
                ind0 = i
                break

        #If first element isn't finite, then replace with temperature
        if (not numpy.isfinite(vtemp[ind0])):
            vtemp[ind0] = temp[ind0]

        #Loop across the sounding profile while summing the pressure changes
        p = [] #Grab the surface pressure in Pa
        j = 1
        for i in range(pres.size):

            if (i < ind0):
                p.append(numpy.nan)
                continue
            elif (i == ind0):
                p.append(pres[ind0])
                continue

            #Compute mean layer virtual temperature and layer thickness
            tvbar = am.layer_interp(pres[i-j], pres[i], (pres[i-j]+pres[i])/2.0, vtemp[i-j], vtemp[i])
            dz = sounding["alt"][i]-sounding["alt"][i-j]
            p.append(p[i-j]*numpy.exp(-(at.G*dz)/(at.RD*tvbar)))

            #For handling layers with missing data
            if (not numpy.isfinite(dz)):
                j += 1
            else:
                j = 1

        #Convert to numpy array and handle units
        if units:
            hpres = numpy.array(p)/100.0*self.sounding_units["pres"]
        else:
            hpres = numpy.array(p)/100.0

        return hpres

    #Method to calculate planetary boundary layer height (PBLH) from the sounding
    #The algorithm finds the first location where the environmental potential temperature is greater
    #than the surface. If a surface inversion is present, than the top of the inversion is called the PBL top.
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
            height = (sounding["alt"]-sounding["release_elv"]) #Height AGL
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
                self.sbih_pres = (sounding["pres"])[ind_sbi]*self.sounding_units["pres"]
                self.sbih_ind = ind_sbi

                #Re-locate PBL top
                ind_pbl = numpy.where(tgrad[ind_sbi:] > 0.0)[0][0]+ind_sbi

            #If no inversion exists below 600 hPa, then use the mixing method
            if ((sounding["pres"])[ind_pbl] < 600.0):

                #Locate first location where surface parcel potential temp is less than environment
                ind_pbl = 0
                ind = 0
                while (ind_pbl < 2): #To avoid getting stuck at the surface
                    ind_pbl = numpy.where(theta[ind] < theta)[0][0]
                    ind += 1

            #Retreive PBL top height and pressure
            pblh = height[ind_pbl]*self.sounding_units["alt"]
            pblh_pres = (sounding["pres"])[ind_pbl]*self.sounding_units["pres"]

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
    # level, float, pressure level (hPa if no unit attached) for which to pull values
    #Outputs
    # data, dictionary keyed with sounding variables, contains values at
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
        ind = numpy.nanargmin(abs(self.sounding["pres"]-level))
        if (self.sounding["pres"][ind] >= level):
            tind = ind+1
            bind = ind
        elif (self.sounding["pres"][ind] < level):
            tind = ind
            bind = ind-1

        #Perform the interpolations
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

        #Open file and write header
        fn = open(spath, "w")
        header = "Time (s)\tGPH (m)\tPrs (hPa)\tPot. Temp ('C)\tTemp ('C)\tMixR (g/kg)\tDewp ('C)\tUspd (m/s)\tVspd (m/s)\tLon\tLat".split("\t")
        fn.write(("{}"+",{}"*(len(header)-1)).format(*header))

        #Now loop through the data
        for i in range(unitless["pres"].size):
            data = (unitless["time"][i], unitless["alt"][i], unitless["pres"][i], unitless["pot_temp"][i], unitless["temp"][i],
                unitless["mixr"][i], unitless["dewp"][i], unitless["uwind"][i], unitless["vwind"][i],
                unitless["lon"][i], unitless["lat"][i])
            fn.write(("\n{:.3f}"+",{:.3f}"*(len(header)-1)).format(*data))

        #Close the file
        fn.close()

        #Returning
        return

    ### Method to output interpolated sounding to a text file
    ### Inputs:
    ###  levs, array of floats, levels to interpolate too in meters or hPa
    ###  spath, string, full path on which to save file
    ###  coords = "Z", string, options are "Z" or "P" for
    ###    gph or pressure interolation respectively.
    ###  write = True, whether to write to a file. If False, instead returns
    ###    interpolated sounding dictionary
    def write_interpolated_csv(self, levs, spath, coords="Z", write=True):

        #First strip all units from sounding
        unitless = self.strip_units()

        #Create dictionary to hold the intepolated values
        isounding = {}

        #Interpolate everything in the unitless sounding
        if (coords.upper() == "Z"):
            #For the Z coordinates, the pressure levels of the desired z levels are
            #found first. Then the sounding is interpolated those pressure levels.
            #This produces a sounding on the desired height grid while preserving the
            #ressure-weighted nature of the interpolation.

            #Calculate teh virtual temperature profile for hypsomteric equation
            vtemps = at.virt_temp(unitless["temp"]+273.15, unitless["mixr"]/1000.0)

            #Calculate the desired pressure levels using the hypsometric equation
            plevs = []
            for lev in levs:

                #Find the sounding levels that bracket this one
                ind = numpy.argmin((unitless["alt"]-lev)**2)
                if (unitless["alt"][ind] > lev):
                    tind = ind
                    bind = ind-1
                else:
                    tind = ind+1
                    bind = ind

                #If this level is below/above the first/last in the sounding, then fill with
                #an artificial pressure to force a Nan during interpolation
                if (tind == 0):
                    plevs.append(2000)
                    continue
                elif (bind == unitless["alt"].size): #Too high
                    plevs.append(-1)
                    continue

                #Calculate mean layer virtual temperature
                vtemp = am.layer_average(unitless["pres"][bind:tind+1], vtemps[bind:tind+1])

                #Compute the new pressure level
                plevs.append(unitless["pres"][bind]*numpy.exp(-at.G*(lev-unitless["alt"][bind])/(at.RD*vtemp)))

            #Convert pressure list to an array
            plevs = numpy.array(plevs)

            #Interpolate
            for k in unitless.keys():
                try:
                    if (k != "alt"):
                        isounding[k] = numpy.interp(numpy.log(plevs), numpy.log(unitless["pres"][::-1]), unitless[k][::-1],
                            left=numpy.nan, right=numpy.nan)
                    else:
                        isounding[k] = levs

                except:
                    pass

        elif (coords.upper() == "P"):

            #Interpolate
            for k in unitless.keys():
                try:
                    if (k != "pres"):
                        isounding[k] = numpy.interp(numpy.log(levs), numpy.log(unitless["pres"][::-1]), unitless[k][::-1],
                            left=numpy.nan, right=numpy.nan)
                    else:
                        isounding[k] = levs

                except:
                    pass
        else:
            raise ValueError("Do not recognize {}. options are 'Z' or 'P'".format(coords))

        if write:
            #Open file and write header
            fn = open(spath, "w")
            header = "Time (s)\tGPH (m)\tPrs (hPa)\tPot. Temp ('C)\tTemp ('C)\tMixR (g/kg)\tDewp ('C)\tUspd (m/s)\tVspd (m/s)\tLon\tLat".split("\t")
            fn.write(("{}"+",{}"*(len(header)-1)).format(*header))

            #Now loop through the data
            for i in range(isounding["pres"].size):
                data = (isounding["time"][i], isounding["alt"][i], isounding["pres"][i], isounding["pot_temp"][i], isounding["temp"][i],
                    isounding["mixr"][i], isounding["dewp"][i], isounding["uwind"][i], isounding["vwind"][i],
                    isounding["lon"][i], isounding["lat"][i])
                fn.write(("\n{:.3f}"+",{:.3f}"*(len(header)-1)).format(*data))

            #Close the file
            fn.close()

        else:

            #Return interpolated sounding
            return isounding

        #Returning
        return

    ### Method to output RAMS input sounding
    ### This sounding is suitable for reading into the RAMS model
    ### with the following namelist options:
    ###  IPSFLG = 0
    ###  ITSFLG = 0
    ###  IRTSFLG = 0
    ###  IUSFLG = 0
    ###  
    ### Inputs:
    ###  spath, string, location to save file to
    ###  maxz, optional, float, maximum height of RAMS sounding in meters
    ###    if None, then the maximum height of the sounding is used
    ###    defaults to None
    def write_rams(self, spath, maxz=None):
    
        #Strip units from sounding
        sounding = self.strip_units()
    
        #Check if user supplied maximum height
        if (maxz == None):
            zind = sounding["alt"].size-1
        else:
            zind = numpy.arange(sounding["alt"].size)[sounding["alt"]>maxz][0]
            
        #Check length of sounding to see if it fits within the RAMS 200pt limit
        #Interpolate if necessary (assume linear in log-p coordinates)
        if (sounding["alt"][:zind].size > 200):
            
            #Get first and last pressure levels for interpolation
            p0 = sounding["pres"][0]
            pf = sounding["pres"][zind]
            
            #Create evenly spaced pressure levels
            ip = numpy.linspace(p0, pf, 200, endpoint=True)
        
            #Interpolate       
            for k in ["dewp", "temp", "uwind", "vwind", "alt"]:
                sounding[k] = numpy.interp(numpy.log(ip[::-1]), numpy.log(sounding["pres"][::-1]), sounding[k][::-1])[::-1]
            sounding["pres"] = ip
            
        #Replace any missing values with 9999
        #9999 is based on the missing wind value in subroutine ARRSND of file rhhi.f90
        #in the RAMS initialization code.
        for k in ["dewp", "temp", "uwind", "vwind", "alt"]:
            sounding[k][~numpy.isfinite(sounding[k])] = 9999
            
        #Write out sounding
        fn = open(spath, "w")
        fn.write("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(
            sounding["pres"][0], sounding["temp"][0], sounding["dewp"][0],
            sounding["uwind"][0], sounding["vwind"][0]))
        for i in range(1, sounding["pres"].size):
            fn.write("\n{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(
            sounding["pres"][i], sounding["temp"][i], sounding["dewp"][i],
            sounding["uwind"][i], sounding["vwind"][i]))
        fn.close()
            

    ### Method to output WRF SCM input sounding
    ### Inputs:
    ###  spath, string, location to save file to
    def write_wrfscm(self, spath):

        #First strip all units from sounding
        unitless = self.strip_units()

        # Remove any lines with missing data
        # Use dewpoint because if anything is missing, dewpoint will be.
        mask = numpy.isfinite(unitless["dewp"])
        for k in self.sounding_units.keys():
            unitless[k] = unitless[k][mask]

        #Do necessary unit conversions
        heights = unitless["alt"]
        temp = unitless["temp"]+273.15 #C -> K
        pres = unitless["pres"]*100.0 #hPa -> Pa

        #Calculate necessary surface variables
        #First compute 10m winds from the sounding using linear interpolation
        ind2 = (numpy.arange(heights.size, dtype="int")[(heights-10)>heights[0]])[0]
        ind1 = ind2-1
        
        # Interpolate winds to 10 m
        weight = (10.0-(heights[ind1]-heights[0]))/(heights[ind2]-heights[ind1])
        u10 = (unitless["uwind"][ind1]*(1-weight))+(unitless["uwind"][ind2]*weight)
        v10 = (unitless["vwind"][ind1]*(1-weight))+(unitless["vwind"][ind2]*weight)

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
        fn.write("{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
            unitless["release_elv"], u10, v10, theta[0], qvapor[0], pres[0]))
            
        for i, h in enumerate(new_height[1:]):
            fn.write("\n{:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                h, new_uwind[i+1], new_vwind[i+1], new_theta[i+1], new_qvapor[i+1]))
                
        fn.close()

        #Returning
        return

    #####---------------METHODS TO READ SOUNDING FORMATS---------------#####

    ### Method to read a PySonde CSV file
    def read_csv(self):

        #Create a dictionary to hold the sounding
        keys = ["time", "pres", "temp", "dewp", "mixr", "uwind", "vwind", "lon", "lat", "alt"]
        self.sounding = {}
        for k in keys:
            self.sounding[k] = []

        #Open the file and read it in
        fn = open(self.fpath)
        header = True
        for line in fn:
            #Skip the first line as it's the header
            if header:
                header = False
                continue

            #Split the line and read data
            dummy = line.split(",")
            self.sounding["time"].append(dummy[0])
            self.sounding["pres"].append(dummy[2])
            self.sounding["temp"].append(dummy[4])
            self.sounding["dewp"].append(dummy[6])
            self.sounding["mixr"].append(dummy[5])
            self.sounding["uwind"].append(dummy[7])
            self.sounding["vwind"].append(dummy[8])
            self.sounding["lon"].append(dummy[9])
            self.sounding["lat"].append(dummy[10])
            self.sounding["alt"].append(dummy[1])

        #Close the file
        fn.close()

        #Now convert variables to arrays and attach units
        for k in keys:
            self.sounding[k] = numpy.array(self.sounding[k], dtype="float")
            self.sounding[k] *= self.sounding_units[k]

        #Attach meta data
        self.release_time = datetime(2000, 1, 1)
        self.release_site = "Unknown"
        try: #In case of missing values in the lowest levels
            self.release_elv = self.sounding["alt"][numpy.isfinite(self.sounding["lat"]/self.sounding_units["alt"])][0]
            self.release_lat = self.sounding["lat"][numpy.isfinite(self.sounding["lat"]/self.sounding_units["lat"])][0]
            self.release_lon = self.sounding["lon"][numpy.isfinite(self.sounding["lat"]/self.sounding_units["lon"])][0]
        except: #In case values are missing entirely
            self.release_elv = self.sounding["alt"][0]
            self.release_lat = self.sounding["lat"][0]
            self.release_lon = self.sounding["lon"][0]

        #Returning
        return

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

    ### Method to read HRRR soundings (makes use of the AWS archive)
    ### Inputs:
    ###   date, datetime object for which to pull the sounding
    ###   loc, tuple of floats, (lon, lat). The location for which to pull the sounding.
    ###   fpath, optional, string, file path to a HRRR file.
    def read_hrrr(self, date, loc, fpath="hrrr.grib2"):
            
        # Try to open HRRR file, if that fails, download a new one
        try:
            grib = pygrib.open(fpath)
        except:
        
            # The internet location of the HRRR data
            url_base = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/"
            hrrr_url = url_base + "hrrr.{}/conus/hrrr.t{:02d}z.wrfprsf00.grib2".format(
            date.strftime("%Y%m%d"), date.hour)
        
            # Download the HRRR file
            ur.urlretrieve(hrrr_url, "hrrr.grib2")
            grib = pygrib.open("hrrr.grib2")
        
        # Extract the sounding
        self.sounding, self.release_elv = hf.get_sounding(loc, grib)
        
        # Adjust units
        self.sounding["temp"] -= 273.15
        self.sounding["dewp"] -= 273.15
        
        # Add header data
        self.release_time = date
        self.release_site = "{:.2f} Lon {:.2f} Lat".format(loc[0], loc[1])
        self.release_lon = loc[0]*mu.deg
        self.release_lat = loc[1]*mu.deg
        self.release_elv = self.release_elv*mu.meter
        
        # Add missing fields to the sounding
        self.sounding["time"] = numpy.zeros(self.sounding["temp"].shape)
        self.sounding["lon"] = numpy.ones(self.sounding["temp"].shape)*loc[0]
        self.sounding["lat"] = numpy.ones(self.sounding["temp"].shape)*loc[1]
        
        # Add units to the variables
        for k in ["time", "pres", "temp", "dewp", "uwind", "vwind", "lon", "lat", "alt"]:
            self.sounding[k] = numpy.array(self.sounding[k], dtype="float")*self.sounding_units[k]
        
        # Close the grib file
        grib.close()
        
        # Delete the downloaded file
        try:
            os.system("rm hrrr.grib2")
        except:
            print("Warning: Unable to remove hrrr.grib2 temporary file.")
        
        #Ensure that heights are AMSL and not AGL
        if (self.sounding["alt"][0] < self.release_elv):
            self.sounding["alt"] += self.release_elv
        
        # Exit the reader
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

            #First check if in header
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

    ### Method to read the PySonde Sounding Format
    ### Note that the AMSL vs. AGL check is not performed due to model
    ### wonkiness. The user is responsible for ensuring that heights are AMSL.
    def read_psf(self):

        #Load the NetCDF file for reading
        fn = nc.Dataset(self.fpath)

        #Grab the missing value
        try:
            missing_value = fn.missing_value
        except:
            missing_value = numpy.nan

        #Grab the launch location and time
        self.release_site = fn.station
        self.release_time = datetime.strptime(fn.release_time, fn.datetime_format)
        self.release_lon = fn.release_lon*mu.deg
        self.release_lat = fn.release_lat*mu.deg
        self.release_elv = fn.release_elv*mu(fn.variables["alt"].units).to(mu.meter)

        #Create a dictionary to hold the sounding
        self.sounding = {}

        #Read in the sounding data and replace missing data with NaNs
        skeys = ["pres", "temp", "dewp", "uwind", "vwind", "alt", "mixr"]
        for sk in skeys:
            self.sounding[sk] = (numpy.array(fn.variables[sk][:])*mu(fn.variables[sk].units)).to(self.sounding_units[sk])
            self.sounding[sk][self.sounding[sk] == missing_value*self.sounding_units[sk]] = numpy.nan*self.sounding_units[sk]

        #Add other variables that may be missing from the file (such as in model runs)
        try:
            self.sounding["time"] = (numpy.array(fn.variables["time"][:])*mu(fn.variables["time"].units)).to(self.sounding_units["time"])
        except:
            self.sounding["time"] = numpy.ones(self.sounding["temp"].shape)*numpy.nan*self.sounding_units["time"]
        try:
            self.sounding["lon"] = (numpy.array(fn.variables["lon"][:])*mu(fn.variables["lon"].units)).to(self.sounding_units["lon"])
        except:
            self.sounding["lon"] = numpy.ones(self.sounding["temp"].shape)*numpy.nan*self.sounding_units["lon"]
        try:
            self.sounding["lat"] = (numpy.array(fn.variables["lat"][:])*mu(fn.variables["lat"].units)).to(self.sounding_units["lat"])
        except:
            self.sounding["lat"] = numpy.ones(self.sounding["temp"].shape)*numpy.nan*self.sounding_units["lat"]

        #Ensure that no values are below AGL (happens occasionally in models
        mask = (self.sounding["alt"] >= self.release_elv)
        if (numpy.sum(mask) < len(mask)):
            for sk in self.sounding.keys():
                self.sounding[sk] = self.sounding[sk][mask]

        #Close the netcdf file
        fn.close()

        #Returning
        return
    
    ### Method read in soundings that are in the UAH UPSTORM group format
    def read_uah(self):
    
        # Create a dictionary to hold the sounding
        keys = ["time", "pres", "temp", "dewp", "uwind", "vwind", "lon", "lat", "alt"]
        self.sounding = {}
        for k in keys:
            self.sounding[k] = []

        # Read the file
        fn = open(self.fpath)
        lines = [line for line in fn]
        
        # Pull the header information
        header = lines[1].strip("#").split(",")
        loc = lines[3].split(",")
        self.release_site = header[2]+", "+header[3]
        self.release_time = datetime.strptime(header[0].strip()+header[1].strip(), "%Y%m%d%H%M UTC")
        self.release_lon = float(loc[0])*mu.deg
        self.release_lat = float(loc[1])*mu.deg
        self.release_elv = float(header[4].strip()[:-1])*mu.meter
        
        # Read in the data
        wspd = []
        wdir = []
        for line in lines[3:]:

            # Check for end of file
            if ("END" in line):
                break

            # Split line into columns
            dummy = line.split(",")
            print(dummy)
            
            # Pull data
            self.sounding["time"].append((datetime.strptime("{}{}".format(header[0].strip(), dummy[2].strip()), "%Y%m%d%H:%M:%S")-
                self.release_time).total_seconds())
            self.sounding["lon"].append(dummy[0])
            self.sounding["lat"].append(dummy[1])
            self.sounding["alt"].append(dummy[3])
            self.sounding["pres"].append(dummy[4])
            self.sounding["temp"].append(dummy[5])
            self.sounding["dewp"].append(dummy[7])
            wspd.append(dummy[8])
            wdir.append(dummy[9].strip())

        # Once the data has been read in, convert everything to numpy arrays and attach units
        for k in keys:
            self.sounding[k] = numpy.array(self.sounding[k], dtype="float")*self.sounding_units[k]

        # Ensure that heights are AMSL and not AGL
        if (self.sounding["alt"][0] < self.release_elv):
            self.sounding["alt"] += self.release_elv

        # Compute wind speed components in m/s
        wspd = numpy.array(wspd, dtype="float")*0.514
        wdir = numpy.array(wdir, dtype="float")

        self.sounding["uwind"] = wspd*numpy.cos((270-wdir)*numpy.pi/180.0)*self.sounding_units["uwind"]
        self.sounding["vwind"] = wspd*numpy.sin((270-wdir)*numpy.pi/180.0)*self.sounding_units["vwind"]

        # Close the sounding object
        fn.close()

        # Returning
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
        self.sounding["lon"] = numpy.ones(temp.shape)*numpy.nan*self.sounding_units["lon"]
        self.sounding["lat"] = numpy.ones(temp.shape)*numpy.nan*self.sounding_units["lat"]

        #Attach meta data
        self.release_time = datetime(2000, 1, 1)
        self.release_site = "WRF SCM"
        self.release_lat = self.sounding["lat"][0]
        self.release_lon = self.sounding["lon"][0]
        self.release_elv = self.sounding["alt"][0]*self.sounding_units["alt"]

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

    ### Method to pull IGRA2 Sounding from internet with Siphon
    ### Input, date, datetime object, date for which to pull sounding
    ### Written by Therese Parkes, UAH
    def read_igra2(self, date):

        #Pull down the sounding
        sounding, header = IGRAUpperAir.request_data(date, self.fpath)
        
        # Pull pressure from the sounding dataset
        # Need this to mask any extraneous rows
        pres = sounding['pressure']
        mask = numpy.isfinite(pres)
        
        #Convert sounding to proper data format and attach to PySonde object
        self.release_time = date
        self.release_site = header["site_id"].values[0]
        self.release_lat = header["latitude"].values[0]*mu(header.units["latitude"]).to(mu.deg)
        self.release_lon = header["longitude"].values[0]*mu(header.units["longitude"]).to(mu.deg)
        self.release_elv = sounding["height"].values[0]*mu(sounding.units["height"]).to(mu.meter)

        s1keys = ["pres", "temp", "dewp", "uwind", "vwind", "alt"]
        s2keys = ["lon", "lat"]
        sounding_keys = ["pressure", "temperature", "dewpoint", "u_wind", "v_wind", "height"]
        header_keys = ["longitude", "latitude"]
        for sk, soundk in zip(s1keys, sounding_keys):
            self.sounding[sk] = sounding[soundk].values[mask]*mu(sounding.units[soundk]).to(self.sounding_units[sk])

        for sk, headk in zip(s2keys, header_keys):
          self.sounding[sk] = header[headk].values*mu(header.units[headk]).to(self.sounding_units[sk])

        #Fill in time array with Nans
        self.sounding["time"] = numpy.ones(self.sounding["pres"].shape)*numpy.nan

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
        try: #Fails during the initial PBLH calculation
            unitless = {"release_site":self.release_site, "release_lat":numpy.array(self.release_lat),
                "release_lon":numpy.array(self.release_lon), "release_elv":numpy.array(self.release_elv),
                "release_time":self.release_time, "pblh":numpy.array(self.pblh), "pblh_pres":numpy.array(self.pblh_pres),
                "lcl_pres":numpy.array(self.lcl_pres), "lcl_temp":numpy.array(self.lcl_temp), "lcl_alt":numpy.array(self.lcl_alt),
                "lfc_pres":numpy.array(self.lfc_pres), "lfc_temp":numpy.array(self.lfc_temp), "lfc_alt":numpy.array(self.lfc_alt)}
        except:
            unitless = {"release_site":self.release_site, "release_lat":numpy.array(self.release_lat),
                "release_lon":numpy.array(self.release_lon), "release_elv":numpy.array(self.release_elv),
                "release_time":self.release_time}

        #Now handle the other arrays
        for k in self.sounding.keys():
            unitless[k] = numpy.array(self.sounding[k])

        #Return unitless sounding
        return unitless
