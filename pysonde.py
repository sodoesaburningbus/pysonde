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
###  WEB - University of Wyoming sounding online archive. (These are pulled from online.)
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
### np

### Importing required modules
import pysonde.atmos_math as am
import pysonde.atmos_thermo as at
import pysonde.hrrr_funcs as hf
import pysonde.severewx as swx
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as pp
from metpy.units import units as mu
import metpy.calc as mc
from metpy.plots import SkewT, Hodograph
import netCDF4 as nc
import numpy as np
import os
import pygrib
from siphon.simplewebservice.wyoming import WyomingUpperAir
from siphon.simplewebservice.igra2 import IGRAUpperAir
import siphon.http_util as shu
import urllib.request as ur
import matplotlib.gridspec as gridspec

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
            if np.isfinite(t):
                break

        #Precipitable Water
        try: # backwards compatibility for older versions of metpy
            self.pw = mc.precipitable_water(self.sounding["pres"], self.sounding["dewp"])
        except:
            self.pw = mc.precipitable_water(self.sounding["dewp"], self.sounding["pres"])
            
        #Lifting condensation level
        self.lcl_pres, self.lcl_temp = mc.lcl(self.sounding["pres"][ind], self.sounding["temp"][ind],
            self.sounding["dewp"][ind])
        self.lcl_alt = self.sounding["alt"][np.nanargmin((self.lcl_pres-self.sounding["pres"])**2)]

        #Level of free convection
        try:
            inds = np.isfinite(self.sounding["dewp"])
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
            print("Likely due to atmospheric stability.")
            self.lfc_pres = np.nan*self.sounding_units['pres']
            self.lfc_temp = np.nan*self.sounding_units['temp']
            self.lfc_alt = np.nan*self.sounding_units['alt']
        
        #Equilibrium Level:
        try:
            self.el_pres, self.el_temp = mc.el(self.sounding["pres"], self.sounding["temp"], self.sounding["dewp"])
            self.el_alt = self.sounding["alt"][np.nanargmin((self.el_pres-self.sounding["pres"])**2)]
        except:
            self.el_pres = np.nan*self.sounding_units['pres']
            self.el_temp = np.nan*self.sounding_units['temp']
            self.el_alt = np.nan*self.sounding_units['alt']

        #Enclose in try, except because not every sounding will have a converging parcel path or CAPE.
        try:

            # Surface-based CAPE and CIN
            self.sfc_cape, self.sfc_cin = mc.cape_cin(self.sounding["pres"][inds], self.sounding["temp"][inds],
                self.sounding["dewp"][inds], self.parcel_path)
            
            # Most unstable CAPE and CIN
            self.mu_cape, self.mu_cin = mc.most_unstable_cape_cin(self.sounding['pres'], self.sounding['temp'],
                self.sounding['dewp'])

            # Mixed layer CAPE and CIN
            self.ml_cape, self.ml_cin = mc.mixed_layer_cape_cin(self.sounding['pres'], self.sounding['temp'], self.sounding['dewp'])

            #3CAPE and 3CIN
            hgt = self.sounding['alt'] - self.release_elv
            km3_ind = np.argmin((hgt.to('km') - 3*mu.km)**2)
            self.cape3, self.cin3 = mc.mixed_layer_cape_cin(self.sounding['pres'][:km3_ind], self.sounding['temp'][:km3_ind], self.sounding['dewp'][:km3_ind])

        #Do this when parcel path fails to converge
        except Exception as e:
            print("WARNING: No CAPE because:\n{}.".format(e))
            self.sfc_cape = 0.0
            self.sfc_cin = 0.0
            self.mu_cape = 0.0
            self.mu_cin = 0.0
            self.ml_cape = 0.0
            self.ml_cin = 0.0
            self.CAPE3 = 0.0
            self.CIN3 = 0.0

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
            dTdz = np.gradient(sounding["temp"], sounding["alt"])
            dTHvdz = np.gradient(thetaV, sounding["alt"])
            dTHdz = np.gradient(sounding["pot_temp"], sounding["alt"])
            dUdz = np.gradient(sounding["uwind"], sounding["alt"])
            dVdz = np.gradient(sounding["vwind"], sounding["alt"])
            dEvdz = np.gradient(ev, sounding["alt"])
            dPdz = np.gradient(pp, sounding["alt"])

            #Compute Richardson number
            Ri = at.G/vtemp*dTHvdz/(dUdz**2+dVdz**2)

            #Compute eddy diffusivity ratio (based on Fiorino and Meier 2016)
            khkm = np.where(Ri <= 1.0, 1.0/(6.873*Ri+(1.0/(1.0+6.873*Ri))), 1/(7.0*Ri))
            khkm = np.where(Ri < 0.01, 1.0, khkm)

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
            Cn2 = np.interp(sounding["alt"], Cn2_z, Cn2)
        
        else:
        
            raise ValueError("{} is not valid value for 'method'. Use 'fiorino' or 'direct'.".format(method))

        #Replace infinities with NaNs
        Cn2[~np.isfinite(Cn2)] = np.nan

        #Compute logarithmic value
        logCn2 = np.log10(Cn2)

        #Remove unrealistically large values
        logCn2[logCn2 > 0] = np.nan

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
        if (not np.isfinite(vtemp[0])):
            vtemp[0] = temp[0]

        #Loop across the sounding profile while summing layer thickness
        z = [0]
        j = 1
        for i in range(1, pres.size):

            tvbar = am.layer_interp(pres[i-j], pres[i], (pres[i-j]+pres[i])/2.0, vtemp[i-j], vtemp[i])
            dz = at.hypsometric(pres[i-j], pres[i], tvbar)
            z.append(z[-j]+dz)

            #For handling layers with missing data
            if (not np.isfinite(dz)):
                j += 1
            else:
                j = 1

        #Convert to np array and add release elevation
        if units:
            gph = np.array(z)*self.sounding_units["alt"]+self.release_elv
        else:
            gph = np.array(z)+np.array(self.release_elv/self.sounding_units["alt"])

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
            if np.isfinite(ps):
                ind0 = i
                break

        #If first element isn't finite, then replace with temperature
        if (not np.isfinite(vtemp[ind0])):
            vtemp[ind0] = temp[ind0]

        #Loop across the sounding profile while summing the pressure changes
        p = [] #Grab the surface pressure in Pa
        j = 1
        for i in range(pres.size):

            if (i < ind0):
                p.append(np.nan)
                continue
            elif (i == ind0):
                p.append(pres[ind0])
                continue

            #Compute mean layer virtual temperature and layer thickness
            tvbar = am.layer_interp(pres[i-j], pres[i], (pres[i-j]+pres[i])/2.0, vtemp[i-j], vtemp[i])
            dz = sounding["alt"][i]-sounding["alt"][i-j]
            p.append(p[i-j]*np.exp(-(at.G*dz)/(at.RD*tvbar)))

            #For handling layers with missing data
            if (not np.isfinite(dz)):
                j += 1
            else:
                j = 1

        #Convert to np array and handle units
        if units:
            hpres = np.array(p)/100.0*self.sounding_units["pres"]
        else:
            hpres = np.array(p)/100.0

        return hpres

    #Method to calculate planetary boundary layer height (PBLH) from the sounding
    #The algorithm finds the first location where the environmental potential temperature is greater
    #than the surface. If a surface inversion is present, than the top of the inversion is called the PBL top.
    #If the PBLH cannot be calculated, it is set to -1
    def calculate_pblh(self):

        try:
            #Set Surbace based inversion flag to false initially
            self.sbi = False
            self.sbih = np.nan
            self.sbih_pres = np.nan
            self.sbih_ind = None

            #Strip units from sounding for use with atmos package
            sounding = self.strip_units()

            #Calculate sounding potential temperature, temperature gradient, and height AGL
            height = (sounding["alt"]-sounding["release_elv"]) #Height AGL
            tgrad = np.gradient(sounding["temp"])
            theta = at.pot_temp(sounding["pres"]*100.0, sounding["temp"]+273.15)

            #Locate elevated temperature inversion, call that the PBLH
            ind_pbl = np.where(tgrad > 0.0)[0][0]

            #If a surface based inversion exists, then look for the top of the inversion
            #Re-calculate the PBLH for the overlying remnant layer
            while (height[ind_pbl] < 100.0):

                #Handle SBI
                self.sbi = True
                ind_sbi = np.where(tgrad[ind_pbl:] <= 0.0)[0][0]+ind_pbl
                self.sbih = height[ind_sbi]*self.sounding_units["alt"]
                self.sbih_pres = (sounding["pres"])[ind_sbi]*self.sounding_units["pres"]
                self.sbih_ind = ind_sbi

                #Re-locate PBL top
                ind_pbl = np.where(tgrad[ind_sbi:] > 0.0)[0][0]+ind_sbi

            #If no inversion exists below 600 hPa, then use the mixing method
            if ((sounding["pres"])[ind_pbl] < 600.0):

                #Locate first location where surface parcel potential temp is less than environment
                ind_pbl = 0
                ind = 0
                while (ind_pbl < 2): #To avoid getting stuck at the surface
                    ind_pbl = np.where(theta[ind] < theta)[0][0]
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
            self.pblh = np.nan
            self.pblh_pres = np.nan
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
        ind1 = np.argmin((self.sounding["pres"]-layer1)**2)
        ind2 = np.argmin((self.sounding["pres"]-layer2)**2)

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

    # Method to calculate wind shear over a layer
    # Inputs:
    #  bottom, float, pressure or height level of layer bottom with units attached
    #  top, float, pressure or height of level of layer top with units attached
    #
    # Outputs:
    #  ushear, float, the u component of the wind shear in m/s
    #  vshear, float, the v component of the wind shear in m/s
    def calculate_shear(self, bottom, top):

        # Locate desired level (make a call to extract level)
        data1 = self.extract_level(bottom)
        data2 = self.extract_level(top)

        # Compute the shear
        ushear = data2['uwind']-data1['uwind']
        vshear = data2['vwind']-data1['vwind']

        # Return
        return ushear, vshear

    #Method to extract sounding variables at a single level
    #This method will interpolate between the two nearest levels.
    #Inputs:
    # level, float, pressure or height AGL level for which to pull values
    #Outputs
    # data, dictionary keyed with sounding variables, contains values at
    #       requested level
    def extract_level(self, level):

        # Check if unit attached
        try:
            level.dimensionality
        except:
            raise ValueError('"level" must have a unit attached')

        # If given altitude, convert to a pressure using hypsometric
        if (level.dimensionality == self.sounding_units['alt'].dimensionality):
            
            # Make level AMSL for consistency with the sounding
            level = level + self.release_elv

            # Bracket height level
            ind = np.argmin((level-self.sounding['alt'])**2)
            if (self.sounding["alt"][ind] <= level):
                tind = ind+2
                bind = ind
            else:
                tind = ind
                bind = ind-2

            # Make sure that the bottom index is not less than 0
            if (bind < 0):
                bind = 0
                tind = 2

            # Compute pressure of that level
            tbar = am.layer_average(self.sounding["pres"][bind:tind].to(self.units.Pa), self.sounding["temp"][bind:tind].to(self.units.K))
            a = (-1.0*(at.G*self.units.meter/self.units.second**2)*(level-self.sounding['alt'][bind]))
            b = ((at.RD)*(self.units.J/self.units.kg/self.units.K)*tbar).to_base_units()
            c = (a/b).to_base_units()

            level = self.sounding["pres"][bind]*np.exp(c.magnitude)
            

        #Force level unit to same as sounding
        try:
            level.to(self.sounding_units["pres"])
        except:
            level = (level*100.0*mu.Pa).to(self.sounding_units["pres"])

        #Create dictionary to hold data
        data = {}

        #Locate nearest levels that bracket the desired level
        ind = np.nanargmin(abs(self.sounding["pres"]-level))
        if (self.sounding["pres"][ind] >= level):
            tind = ind+1
            bind = ind
        elif (self.sounding["pres"][ind] < level):
            tind = ind
            bind = ind-1

        # Make sure that the bottom index is not less than 0
        if (bind < 0):
            bind = 0
            tind = 1

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
        wspd = np.sqrt(self.sounding["uwind"]**2+self.sounding["vwind"]**2)
        wdir = (270.0*self.units.degrees-
            np.arctan2(self.sounding["vwind"]/self.sounding_units["vwind"],
            self.sounding["uwind"]/self.sounding_units["uwind"])*180.0/np.pi*
            self.units.degrees)

        #Eliminate all values above 700 hPa
        mask = self.sounding["pres"] >= 700*self.units.hPa
        wspd = wspd[mask]
        wdir = wdir[mask]

        #Locate the maximum wind speed
        maxind = np.argmax(wspd)
        wmax = wspd[maxind]

        #Locate the minimum above the jet core
        minind = np.argmin(wspd[maxind:])
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
            jet["alt"] = np.nan*self.sounding_units["alt"]
            jet["pres"] = np.nan*self.sounding_units["pres"]
            jet["wspd"] = np.nan*self.sounding_units["uwind"]
            jet["wdeg"] = np.nan*self.units.degrees
            jet["falloff"] = np.nan*self.sounding_units["uwind"]

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

    ### Method to plot an advanced sounding
    ### Includes dozens of calculated variables alongside the advanced Skew-T plot and hodgraph.
    ### Inputs:
    ###  nbarbs, optional, integer, spacing between wind barbs
    ###  llj, optional, boolean, whether to check for a Low-level jet and highlight it. Defaults to False.
    ###  pblh, optional, boolean, whether to plot the PBLH height. Defaults to False.
    ###  sbi, optional, boolean, whether to plot the Surface-Based Inversion height. Defaults to False.
    ###  sonde_table, optional, boolean, whether to plot a text table of PySonde-unique values (LLJ, PBLH, etc.). Defaults to False.
    ###  save_dir, optional, string path name to the directory you wish to save the resultant plot
    ###
    ### Outputs:
    ###  fig, the pyplot figure object
    ###  skewt, the MetPy SkewT axis object

    def advanced_skewt(self, nbarbs=None, llj=False, pblh=False, sbi=False, sonde_table = False, save_dir = None):

        #Change default to be better for Skew-T plot:
        fig = pp.figure(figsize = (15,15), facecolor = 'whitesmoke',constrained_layout=True)
        gs = gridspec.GridSpec(2, 2)
        skew = SkewT(fig, rotation=45, subplot=gs[:, 0])

        #Plot the parcel profiles:
        prof = mc.parcel_profile(self.sounding['pres'], self.sounding['temp'][0], self.sounding['dewp'][0]).to('degC')
        ml_p, ml_t, ml_td = mc.mixed_parcel(self.sounding['pres'], self.sounding['temp'], self.sounding['dewp'], depth=50 * mu.hPa)
        mu_p, mu_t, mu_td, _ = mc.most_unstable_parcel(self.sounding['pres'], self.sounding['temp'], self.sounding['dewp'], depth=50 * mu.hPa)
        ml_prof = mc.parcel_profile(self.sounding['pres'], ml_t, ml_td).to('degC')
        mu_prof = mc.parcel_profile(self.sounding['pres'], mu_t, mu_td).to('degC')
        skew.plot(self.sounding['pres'], ml_prof, color='tab:blue', linewidth = 0.65, linestyle = '--') #ML profile
        skew.plot(self.sounding['pres'], mu_prof, color = 'maroon', linewidth = 0.65, linestyle = '--') #MU profile
        skew.plot(self.sounding['pres'], prof, color='black', linewidth=1.15) #SB profile

        #Plot the temp and dewpoint data:
        skew.plot(self.sounding['pres'], self.sounding['temp'], 'tab:red', linewidth=1.5)
        skew.plot(self.sounding['pres'], self.sounding['dewp'], 'tab:green', linewidth=1.5)

        #Plot Wind Barbs every 50 hPa:
        def pressure_interval(p,u,v,upper=100,lower=1000,spacing=50):

            intervals = list(range(upper,lower,spacing))

            ix = []
            for center in intervals:
                index = (np.abs(p-center)).argmin()
                if index not in ix:
                    ix.append(index)

            return p[ix],u[ix],v[ix]

        p_,u_,v_ = pressure_interval(self.sounding['pres'].m,self.sounding['uwind'].to('kt'),self.sounding['vwind'].to('kt'))

        if (nbarbs == None):
            skew.plot_barbs(p_,u_,v_, xloc=1.1)
        else:
            skew.plot_barbs(self.sounding["pres"][::nbarbs], self.sounding["uwind"].to('kt')[::nbarbs], self.sounding["vwind"].to('kt')[::nbarbs], xloc=1.1)

        #Add special lines to Skew-T plot:
        skew.plot_dry_adiabats(t0=np.arange(-60, 240, 10) * mu.degC, color='darkorange', linewidth = 0.5)
        skew.plot_moist_adiabats(color='lightgreen', linewidth = 0.5, linestyle='dotted')
        skew.plot_mixing_lines(pressure=np.arange(1000, 99, -20) * mu.hPa, color='tab:blue', linewidth = 0.5)

        #Plot lcl, lfc, and el:
        skew.plot(self.lcl_pres, self.lcl_temp, '_', label='LCL', mew = 2, color = 'black')
        skew.plot(self.lfc_pres, self.lfc_temp, '_', label='LFC',  mew = 2, color = 'black')
        skew.plot(self.el_pres, self.el_temp, '_', label='EL', mew = 2, color = 'black')
        skew.ax.text(self.lcl_temp.m - 5, self.lcl_pres.m +20, 'LCL', fontweight = 'bold')
        skew.ax.text(self.lfc_temp.m - 5, self.lfc_pres.m +20, 'LFC', fontweight = 'bold')
        skew.ax.text(self.el_temp.m - 5, self.el_pres.m, 'EL', fontweight = 'bold')
        skew.ax.text(self.sounding['temp'][0].m - 0, self.sounding['pres'][0].m + 50, f'{self.sounding["temp"][0].m * (9/5) + 32:5.0f}', color = 'tab:red', fontweight = 'bold', fontsize = 8)
        skew.ax.text(self.sounding['dewp'][0].m - 2, self.sounding['pres'][0].m + 50, f'{self.sounding["dewp"][0].m * (9/5) + 32:5.0f}', color = 'tab:green', fontweight = 'bold', fontsize = 8)

        #Shade CAPE and CIN areas on the Skew-T:
        skew.ax.fill_betweenx(self.sounding['pres'], self.sounding['temp'], prof, where=self.sounding['pres']>self.lfc_pres, facecolor='lightblue') #ensures fill is only below EL
        skew.shade_cape(self.sounding['pres'], self.sounding['temp'], prof, color='pink')

        ### Add levels of concern:
        #Freezing Level:
        try:
            fzl_index = np.argmin(np.abs(self.sounding['temp'] - 273.15 * mu.kelvin))
            fzl_pressure = self.sounding['pres'][fzl_index]
            skew.plot(fzl_pressure, self.sounding['temp'][fzl_index], '_', label='FRZ', mew = 2, color = 'dodgerblue')
            skew.ax.text(self.sounding['temp'][fzl_index].m - 5, fzl_pressure.m + 20, 'FRZ', c = 'dodgerblue', fontweight = 'bold')
        except:
            pass

        #Shade lightly the region of SCW/mixed phase ice:
        skew.ax.fill_betweenx(self.sounding['pres'], -30 * mu.degC, -10 * mu.degC, facecolor='lightgrey', alpha=0.3)

        ### Dual-y axis for hPa and km:
        heights = np.array([0, 1, 3, 6, 9, 12, 15]) * mu.km 
        h_idx = []
        for h in heights+self.release_elv:
            std_h = np.argmin(np.abs(self.sounding['alt'] - h))
            h_idx.append(std_h)
        for height_tick, p_tick in zip(heights, h_idx):
            trans, _, _ = skew.ax.get_yaxis_text1_transform(0)
            skew.ax.text(0.01, self.sounding["pres"][p_tick], '{:~.0f}'.format(height_tick), transform=trans, fontweight = 'bold', fontsize = 8, fontstyle = 'italic')

        # Add a horizontal grey line to illustrate ground level:
        skew.ax.axhline(self.sounding['pres'][0], color='saddlebrown', linestyle='-', alpha=0.7)

        #Set bounds:
        if self.sounding['temp'][0].m > 40.0:
            skew.ax.set_x_lim(-20, 50)
        elif self.sounding['temp'][0].m < -20.0:
            skew.ax.set_xlim(-50,20)
        else:
            skew.ax.set_xlim(-30,40)
        skew.ax.set_xlabel('Temperature ($^\circ$C)')
        skew.ax.set_ylabel('Pressure (hPa)')

        #Define heights as AGL and u,v as kt:
        h = self.sounding['alt'] - self.release_elv
        u = self.sounding['uwind'].to('kt')
        v = self.sounding['vwind'].to('kt')

        ##Effective Inflow Layer:
        try:
            (eil_idx_bot, eil_idx_top), _ = swx.get_effective_layer_indices(self)
            pbot = self.sounding['pres'][eil_idx_bot]
            ptop = self.sounding['pres'][eil_idx_top]
            hbot = self.sounding['alt'][eil_idx_bot]
            htop = self.sounding['alt'][eil_idx_top]


            y_min_norm = 100
            y_max_norm = self.sounding['pres'][0].m
            # Calculate the fractions with logarithmic scaling:
            p0_loc = ((np.log10(self.sounding['pres'][0].m) - np.log10(skew.ax.get_ylim()[0])) / (np.log10(skew.ax.get_ylim()[1]) - np.log10(skew.ax.get_ylim()[0])))
            frac_pbot = (np.log10(y_max_norm) - np.log10(pbot.m)) / (np.log10(y_max_norm) - np.log10(y_min_norm)) + p0_loc
            frac_ptop = (np.log10(y_max_norm) - np.log10(ptop.m)) / (np.log10(y_max_norm) - np.log10(y_min_norm)) + p0_loc

            # Plot the EIL line in axis coords:
            skew.ax.plot((0.95, 0.95), (frac_pbot, frac_ptop), c='mediumvioletred', label='EIL', lw = 1.75, transform=skew.ax.transAxes)
            skew.ax.text(0.95, frac_ptop, 'EIL', color='mediumvioletred', fontweight = 'bold', ha='center', va='bottom', transform=skew.ax.transAxes)
            
            #STP, SCP, Effective SRH/SHR:           
            shear_eff, SRH_eff = swx.get_eshr_esrh(self)
            shear_eff = shear_eff.to('kt')
            supcomp = swx.get_scp(self)
            supcomp = supcomp
        except:
            supcomp = np.nan
            SRH_eff = np.nan
            shear_eff = np.nan

        stp = swx.get_stp(self)

        ##Hail Growth Zone:
        # Find the index where temperature is closest to -10°C
        index_minus_10 = np.argmin(np.abs(self.sounding['temp'].m - (-10)))
        # Find the corresponding pressure level for -10°C
        pressure_at_minus_10 = self.sounding['pres'][index_minus_10]
        # Find the index where temperature is closest to -30°C
        index_minus_30 = np.argmin(np.abs(self.sounding['temp'].m - (-30)))
        # Find the corresponding pressure level for -30°C
        pressure_at_minus_30 = self.sounding['pres'][index_minus_30]

        if self.sounding['temp'][0].m > 0:
            frac_hg_bot = (np.log10(y_max_norm) - np.log10(pressure_at_minus_10.m)) / (np.log10(y_max_norm) - np.log10(y_min_norm)) + p0_loc
            frac_hg_top = (np.log10(y_max_norm) - np.log10(pressure_at_minus_30.m)) / (np.log10(y_max_norm) - np.log10(y_min_norm)) + p0_loc
            # Plot the HGZ line in axis coords:
            skew.ax.plot((0.95, 0.95), (frac_hg_bot, frac_hg_top), c='teal', label='HGZ', lw = 1.75, transform=skew.ax.transAxes)
            skew.ax.text(0.95, frac_hg_top, 'HGZ', color='teal', fontweight = 'bold', ha='center', va='bottom', transform=skew.ax.transAxes)
        elif self.sounding['temp'][0].m <= 0:
            pass

        ## PySonde-Unique Layers:
        #Add jet highlight
        if llj:
            jet = self.find_llj()
            if (jet["category"] != -1):
                llj_index = np.argmin(np.abs(self.sounding['pres'] - jet['pres']))
                llj_temp = self.sounding['temp'][llj_index]
                skew.plot(self.sounding['pres'][llj_index], llj_temp, '_', label='LLJ', mew = 2, color = 'darkviolet')
                skew.ax.text(llj_temp.m + 2.5, self.sounding['pres'][llj_index].m + 20, 'LLJ', c = 'darkviolet', fontweight = 'bold')
        
        #Add PBLH
        if pblh:
            pblh_index = np.argmin(np.abs(self.sounding['pres'] - self.pblh_pres))
            pblh_temp = self.sounding['temp'][pblh_index]
            skew.plot(self.sounding['pres'][pblh_index], pblh_temp, '_', label='PBLH', mew = 2, color = 'darkorange')
            skew.ax.text(pblh_temp.m + 2.5, self.sounding['pres'][pblh_index].m + 20, 'PBLH', c = 'darkorange', fontweight = 'bold')
    
        #Add SBI
        if (sbi and self.sbi):
            skew.ax.axhline(self.sbih_pres, color="black", linestyle=":")
            sbi_index = np.argmin(np.abs(self.sounding['pres'] - self.sbih_pres))
            sbi_temp = self.sounding['temp'][sbi_index]
            skew.plot(self.sounding['pres'][sbi_index], sbi_temp, '_', label='SBI', mew = 2, color = 'darkkhaki')
            skew.ax.text(sbi_temp.m + 2.5, self.sounding['pres'][sbi_index].m + 20, 'SBI', c = 'darkkhaki', fontweight = 'bold')
    
        ###Calculate other variables for plotting values:
        #Calculate the surface measurements:
        Es = mc.saturation_vapor_pressure(self.sounding['temp'])
        r_v = (0.622*Es)/(self.sounding['pres'] - Es)
        Tv = mc.virtual_temperature(self.sounding['temp'], r_v, molecular_weight_ratio=0.6219569100577033)
        RH = mc.relative_humidity_from_dewpoint(self.sounding['temp'], self.sounding['dewp'])
        theta_e = mc.equivalent_potential_temperature(self.sounding['pres'], self.sounding['temp'], self.sounding['dewp'])
        Tw = mc.wet_bulb_temperature(self.sounding['pres'], self.sounding['temp'], self.sounding['dewp'])
        heat_index = mc.heat_index(self.sounding['temp'], RH)
        speed = mc.wind_speed(u, v)
        windchill = mc.windchill(self.sounding['temp'], speed)
        mslp = at.mslp(self.sounding["pres"][0].m, self.sounding["temp"][0].to('K').m, self.sounding["alt"][0].m, (self.sounding["mixr"][0]/1000).m)
        
        #Calculate Bunkers Storm Motion for Hodograph plotting:
        bunkers_right, bunkers_left, wind_mean = mc.bunkers_storm_motion(self.sounding['pres'], u, v, h)
        BR =np.sqrt(bunkers_right[0]**2 + bunkers_right[1]**2)
        BL =np.sqrt(bunkers_left[0]**2 + bunkers_left[1]**2)
        MW = np.sqrt(wind_mean[0]**2 + wind_mean[1]**2)

        #Calculate DTM (Nixon):
        mask0 = h <= 0.5 * mu.km
        DTMu, DTMv = ((bunkers_right[0] + np.average(u[mask0]))/ 2), ((bunkers_right[1] +np.average(v[mask0]))/2)
        DTM = np.sqrt(DTMu**2+DTMv**2)

        #Calculate the storm-relative wind:
        sr_wind = np.sqrt(u[0]**2+v[0]**2) - MW
        sr_wind_dir = np.arctan(u[0].m/v[0].m) - np.arctan(wind_mean[1]/wind_mean[0])

        #Calculate Critical Angle:
        (u_storm, v_storm), *_ = mc.bunkers_storm_motion(self.sounding['pres'], u, v, h)
        crit_agl = mc.critical_angle(self.sounding['pres'], u, v, h, u_storm, v_storm)

        #Helicity:
        SRH01 = mc.storm_relative_helicity(h, u, v, 1*mu.km)
        SRH03 = mc.storm_relative_helicity(h, u, v, 3*mu.km)
        SRH06 = mc.storm_relative_helicity(h, u, v, 6*mu.km)

        #Bulk Shear:
        shearu01, shearv01 = mc.bulk_shear(self.sounding["pres"], u, v, depth = 1*mu.km)
        shear01 = np.sqrt(shearu01**2 + shearv01**2)
        shearu03, shearv03 = mc.bulk_shear(self.sounding["pres"], u, v, depth = 3*mu.km)
        shear03 = np.sqrt(shearu03**2 + shearv03**2)
        shearu06, shearv06 = mc.bulk_shear(self.sounding["pres"], u, v, depth = 6*mu.km)
        shear06 = np.sqrt(shearu06**2 + shearv06**2)

        ##Add Calculated Variables as Text:
        #Plot values as text beside the skew-t:
        ax2 = pp.gca()
        pp.title('{} ({:.2f}, {:.2f}) Sounding for {}'.format(self.release_site, self.release_lat.m, self.release_lon.m, self.release_time.strftime("%H%M UTC %d %b %Y")), fontweight='bold')

        pp.text(0, -0.15, f'LCL  = {self.lcl_pres.m:4.0f} hPa ({self.lcl_alt.to("km").m:2.2f} km)', color='black', transform=ax2.transAxes)
        pp.text(0, -0.19, f'LFC  = {self.lfc_pres.m:4.0f} hPa ({self.lfc_alt.to("km").m:2.2f} km)', color='black', transform=ax2.transAxes)
        pp.text(0, -0.23, f'EL   = {self.el_pres.m:4.0f} hPa ({self.el_alt.to("km").m:2.2f} km)', color='black', transform=ax2.transAxes)
        
        pp.text(0, -0.28, 'Surface Measurements', fontweight = 'bold',transform=ax2.transAxes)
        pp.text(0, -0.32, f'T = {self.sounding["temp"].magnitude[0] * (9/5) + 32:5.2f} $^\circ$F', color='black', transform=ax2.transAxes)
        pp.text(0, -0.36, f'$T_d$ = {self.sounding["dewp"].magnitude[0] * (9/5) + 32:5.2f} $^\circ$F', color='black', transform=ax2.transAxes)
        pp.text(0, -0.40, f'$T_w$ = {Tw.magnitude[0] * (9/5) + 32:5.2f} $^\circ$F', color='black', transform=ax2.transAxes)
        pp.text(0, -0.44, f'$T_v$  = {Tv.magnitude[0] * (9/5) + 32:5.2f} $^\circ$F', color='black', transform=ax2.transAxes)
        pp.text(0, -0.48, f'RH = {RH.magnitude[0] * 100:3.0f} %', color='black', transform=ax2.transAxes)
        pp.text(0, -0.52, f'P = {self.sounding["pres"].magnitude[0]:5.2f} hPa (MSLP={mslp:5.2f} hPa)', color='black', transform=ax2.transAxes)
        pp.text(0, -0.56, f'$\\theta$  = {self.sounding["pot_temp"].magnitude[0]:5.2f} K', color='black', transform=ax2.transAxes)
        pp.text(0, -0.60, f'$\\theta_e$  = {theta_e.magnitude[0]:5.2f} K', color='black', transform=ax2.transAxes)
        pp.text(0, -0.64, f'Heat Index  = {heat_index.magnitude[0]:5.2f} $^\circ$F', color='black', transform=ax2.transAxes)
        pp.text(0, -0.68, f'Windchill  = {windchill.magnitude[0] * (9/5) + 32:5.2f} $^\circ$F', color='black', transform=ax2.transAxes)

        pp.text(0.65, -0.15, 'CAPE and CIN', fontweight = 'bold',transform=ax2.transAxes)
        pp.text(0.65, -0.19, f'SBCAPE = {self.sfc_cape.magnitude:7.2f} J/kg', color='salmon', transform=ax2.transAxes)
        pp.text(0.65, -0.23, f'MLCAPE = {self.ml_cape.magnitude:7.2f} J/kg', color='tab:red', transform=ax2.transAxes)
        pp.text(0.65, -0.27, f'MUCAPE = {self.mu_cape.magnitude:7.2f} J/kg', color='darkred', transform=ax2.transAxes)
        pp.text(0.65, -0.31, f'3CAPE = {self.cape3.magnitude:7.2f} J/kg', color='red', transform=ax2.transAxes)
        pp.text(0.65, -0.35, f'SBCIN  = {self.sfc_cin.magnitude:7.2f} J/kg', color='deepskyblue', transform=ax2.transAxes)
        pp.text(0.65, -0.39, f'MLCIN = {self.ml_cin.magnitude:7.2f} J/kg', color='royalblue', transform=ax2.transAxes)
        pp.text(0.65, -0.43, f'MUCIN  = {self.mu_cin.magnitude:7.2f} J/kg', color='darkblue', transform=ax2.transAxes)

        pp.text(0.65, -0.48, 'Derived Measurements', fontweight = 'bold',transform=ax2.transAxes)
        pp.text(0.65, -0.52, f'LI   = {swx.LI(self).magnitude:5.2f}', color='black', transform=ax2.transAxes)
        pp.text(0.65, -0.56, f'PWAT  = {self.pw.to("in").m:5.2f} in', color='black', transform=ax2.transAxes)
        try:
            pp.text(0.65, -0.60, f'SUP  = {supcomp.magnitude:5.2f}', color='black', transform=ax2.transAxes)
        except:
            pp.text(0.65, -0.60, f'SUP  = {supcomp:5.2f}', color='black', transform=ax2.transAxes)
        pp.text(0.65, -0.64, f'STP  = {stp:5.2f}', color='black', transform=ax2.transAxes)
        pp.text(0.65, -0.68, f'WBI = {swx.get_wbi(self).magnitude:5.2f}',color='black', transform=ax2.transAxes)

        pp.text(1.3, -0.15, 'SRH and Bulk Shear', fontweight = 'bold',transform=ax2.transAxes)
        pp.text(1.3, -0.20, f'SRH 0-1 km  = {SRH01[0].magnitude:7.2f} m$^2$/s$^2$', color='palevioletred', transform=ax2.transAxes)
        pp.text(1.3, -0.24, f'SRH 0-3 km  = {SRH03[0].magnitude:7.2f} m$^2$/s$^2$', color='deeppink', transform=ax2.transAxes)
        pp.text(1.3, -0.28, f'SRH 0-6 km  = {SRH06[0].magnitude:7.2f} m$^2$/s$^2$', color='mediumvioletred', transform=ax2.transAxes)
        try:
            pp.text(1.3, -0.32, f'Effective SRH  = {SRH_eff.magnitude:7.2f} m$^2$/s$^2$', color='crimson', transform=ax2.transAxes)
        except:
            pp.text(1.3, -0.32, f'Effective SRH  = {SRH_eff:7.2f} m$^2$/s$^2$', color='crimson', transform=ax2.transAxes)
        pp.text(1.3, -0.36, f'SHR 0-1 km  = {shear01.magnitude:7.2f} kt', color='deepskyblue', transform=ax2.transAxes)
        pp.text(1.3, -0.40, f'SHR 0-3 km  = {shear03.magnitude:7.2f} kt', color='royalblue', transform=ax2.transAxes)
        pp.text(1.3, -0.44, f'SHR 0-6 km  = {shear06.magnitude:7.2f} kt', color='darkblue', transform=ax2.transAxes)
        try:
            pp.text(1.3, -0.48, f'Effective SHR  = {shear_eff.magnitude:7.2f} kt', color='cadetblue', transform=ax2.transAxes)
        except:
            pp.text(1.3, -0.48, f'Effective SHR  = {shear_eff} kt', color='cadetblue', transform=ax2.transAxes)

        pp.text(1.3, -0.52, f'Storm Relative Wind = {np.abs(sr_wind.m):3.0f}/{np.abs(sr_wind_dir.m)*57.2958+180:3.0f}', color='purple', transform=ax2.transAxes)
        pp.text(1.3, -0.56, f'Bunkers Left Mover = {BL.m:3.0f}/{np.rad2deg(np.arctan(bunkers_left[0].m/bunkers_left[1].m)) + 180:3.0f}', color='steelblue', transform=ax2.transAxes)
        pp.text(1.3, -0.60, f'Bunkers Right Mover = {BR.m:3.0f}/{np.rad2deg(np.arctan(bunkers_right[0].m/bunkers_right[1].m)) + 180:3.0f}', color='tab:red', transform=ax2.transAxes)
        pp.text(1.3, -0.64, f'Bunkers Mean Wind = {MW.m:3.0f}/{np.rad2deg(np.arctan(wind_mean[0].m/wind_mean[1].m)) + 180:3.0f}', color='dimgrey', transform=ax2.transAxes)
        pp.text(1.3, -0.68, f'Deviant Tornado Motion = {DTM.m:3.0f}/{np.rad2deg(np.arctan(DTMu.m/DTMv.m)) + 180:3.0f}', color='darkred', transform=ax2.transAxes)


        #Place boxes around printed calculations:
        ### (x1, x2), (y1, y2); clip_on=False allows the line to be beyond the skew-t plot

        #~LCL, LFC, EL Box~#
        pp.plot((-0.05, 0.5), (-0.12, -0.12), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #bottom
        pp.plot((-0.05, 0.5), (-0.24, -0.24), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #top
        pp.plot((0.5, 0.5), (-0.12, -0.24), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #right
        pp.plot((-0.05, -0.05), (-0.12, -0.24), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #left

        #~Surface Var. Box~#
        pp.plot((-0.05, 0.5), (-0.69, -0.69), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #bottom
        pp.plot((-0.05, 0.5), (-0.25, -0.25), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #top
        pp.plot((0.5, 0.5), (-0.25, -0.69), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #right
        pp.plot((-0.05, -0.05), (-0.25, -0.69), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #left

        #~CAPE/CIN Box~#
        pp.plot((0.60, 1.15), (-0.44, -0.44), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #bottom
        pp.plot((0.60, 1.15), (-0.12, -0.12), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #top
        pp.plot((1.15, 1.15), (-0.12, -0.44), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #right
        pp.plot((0.60, 0.60), (-0.12, -0.44), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #left

        #~Derived Var. Box~#
        pp.plot((0.60, 1.15), (-0.69, -0.69), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #bottom
        pp.plot((0.60, 1.15), (-0.45, -0.45), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #top
        pp.plot((1.15, 1.15), (-0.45, -0.69), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #right
        pp.plot((0.60, 0.60), (-0.45, -0.69), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #left

        #~SRH/SHR Box~#
        pp.plot((1.25, 1.90), (-0.69, -0.69), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #bottom
        pp.plot((1.25, 1.90), (-0.12, -0.12), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #top
        pp.plot((1.90, 1.90), (-0.12, -0.69), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #right
        pp.plot((1.25, 1.25), (-0.12, -0.69), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #left

        if sonde_table == True:
            pp.text(2.0, -0.15, 'PySonde Table', fontweight = 'bold',transform=ax2.transAxes)
            jet = self.find_llj()
            if (jet["category"] != -1):
                pp.text(2.0, -0.20, f'LLJ = {jet["pres"].m:5.0f} hPa ({jet["alt"].to("km").m:3.2f} km)', color='black', transform=ax2.transAxes)
            else:
                pp.text(2.0, -0.20, f'LLJ = nan', color='black', transform=ax2.transAxes)
            if pblh:
                pblh_alt = self.pblh.to('km')
                pp.text(2.0, -0.24, f'PBLH = {self.pblh_pres.m:5.0f} hPa ({pblh_alt.m:3.2f} km)', color='black', transform=ax2.transAxes)
                pp.text(2.0, -0.28, f'SBI = {self.sbi}', color='black', transform=ax2.transAxes)
            else:
                pp.text(2.0, -0.24, f'PBLH = nan', color='black', transform=ax2.transAxes)
                pp.text(2.0, -0.24, f'SBI = nan', color='black', transform=ax2.transAxes)

            pp.text(2.0, -0.32, f'Sfc. $C_n^2$ = {self.calculate_Cn2(method = "fiorino")[0]:3.2f}', color='black', transform=ax2.transAxes)
            ip500 = np.argmin((self.sounding['pres'] - 500*mu.hPa)**2)
            ip1000 = np.argmin((self.sounding['pres'] - 1000*mu.hPa)**2)
            thick1 = self.sounding['pres'][ip1000]
            thick2 = self.sounding['pres'][ip500]
            pp.text(2.0, -0.36, f'1000-500 Thickness = {self.calculate_layer_thickness(thick1, thick2).m:5.0f} m', color='black', transform=ax2.transAxes)

            #~SRH/SHR Box~#
            pp.plot((1.95, 2.45), (-0.37, -0.37), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #bottom
            pp.plot((1.95, 2.45), (-0.12, -0.12), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #top
            pp.plot((2.45, 2.45), (-0.12, -0.37), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #right
            pp.plot((1.95, 1.95), (-0.12, -0.37), 'k-', lw=1, transform=ax2.transAxes, clip_on=False) #left

        #Add Hodograph:
        ax3 = fig.add_subplot(gs[:, 1:])
        hodo = Hodograph(ax3, component_range=60.)
        ##Change size of hodograph lines??
        #Plot Hodograph Wind by Km AGL Intervals:
        mask = h <= 12 * mu.km
        intervals = np.array([0, 1, 3, 6, 9, 12]) * mu.km
        colors = ['magenta', 'tab:red', 'tab:green', 'tab:olive', 'cyan']
        hodo.add_grid(increment=10)
        hodo.plot(u[mask],v[mask],linewidth=2)
        hodo.ax.set_xlabel('Wind Speed (kt)')
        hodo.ax.set_ylabel(' ')
        hodo.ax.set_yticks([])
        hodo.ax.set_xlim(-25, 65)
        hodo.ax.set_ylim(-25, 65)
        hodo.plot_colormapped(u[mask], v[mask], h[mask], intervals=intervals, colors=colors, linewidth=2)

        #Plot Bunkers and DTM on Hodograph:
        hodo.plot(bunkers_right[0].m,bunkers_right[1].m, color='tab:red',marker='o', markersize=10,zorder=5,clip_on=True,label='Right Mover',fillstyle='none',mew = 2)
        hodo.plot(bunkers_left[0].m,bunkers_left[1].m, color='steelblue', marker='o', markersize=10, zorder=5,clip_on=True, label='Left Mover',fillstyle='none',mew = 2)
        hodo.plot(wind_mean[0].m,wind_mean[1].m, color='black', marker='s', markersize=10, zorder=5,clip_on=True, label='Mean Wind',fillstyle='none',mew = 2)
        hodo.plot(DTMu.m,DTMv.m, color='darkred', marker='v', markersize=10, zorder=5,clip_on=True, label='DTM',fillstyle='none',mew = 2)
        hodo.ax.plot((0,bunkers_right[0].m),(0,bunkers_right[1].m), color='midnightblue', linewidth=0.5, linestyle='-',clip_on=True)
        hodo.ax.plot((u[0].m,bunkers_right[0].m),(v[0].m,bunkers_right[1].m), color='midnightblue', linewidth=0.5, linestyle='dashdot',clip_on=True)
        hodo.ax.text(bunkers_right[0].m + 3 , bunkers_right[1].m - 3, 'BR', fontweight='bold',clip_on=True)
        hodo.ax.text(bunkers_left[0].m + 3 , bunkers_left[1].m - 3, 'BL', fontweight='bold',clip_on=True)
        hodo.ax.text(wind_mean[0].m + 3 , wind_mean[1].m - 3, 'MW', fontweight='bold',clip_on=True)
        hodo.ax.text(DTMu.m + 3 , DTMv.m - 3, 'DTM', fontweight='bold',clip_on=True)
        hodo.ax.text(bunkers_right[0].m, bunkers_right[1].m - 8, f'{BR.m:3.0f}/{np.rad2deg(np.arctan(bunkers_right[0].m/bunkers_right[1].m)) + 180:3.0f}',clip_on=True)
        hodo.ax.text(bunkers_left[0].m, bunkers_left[1].m - 8, f'{BL.m:3.0f}/{np.rad2deg(np.arctan(bunkers_left[0].m/bunkers_left[1].m)) + 180:3.0f}',clip_on=True)
        hodo.ax.text(wind_mean[0].m, wind_mean[1].m - 8, f'{MW.m:3.0f}/{np.rad2deg(np.arctan(wind_mean[0].m/wind_mean[1].m)) + 180:3.0f}',clip_on=True)
        hodo.ax.text(DTMu.m, DTMv.m - 8, f'{DTM.m:3.0f}/{np.rad2deg(np.arctan(DTMu.m/DTMv.m)) + 180:3.0f}',clip_on=True)
        hodo.ax.text(-23, -33, f'Critical Angle:{crit_agl.m:4.0f}$\degree$', fontweight = 'semibold', fontsize = 8, color = 'maroon')

        for i in intervals:
            ind = np.argmin((h-i)**2)
            hodo.ax.text(u[ind], v[ind], '{}'.format(i.magnitude), clip_on=True)

        #Show the plot:
        if save_dir != None:
            pp.savefig('{}{}_{}_({}_{}).png'.format(save_dir, self.release_time.strftime("%Y%m%d_%H%M"), self.release_site, self.release_lat, self.release_lon), dpi=150, bbox_inches='tight')

        return fig, skew

    ### Method to create an empty SkewT diagram
    ### Outputs:
    ###  fig, the pyplot figure object
    ###  skewt, the MetPy SkewT axis object
    def empty_skewt(self):

        #First create the figure and SkewT objects
        fig = pp.figure(figsize=(9,9))
        skewt = SkewT(fig, rotation=45)

        #Now set the limits
        pmask = self.sounding['pres'] >= 100.0*mu.hPa
        skewt.ax.set_xlim(-40, 60)
        skewt.ax.set_ylim(self.sounding['pres'][0], 100.0*mu.hPa)

        #Add the adiabats, etc
        skewt.plot_dry_adiabats(t0=np.arange(-40, 200, 10)*self.sounding_units["temp"])
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

            #Calculate the virtual temperature profile for hypsomteric equation
            vtemps = at.virt_temp(unitless["temp"]+273.15, unitless["mixr"]/1000.0)

            #Calculate the desired pressure levels using the hypsometric equation
            plevs = []
            for lev in levs:

                #Find the sounding levels that bracket this one
                ind = np.argmin((unitless["alt"]-lev)**2)
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
                plevs.append(unitless["pres"][bind]*np.exp(-at.G*(lev-unitless["alt"][bind])/(at.RD*vtemp)))

            #Convert pressure list to an array
            plevs = np.array(plevs)

            #Interpolate
            for k in unitless.keys():
                try:
                    if (k != "alt"):
                        isounding[k] = np.interp(np.log(plevs), np.log(unitless["pres"][::-1]), unitless[k][::-1],
                            left=np.nan, right=np.nan)
                    else:
                        isounding[k] = levs

                except:
                    pass

        elif (coords.upper() == "P"):

            #Interpolate
            for k in unitless.keys():
                try:
                    if (k != "pres"):
                        isounding[k] = np.interp(np.log(levs), np.log(unitless["pres"][::-1]), unitless[k][::-1],
                            left=np.nan, right=np.nan)
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
            zind = np.arange(sounding["alt"].size)[sounding["alt"]>maxz][0]
            
        #Check length of sounding to see if it fits within the RAMS 200pt limit
        #Interpolate if necessary (assume linear in log-p coordinates)
        if (sounding["alt"][:zind].size > 200):
            
            #Get first and last pressure levels for interpolation
            p0 = sounding["pres"][0]
            pf = sounding["pres"][zind]
            
            #Create evenly spaced pressure levels
            ip = np.linspace(p0, pf, 200, endpoint=True)
        
            #Interpolate       
            for k in ["dewp", "temp", "uwind", "vwind", "alt"]:
                sounding[k] = np.interp(np.log(ip[::-1]), np.log(sounding["pres"][::-1]), sounding[k][::-1])[::-1]
            sounding["pres"] = ip
            
        #Replace any missing values with 9999
        #9999 is based on the missing wind value in subroutine ARRSND of file rhhi.f90
        #in the RAMS initialization code.
        for k in ["dewp", "temp", "uwind", "vwind", "alt"]:
            sounding[k][~np.isfinite(sounding[k])] = 9999
            
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
        mask = np.isfinite(unitless["dewp"])
        for k in self.sounding_units.keys():
            unitless[k] = unitless[k][mask]

        #Do necessary unit conversions
        heights = unitless["alt"]
        temp = unitless["temp"]+273.15 #C -> K
        pres = unitless["pres"]*100.0 #hPa -> Pa

        #Calculate necessary surface variables
        #First compute 10m winds from the sounding using linear interpolation
        ind2 = (np.arange(heights.size, dtype="int")[(heights-10)>heights[0]])[0]
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
            pmid = np.linspace(pres[1], pres[-2], 900) #Only using 900 levels, to give WRF plenty of space
            bind = np.array(list((np.arange(0, pres.size, dtype="int")[(pres-pm)>0][-1] for pm in pmid)))
            tind = np.array(list((np.arange(0, pres.size, dtype="int")[(pres-pm)<=0][0] for pm in pmid)))
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
            self.sounding[k] = np.array(self.sounding[k], dtype="float")
            self.sounding[k] *= self.sounding_units[k]

        #Attach meta data
        self.release_time = datetime(2000, 1, 1)
        self.release_site = "Unknown"
        try: #In case of missing values in the lowest levels
            self.release_elv = self.sounding["alt"][np.isfinite(self.sounding["lat"]/self.sounding_units["alt"])][0]
            self.release_lat = self.sounding["lat"][np.isfinite(self.sounding["lat"]/self.sounding_units["lat"])][0]
            self.release_lon = self.sounding["lon"][np.isfinite(self.sounding["lat"]/self.sounding_units["lon"])][0]
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

                self.sounding["time"].append(np.nan)
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
        wspd = np.array(wspd, dtype="float")
        wdir = np.array(wdir, dtype="float")
        self.sounding["uwind"] = wspd*np.cos((270-wdir)*np.pi/180.0)
        self.sounding["vwind"] = wspd*np.sin((270-wdir)*np.pi/180.0)

        #Replace missing wind values with Nans
        self.sounding["uwind"][wspd == -999] = np.nan
        self.sounding["vwind"][wspd == -999] = np.nan

        #Now convert the other variables to arrays and attach units
        #And eliminate missing values
        for k in keys:
            self.sounding[k] = np.array(self.sounding[k], dtype="float")
            self.sounding[k][self.sounding[k] == -999] = np.nan
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
            self.sounding[sk] = np.array(fn.variables[fk][:])*self.sounding_units[sk]
            self.sounding[sk][self.sounding[sk] == missing_value*self.sounding_units[sk]] = np.nan*self.sounding_units[sk]

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
        self.sounding["time"] = np.zeros(self.sounding["temp"].shape)
        self.sounding["lon"] = np.ones(self.sounding["temp"].shape)*loc[0]
        self.sounding["lat"] = np.ones(self.sounding["temp"].shape)*loc[1]
        
        # Add units to the variables
        for k in ["time", "pres", "temp", "dewp", "uwind", "vwind", "lon", "lat", "alt"]:
            self.sounding[k] = np.array(self.sounding[k], dtype="float")*self.sounding_units[k]
        
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

        #Once the data has been read in, convert everything to np arrays and attach units
        for k in keys:
            self.sounding[k] = np.array(self.sounding[k], dtype="float")*self.sounding_units[k]

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
            missing_value = np.nan

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
            self.sounding[sk] = (np.array(fn.variables[sk][:])*mu(fn.variables[sk].units)).to(self.sounding_units[sk])
            self.sounding[sk][self.sounding[sk] == missing_value*self.sounding_units[sk]] = np.nan*self.sounding_units[sk]

        #Add other variables that may be missing from the file (such as in model runs)
        try:
            self.sounding["time"] = (np.array(fn.variables["time"][:])*mu(fn.variables["time"].units)).to(self.sounding_units["time"])
        except:
            self.sounding["time"] = np.ones(self.sounding["temp"].shape)*np.nan*self.sounding_units["time"]
        try:
            self.sounding["lon"] = (np.array(fn.variables["lon"][:])*mu(fn.variables["lon"].units)).to(self.sounding_units["lon"])
        except:
            self.sounding["lon"] = np.ones(self.sounding["temp"].shape)*np.nan*self.sounding_units["lon"]
        try:
            self.sounding["lat"] = (np.array(fn.variables["lat"][:])*mu(fn.variables["lat"].units)).to(self.sounding_units["lat"])
        except:
            self.sounding["lat"] = np.ones(self.sounding["temp"].shape)*np.nan*self.sounding_units["lat"]

        #Ensure that no values are below AGL (happens occasionally in models
        mask = (self.sounding["alt"] >= self.release_elv)
        if (np.sum(mask) < len(mask)):
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

        # Once the data has been read in, convert everything to np arrays and attach units
        for k in keys:
            self.sounding[k] = np.array(self.sounding[k], dtype="float")*self.sounding_units[k]

        # Ensure that heights are AMSL and not AGL
        if (self.sounding["alt"][0] < self.release_elv):
            self.sounding["alt"] += self.release_elv

        # Compute wind speed components in m/s
        wspd = np.array(wspd, dtype="float")*0.514
        wdir = np.array(wdir, dtype="float")

        self.sounding["uwind"] = wspd*np.cos((270-wdir)*np.pi/180.0)*self.sounding_units["uwind"]
        self.sounding["vwind"] = wspd*np.sin((270-wdir)*np.pi/180.0)*self.sounding_units["vwind"]

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
        self.sounding["time"] = np.ones(self.sounding["pres"].shape)*np.nan

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
        self.sounding["alt"] = np.array(self.sounding["alt"], dtype="float")
        self.sounding["uwind"] = np.array(self.sounding["uwind"], dtype="float")*self.sounding_units["uwind"]
        self.sounding["vwind"] = np.array(self.sounding["vwind"], dtype="float")*self.sounding_units["vwind"]
        theta = np.array(theta, dtype="float")
        qvapor = np.array(qvapor, dtype="float")

        #Calculate surface density
        tv = at.virt_temp(theta_sfc*(pres_sfc/100000.0)**(at.RD/at.CP), qvapor_sfc)
        rho_sfc = pres_sfc/(at.RD*tv)

        #Calculate pressure levels that correspond to sounding heights
        #Use the method present in module_initialize_scm_xy in WRF/dyn_em
        #Create arrays to hold values
        rho = np.zeros(theta.shape)
        pres = np.zeros(theta.shape)

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
        self.sounding["time"] = np.ones(temp.shape)*np.nan*self.sounding_units["time"]
        self.sounding["alt"] = self.sounding["alt"]*self.sounding_units["alt"]
        self.sounding["pres"] = pres/100.0*self.sounding_units["pres"] #Pa -> hPa
        self.sounding["temp"] = (temp-273.15)*self.sounding_units["temp"] #K -> 'C
        self.sounding["dewp"] = (at.dewpoint(at.wtoe(pres, qvapor))-273.15)*self.sounding_units["dewp"]
        self.sounding["lon"] = np.ones(temp.shape)*np.nan*self.sounding_units["lon"]
        self.sounding["lat"] = np.ones(temp.shape)*np.nan*self.sounding_units["lat"]

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
                self.sounding["time"].append(np.nan)
                self.sounding["pres"].append(dummy[0])
                self.sounding["temp"].append(dummy[2])
                self.sounding["dewp"].append(dummy[3])
                wdir.append(dummy[6])
                wspd.append(dummy[7])
                self.sounding["lon"].append(np.nan)
                self.sounding["lat"].append(np.nan)
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
            self.sounding[k] = np.array(self.sounding[k], dtype="float")*self.sounding_units[k]

        #Calculate wind components
        wspd = np.array(wspd, dtype="float")
        wdir = np.array(wdir, dtype="float")
        self.sounding["uwind"] = (np.array(wspd*np.cos((270-wdir)*np.pi/180.0), dtype="float")*mu.knot).to(self.sounding_units["uwind"])
        self.sounding["vwind"] = (np.array(wspd*np.sin((270-wdir)*np.pi/180.0), dtype="float")*mu.knot).to(self.sounding_units["vwind"])

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
        dewp = sounding['dewpoint']
        mask = np.isfinite(dewp)
        
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
        
        # Pull in profile
        for sk, soundk in zip(s1keys, sounding_keys):
            self.sounding[sk] = sounding[soundk].values[mask]*mu(sounding.units[soundk]).to(self.sounding_units[sk])

        # Handle lat/lons
        lons = header["longitude"].values
        lats = header["latitude"].values
        if lats.size < np.sum(mask):
            lons = (np.ones(np.sum(mask))*lons)*mu(header.units["longitude"]).to(self.sounding_units["lon"])
            lats = (np.ones(np.sum(mask))*lats)*mu(header.units["latitude"]).to(self.sounding_units["lat"])
        else:
            lons = lons[mask]*mu(header.units["longitude"]).to(self.sounding_units["lon"])
            lats = lats[mask]*mu(header.units["longitude"]).to(self.sounding_units["lon"])

        self.sounding['lon'] = lons
        self.sounding['lat'] = lats

        #Fill in time array with Nans
        self.sounding["time"] = np.ones(self.sounding["pres"].shape)*np.nan

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
            unitless = {"release_site":self.release_site, "release_lat":np.array(self.release_lat),
                "release_lon":np.array(self.release_lon), "release_elv":np.array(self.release_elv),
                "release_time":self.release_time, "pblh":np.array(self.pblh), "pblh_pres":np.array(self.pblh_pres),
                "lcl_pres":np.array(self.lcl_pres), "lcl_temp":np.array(self.lcl_temp), "lcl_alt":np.array(self.lcl_alt),
                "lfc_pres":np.array(self.lfc_pres), "lfc_temp":np.array(self.lfc_temp), "lfc_alt":np.array(self.lfc_alt)}
        except:
            unitless = {"release_site":self.release_site, "release_lat":np.array(self.release_lat),
                "release_lon":np.array(self.release_lon), "release_elv":np.array(self.release_elv),
                "release_time":self.release_time}

        #Now handle the other arrays
        for k in self.sounding.keys():
            unitless[k] = np.array(self.sounding[k])

        #Return unitless sounding
        return unitless
