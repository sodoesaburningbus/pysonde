### This module contains functions for computing severe weather forecast indices.
### Indices included are: Supercell Composite, Significant Tornado,
### Whirly Boi Index, and Lifted Index.
### It also includes the supporting functions for finding effective inflow layer, etc.

# Import modules
import metpy.calc as mpc
from metpy.units import units
import numpy as np

### Function to get the Lifted Index
### Inputs:
###   self, a pysonde sounding object
###
### Outputs:
###   LI, a float
def LI(self):
    '''The lifted index uses the difference in envirmonental temperature and parcel
    temperature at 500 hPa to determine whether severe weather is possible.'''
    ip500 = np.argmin((self.sounding['pres'] - 500*units.hPa)**2)
    prof = mpc.parcel_profile(self.sounding['pres'], self.sounding['temp'][0], self.sounding['dewp'][0]).to('degC')
    lifted_index = self.sounding['temp'][ip500] - prof[ip500]
    return lifted_index


### Function to get the effective SHR and SRH
### Inputs:
###   self, a pysonde sounding object
###
### Outputs:
###   effective shear and effective SHR, floats
def get_eshr_esrh(self):
    # Get layers for the ESRH and Eshear calculations
    (srh_bind, srh_tind), (shear_bind, shear_tind) = get_effective_layer_indices(self)
    height = self.sounding['alt'][srh_bind]
    depth = self.sounding['alt'][srh_tind]-self.sounding['alt'][srh_bind]

    # Get effective SRH and effective shear
    psrh, nsrh, esrh = get_esrh(self, height, depth)
    eshear = np.sqrt((self.sounding['uwind'][shear_tind]-self.sounding['uwind'][shear_bind])**2+
                     (self.sounding['vwind'][shear_tind]-self.sounding['vwind'][shear_bind])**2)
    return eshear, esrh

### Function to get the Supercell Composite Parameter
### Inputs:
###   self, a pysonde sounding object
###
### Outputs:
###   scp, float, Supercell Composite parameter for the provided sounding
def get_scp(self):

    eshear, esrh = get_eshr_esrh(self)

    # Apply thresholds to ESHEAR
    if (eshear > 20*units('m/s')):
        eshear = 20*units('m/s')
    elif (eshear < 10*units('m/s')):
        eshear = 0*units('m/s')

    # Apply CIN thresholds
    muCIN = np.abs(self.mu_cin)
    if (muCIN > 40.0*units('J/kg')):
        muCIN = 40.0*units('J/kg')

    # Compute supercell composite parameter
    scp = self.mu_cape/(1000.0*units('J/kg'))*(esrh/(50.0*units('m^2/s^2')))*(eshear/(20.0*units('m/s')))#*(40.0*units('J/kg')/muCIN)

    return max(scp, 0)

### Function to get the Supercell Composite Parameter
### Inputs:
###   self, a pysonde sounding object
###
### Outputs:
###   stp, float, Significant Tornado Parameter for the provided sounding
def get_stp(self):

    # Get the 0-1 km SRH
    psrh, nsrh, srh = get_esrh(self, self.release_elv, 1000*units('m'))

    # Get the 0-6 km shear
    ushear6, vshear6 = self.calculate_shear(0*self.units.m, 6000*self.units.m)
    shear = np.sqrt(ushear6**2+vshear6**2)

    # Clip shear to bounds
    if (shear < 12.5*units('m/s')):
        shear = 0.0*units('m/s')
    elif (shear > 30*units('m/s')):
        shear = 30*units('m/s')

    #Pull the LCL height
    lcl = min(self.lcl_alt-self.release_elv, 2000.0*units('m'))
    lcl = max(lcl, 1000.0*units('m'))

    stp = self.sfc_cape/(1500*units('J/kg'))*(2000.0*units('m')-lcl)/(1000.0*units('m'))*srh/(150*units('m^2/s^2'))*shear/(20*units('m/s'))

    return max(stp, 0)


### Function to compute Effective SRH using MetPy
### Inputs:
###  self, a PySonde sounding object
###  height, float, unit aware height AMSL that is bottom of the ESRH layer
###  depth, float, unit aware depth of ESRH layer
###
### Outputs:
###  esrh, float, the effective storm relative helicity
def get_esrh(self, height, depth):

    # Estimate storm motion using Bunker's Right Mover
    brm, blm, bwm = mpc.bunkers_storm_motion(self.sounding['pres'], self.sounding['uwind'], self.sounding['vwind'], self.sounding['alt'])

    return mpc.storm_relative_helicity(self.sounding['alt']-self.release_elv, self.sounding['uwind'], self.sounding['vwind'],
                                       depth, bottom=height-self.release_elv, storm_u=brm[0], storm_v=brm[1])

### Function to compute the effective inflow layer and thunderstorm top
### following the methodology of Thompson et al. 2007
### Effective Storm-Relative Helicity and Bulk Shear in Supercell Thunderstorm Environments
### Weather and Forecasting Volume 22
### Inputs:
###  self, a PySonde sounding object
###
### Outputs:
###   esrh_layer, tuple of integer indices, (bottom, top),
###    for the effective inflow layer used to compute ESRH
###
###   eshear_layer, tuple of integer indices, (bottom, top),
###    for the effective storm layer used to compute effective wind shear
###
###   Output of None indicates that such a layer could not be identified
def get_effective_layer_indices(self):

    # Define criteria for effective inflow layer detection
    cape_thresh = 100.0*units('J/kg') # Minimum CAPE
    cin_thresh = -250*units('J/kg')   # Maximum CIN

    # Check that pressures decrease with height
    if (self.sounding['pres'][1] > self.sounding['pres'][0]):
        raise ValueError('ERROR: Pressure increasing with height. Check that lowest levels are first in profile.')

    # Compute the most unstable CAPE
    # If less than cape_thresh, then skip since no level will meet the thresholds        
    if (self.mu_cape < cape_thresh):
        print('WARNING: MUCAPE < 100 J/kg. No inflow layer found, returning (None, None)')
        return (None, None), (None, None)

    ### First identify the effective inflow region
    bind = None
    tind = None
    for k in range(self.sounding['pres'].size): # Find bottom of layer
    
        prof = mpc.parcel_profile(self.sounding['pres'][k:], self.sounding['temp'][k],
            self.sounding['dewp'][k])
        cape, cin = mpc.cape_cin(self.sounding['pres'][k:], self.sounding['temp'][k:],
            self.sounding['dewp'][k:], prof, which_lfc='bottom', which_el='top')
            
        if ((cape >= cape_thresh) and (cin >= cin_thresh)):
            bind = k
            break
     
    # Return early if cannot find an inflow layer
    if (bind == None):
        print('WARNING: No inflow layer found, returning (None, None)')
        return (bind, tind), (bind, None)
     
    for k in range(bind+1, self.sounding['pres'].size): # Find top of layer
    
        prof = mpc.parcel_profile(self.sounding['pres'][k:], self.sounding['temp'][k],
            self.sounding['dewp'][k])
        cape, cin = mpc.cape_cin(self.sounding['pres'][k:], self.sounding['temp'][k:],
            self.sounding['dewp'][k:], prof, which_lfc='bottom', which_el='most_cape')
            
        if ((cape < cape_thresh) or (cin < cin_thresh)):
            tind = k-1
            break
            
    # Check that a top of inflow level was found
    if (tind == None):
        print('WARNING: No top of inflow layer found, setting top index to last element')
        tind = self.sounding['pres'].size-1
        
    # Check that tind and bind are not the same
    if (tind == bind):
        print('WARNING: Bottom and top of inflow level are the same. Incrementing top index by one.')
        tind += 1

    # Locate equilibrium level of most unstable parcel
    # in the lowest 300 hPa
    pmask = (self.sounding['pres']-self.sounding['pres'][0]) <= 300*units('hPa')
    mu_p, mu_t, mu_d, mu_ind = mpc.most_unstable_parcel(self.sounding['pres'][pmask],
        self.sounding['temp'][pmask], self.sounding['dewp'][pmask])
    ind = np.argmin(np.abs(self.sounding['pres']-mu_p))
    el_p, el_t = mpc.el(self.sounding['pres'][ind:], self.sounding['temp'][ind:], self.sounding['dewp'][ind:])
    el_ind = np.argmin(np.abs(self.sounding['pres']-el_p))
    
    # Combine the equilibrium level and bottom of inflow to find
    # the half-storm depth index
    b_z = self.sounding['alt'][bind]
    el_z = self.sounding['alt'][el_ind]
    hsd_z = b_z+(el_z-b_z)/2.0
    hsd_ind = np.argmin(np.abs(self.sounding['alt']-hsd_z))

    return (bind, tind), (bind, hsd_ind)

### Function to compute Whirly Boi Index
def get_wbi(self):

    # Compute 0-1 km and 0-6 km shear
    ushear1, vshear1 = self.calculate_shear(0*self.units.m, 1000*self.units.m)
    ushear6, vshear6 = self.calculate_shear(0*self.units.m, 6000*self.units.m)

    # Ensure uniture
    ushear1 = ushear1.to(self.units.meter/self.units.second)
    vshear1 = vshear1.to(self.units.meter/self.units.second)
    ushear6 = ushear6.to(self.units.meter/self.units.second)
    vshear6 = vshear6.to(self.units.meter/self.units.second)

    print(np.sqrt(ushear1**2+vshear1**2).to(self.units.kt))
    print(np.sqrt(ushear6**2+vshear6**2).to(self.units.kt))

    # Compute shear angle
    alpha1km= np.arctan(vshear1/ushear1)
    alpha6km= np.arctan(vshear6/ushear6)
    alpha = alpha6km-alpha1km

    # Clean up CIN and CAPE to ensure they are on the proper bounds
    mu_cape = max(0, self.mu_cape)
    if (self.mu_cin < 0):
        mu_cin = self.mu_cin
    else:
        mu_cin = -1*self.mu_cin

    # Compute the WBI
    wbi = (mu_cape/(500.0*(self.units.J/self.units.kg)))*np.sqrt(ushear1**2+vshear1**2)/(self.units.meter/self.units.second)*(np.sin(alpha)**2)*(1.05**(mu_cin/(self.units.J/self.units.kg)))

    return wbi