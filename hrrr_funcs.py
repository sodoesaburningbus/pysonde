### This module contains functions for retrieving variables from HRRR
### GRIB2 files.
### Written by Christopher Phillips

#Importing modules
import pysonde.atmos_thermo as at
import cartopy.crs as ccrs
import numpy


### Method to retrieve closest grid point to desired location
### Inputs:
###   point, tuple of floats, (lon, lat).
###   grib, pygrib file object
### Outputs:
###   (xi, yj), tuple of ints or list of ints, index of grid point closest to given point
###     xi corresponds to lon; yj corresponds to lat.
def get_point(point, grib):

    # Extract user longitudes and latitudes
    ulon = point[0]
    ulat = point[1]

    #Now store some basic grid info
    nx = grib[1].Nx
    ny = grib[1].Ny
    dx = grib[1].DxInMetres
    dy = grib[1].DyInMetres

    lons = numpy.reshape(grib[1].longitudes, grib[1].values.shape)-360.0 #West is negative
    lats = numpy.reshape(grib[1].latitudes, grib[1].values.shape)
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]    

    #Check that desired location is within model grid
    if ((ulon < extent[0]) or (ulon > extent[1]) or (ulat < extent[2]) or (ulat > extent[3])):
        raise ValueError("Point Lon: {}, Lat: {} is outside the analysis grid.".format(ulon,ulat))

    #Attach projection information to object
    proj_name = "Lambert Conformal"
    center_lon = (lons.min()+lons.max())/2
    proj = ccrs.LambertConformal(central_longitude=center_lon)
    pcp = ccrs.PlateCarree() #This is for transformations within the object

    #Grab first domain point
    (x0, y0) = proj.transform_point(lons[0,0], lats[0,0], pcp)

    #Transform point to hrrr projection
    (x, y) = proj.transform_point(ulon, ulat, pcp)

    #Calculate index of each point
    xi = int(round(abs(x-x0)/dx))
    yj = int(round(abs(y-y0)/dy))

    #Return indices to user
    return xi, yj


### Method to retreive sounding
### Inputs:
###   point, tuple of floats, (lon, lat).
###   grib, pygrib file object
###
### Outputs:
###   sounding, dictionary of lists containing sounding info, or list of such dictionaries with len(points)
###     dictionaries are keyed ["temp", "pres", "dewp", "uwind", "vwind"] for temperature (K), pressure (hPa),
###     dewpoint (K), zonal wind speed (m/s), meriodinal wind speed (m/s), and geopotential height (m) respectively.
###     Arrays are (Time, Level) with lowest level first.
def get_sounding(point, grib):

    #Variable name list
    var_names = ["Temperature", "U component of wind", "V component of wind", "Relative humidity", "Geopotential Height"]
    alt_var_names = ["Temperature", "U component of wind", "V component of wind", "Relative humidity", "Geopotential height"]
    dict_keys = ["temp", "uwind", "vwind", "dewp", "alt"]

    #Retrieve grid indices
    [xind, yind] = get_point(point, grib)

    #Create dictionary to hold sounding
    data = {}

    #Retrieve messages from files
    for [vn, dk, avn] in zip(var_names, dict_keys, alt_var_names):
    
        # Get grib messages for this variable
        try:
            messages = grib.select(name=vn, typeOfLevel="isobaricInhPa")
        except:
            messages = grib.select(name=avn, typeOfLevel="isobaricInhPa")

        #Loop over time and layers
        data[dk] = []
        dummy = numpy.zeros(len(messages))
        for i in range(len(messages)):
            dummy[i] = messages[i].values[yind, xind]
        data[dk].append(dummy)

    #Now force everything into arrays
    for k in data.keys():
        data[k] = numpy.squeeze(numpy.array(data[k]))

    # Rotate winds
    ru, rv = rotate_winds(data["uwind"], data["vwind"], point[0])
    data["uwind"] = ru
    data["vwind"] = rv

    #Grab pressure levels
    data["pres"] = list(m.level for m in messages[:])

    #Calculate dewpoint
    data["dewp"] = at.dewpoint(at.sat_vaporpres(data["temp"])*(data["dewp"]/100))

    #Reverse levels if pressure not start at surface
    if (data["pres"][0] < data["pres"][-1]):
        for k in data.keys():
            data[k] = numpy.flip(data[k], axis=0)

    # Get elevation at this location
    z = grib.select(name="Orography", level=0)[0].values[yind, xind]

    # Add in surface data
    sfc_p = grib.select(name="Surface pressure", level=0)[0].values[yind, xind]/100.0
    sfc_t = grib.select(name="2 metre temperature", level=2)[0].values[yind, xind]
    sfc_td = grib.select(name="2 metre dewpoint temperature", level=2)[0].values[yind, xind]
    sfc_u = grib.select(name="10 metre U wind component", level=10)[0].values[yind, xind]
    sfc_v = grib.select(name="10 metre V wind component", level=10)[0].values[yind, xind]
    sfc_z = z+2.0
    
    data["temp"] = numpy.concatenate([numpy.array(sfc_t, ndmin=1), data["temp"][data["pres"] < sfc_p]])
    data["dewp"] = numpy.concatenate([numpy.array(sfc_td, ndmin=1), data["dewp"][data["pres"] < sfc_p]])
    data["uwind"] = numpy.concatenate([numpy.array(sfc_u, ndmin=1), data["uwind"][data["pres"] < sfc_p]])
    data["vwind"] = numpy.concatenate([numpy.array(sfc_v, ndmin=1), data["vwind"][data["pres"] < sfc_p]])
    data["alt"] = numpy.concatenate([numpy.array(sfc_z, ndmin=1), data["alt"][data["pres"] < sfc_p]])
    data["pres"] = numpy.concatenate([numpy.array(sfc_p, ndmin=1), data["pres"][data["pres"] < sfc_p]])

    #Return sounding
    return data, z

### Method to rotate HRRR winds to Earth coordinate frame
### Based on https://rapidrefresh.noaa.gov/faq/HRRR.faq.html
### Inputs:
###   uwind, numpy nD array, zonal wind field
###   vwind, numpy nD array, meridional wind field
###   lons, numpy nD array, longitude coordinates
### Outputs:
###   ruwind, rotated zonal wind field
###   rvwind, rotated meridional wind field
def rotate_winds(uwind, vwind, lon):
    
    #Calculate angles for rotation
    angle = 0.622515*(lon+97.5)*0.017453
    sinx = numpy.sin(angle)
    cosx = numpy.cos(angle)

    #Rotate the winds
    ruwind = cosx*uwind+sinx*vwind
    rvwind = -sinx*uwind+cosx*vwind
    
    #Return rotated winds
    return ruwind, rvwind
