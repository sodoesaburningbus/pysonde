# This module contains mathematical functions and constants
# that are used in the PySonde package.
# Written by Christopher Phillips
#
# Requires:
#    Python 3+
#    Numpy
#
# History:
#    January 24, 2020 - First Write
#       Function to interpolate variables to a given layer
#    January 29th, 2020 - Added function
#       Function to average a variable over a given atmospheric layer.
#
# Copyright:
# This module may be freely distributed and used provided this header
# remains attached.
# ~Christopher Phillips

#Import required modules
import numpy

############################################################################
#++++++++++++++++++++++++++++++ FUNCTIONS +++++++++++++++++++++++++++++++++#
############################################################################

#This function interpolates variables between two layers
#It assumes that variables vary linearly with log-pressure
#Inputs:
# pbot, ptop, pmid, type = float or 2D array of floats, bottom, top,
# and desired pressure levels respectively.
# varbot, vartop, type = float or 2D array of floats, bottom and
#top layer of variable to interpolate.
#
#Outputs:
# varmid, float or 2D array of floats, the interpolated variable.
#
def layer_interp(pbot, ptop, pmid, varbot, vartop):

    #Compute interpolation weight
    alpha = numpy.log(pmid/pbot)/numpy.log(ptop/pbot)

    #Interpolate and return
    return alpha*vartop+(1-alpha)*varbot

#This function calculate the pressure-weighted layer average of a variable
#It naively skips layers with nans in them
#Inputs:
# pres, 1D array of floats, vertical pressure levels to average over (in Pa)
# var, 1D array of floats, the vertical profile of the variable to average
#
#Outputs:
# meanvar, float, the layer-averaged variable
def layer_average(pres, var):
    #Need to integrate over the whole layer in order to calculate average.
    #Will do so using mid-points of given pressure levels.
    var_integration = 0 #Variable to store sum for integration
    for i in range(var.size-1):
    
        #Interpolate to center of each sub-layer
        var_mid = layer_interp(pres[i], pres[i+1], (pres[i]+pres[i+1])/2,
            var[i], var[i+1])

        #Check for nan
        if numpy.isnan(var_mid):
            continue

        #Integrate this sub-layer
        var_integration += var_mid*numpy.log(pres[i+1]/pres[i])

    #Now divide integral by entire layer thickness (using log-pressure)
    #and return value.
    return var_integration/numpy.log(pres[-1]/pres[0])

#This function calculate the pressure-wighted layer average of a variable
#Inputs:
# pres, list of 2D array of floats, vertical pressure levels to average over
# var, list of 2D array of floats, the vertical profile of the variable to average
#
#Outputs:
# meanvar, float, the layer-averaged variable
def layer_average2d(pres, var):
    #Need to integrate over the whole layer in order to calculate average.
    #Will do so using mid-points of given pressure levels.
    var_integration = numpy.zeros(pres[0].shape) #Variable to store sums for integration
    for i in range(len(var)-1):
        #Interpolate to center of each sub-layer
        var_mid = layer_interp(pres[i], pres[i+1], (pres[i]+pres[i+1])/2,
            var[i], var[i+1])

        #Integrate this sub-layer
        var_integration += var_mid*numpy.log(pres[i+1]/pres[i])

    #Now divide integral by entire layer thickness (using log-pressure)
    #and return value.
    return var_integration/numpy.log(pres[-1]/pres[0])
    
