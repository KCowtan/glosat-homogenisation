#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: pkl_to_netcdf_archive.py
#------------------------------------------------------------------------------
# Verion 0.3
# 18 October, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------

import numpy as np
import pandas as pd
#import xarray as xr
import pickle
from datetime import datetime
import netCDF4
from netCDF4 import Dataset, num2date, date2num
import os

#------------------------------------------------------------------------------
# EXAMPLE: archive format
#------------------------------------------------------------------------------

#ds = xr.open_dataset('SC-Earth_tmean_observation.nc', decode_cf=True) # SC-Earth

#------------------------------------------------------------------------------
# LOAD: pkl
#------------------------------------------------------------------------------

df_temp = pd.read_pickle('DATA/df_temp_expect.pkl', compression='bz2') # dataframe of GloSAT absolute temperatures in degrees C

#Index(['year', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
#       'stationcode', 'stationlat', 'stationlon', 'stationelevation',
#       'stationname', 'stationcountry', 'stationfirstyear', 'stationlastyear',
#       'stationsource', 'stationfirstreliable', 'n1', 'n2', 'n3', 'n4', 'n5',
#       'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'e1', 'e2', 'e3', 'e4',
#       'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12', 's1', 's2', 's3',
#       's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12'],

# TRIM: to 1780 ( start of GloSAT )

df_temp = df_temp[ df_temp.year >= 1780 ]

# EXTRACT: station metadata

station_codes = df_temp['stationcode'].unique()
station_indices = np.arange( len(station_codes) )
station_latitudes = df_temp.groupby('stationcode')['stationlat'].mean()
station_longitudes = df_temp.groupby('stationcode')['stationlon'].mean()
station_elevations = df_temp.groupby('stationcode')['stationelevation'].mean()
station_names = df_temp.groupby('stationcode')['stationname'].unique()
station_names = np.array( [ station_names[i][0] for i in range(len(station_names)) ], dtype='object')
station_countries = df_temp.groupby('stationcode')['stationcountry'].unique()
station_countries = np.array([ station_countries[i][0] for i in range(len(station_countries)) ], dtype='object')
station_firstyears = df_temp.groupby('stationcode')['stationfirstyear'].mean().astype(int)
station_lastyears = df_temp.groupby('stationcode')['stationlastyear'].mean().astype(int)
station_sources = df_temp.groupby('stationcode')['stationsource'].mean().astype(int)
station_firstreliables = df_temp.groupby('stationcode')['stationfirstreliable'].mean().astype(int)

#------------------------------------------------------------------------------
# CONSTRUCT: station and local expectation timeseries 
#------------------------------------------------------------------------------

# INITIALISE: arrays and define time axis

time = pd.date_range(start='1780', end='2021', freq='MS')[0:-1]
tas = np.zeros( [len(station_codes), len(time)] )
lek_n = np.zeros( [len(station_codes), len(time)] )
lek_e = np.zeros( [len(station_codes), len(time)] )
lek_s = np.zeros( [len(station_codes), len(time)] )

for i in range( len(station_codes) ):

    station_code = station_codes[i]
    df = df_temp[df_temp.stationcode==station_code].sort_values(by='year').reset_index(drop=True) # extract station dataframe    
    o = np.array(df.groupby('year').mean().iloc[:,0:12]).ravel()
    n = np.array(df.groupby('year').mean().iloc[:,19:31]).ravel()
    e = np.array(df.groupby('year').mean().iloc[:,31:43]).ravel()
    s = np.array(df.groupby('year').mean().iloc[:,43:55]).ravel()      
    tas[i,:] = o
    lek_n[i,:] = n
    lek_e[i,:] = e
    lek_s[i,:] = s

#------------------------------------------------------------------------------
# EXPORT: data to netCDF-4
#------------------------------------------------------------------------------
            
# OPEN: netCDF file for writing
    
ncout = Dataset('OUT/station_archive.nc', 'w', format='NETCDF4')
    
# ADD: // global attributes
    
ncout.title = 'GloSAT station air temperature at 2m with local expectation'
ncout.source = 'GloSAT.p03'
ncout.version = 'GloSAT.p03-lek'
ncout.Conventions = 'CF-1.7'
ncout.reference = 'Osborn, T. J., P. D. Jones, D. H. Lister, C. P. Morice, I. R. Simpson, J. P. Winn, E. Hogan and I. C. Harris (2020), Land surface air temperature variations across the globe updated to 2019: the CRUTEM5 data set, Journal of Geophysical Research: Atmospheres, 126, e2019JD032352. https://doi.org/10.1029/2019JD032352'
ncout.institution = 'Climatic Research Unit, University of East Anglia / Met Office Hadley Centre / University of York'
ncout.licence = 'GloSAT is licensed under the Open Government Licence v3.0 except where otherwise stated. To view this licence, visit https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3'
ncout.history = 'File generated on {} (UTC) by {}'.format(datetime.utcnow().strftime('%c'), os.path.basename(__file__))
            
# CREATE: dimensions

ncout.createDimension( 'time', len(time) )
ncout.createDimension( 'stationindex', len(station_codes) )    

# SAVE: data to variables
    
# datatype specifiers include: 
# 'f4' (32-bit floating point), 
# 'f8' (64-bit floating point), 
# 'i4' (32-bit signed integer), 
# 'i2' (16-bit signed integer), 
# 'i8' (64-bit signed integer), 
# 'i1' (8-bit signed integer), 
# 'u1' (8-bit unsigned integer), 
# 'u2' (16-bit unsigned integer), 
# 'u4' (32-bit unsigned integer), 
# 'u8' (64-bit unsigned integer), 
# 'S1' (single-character string)
        
ncout_time = ncout.createVariable('time', 'i4', ('time',))
units = 'months since 1850-01-01 00:00:00'
ncout_time.setncattr('unit',units)
ncout_time[:] = [ date2num(time[i], units, calendar='360_day') for i in range(len(time)) ]
# calendar: 'standard’, ‘gregorian’, ‘proleptic_gregorian’ ‘noleap’, ‘365_day’, ‘360_day’, ‘julian’, ‘all_leap’, ‘366_day’
    
ncout_stationindex = ncout.createVariable('stationindex', 'u2', ('stationindex',))
ncout_stationindex.standard_name = "station_index"
ncout_stationindex.long_name = "sequential station_index"
ncout_stationindex[:] = station_indices    

ncout_stationcode = ncout.createVariable('stationcode', str, ('stationindex',))
ncout_stationcode.standard_name = "station_code"
ncout_stationcode.long_name = "sequential station_code"
ncout_stationcode[:] = station_codes

ncout_stationname = ncout.createVariable('stationname', str, ('stationindex',))
ncout_stationname.standard_name = "station_name"
ncout_stationname.long_name = "GloSAT station_name"
ncout_stationname[:] = station_names

ncout_stationcountry = ncout.createVariable('stationcountry', str, ('stationindex',))
ncout_stationcountry.standard_name = "station_country"
ncout_stationcountry.long_name = "GloSAT station_country"
ncout_stationcountry[:] = station_countries

ncout_latitude = ncout.createVariable('latitude', 'f4', ('stationindex',))
ncout_latitude.units = "degrees_north"
ncout_latitude.standard_name = "latitude"
ncout_latitude.long_name = "latitude"
ncout_latitude[:] = station_latitudes
    
ncout_longitude = ncout.createVariable('longitude', 'f4', ('stationindex',))
ncout_longitude.units = "degrees_east"
ncout_longitude.standard_name = "longitude"
ncout_longitude.long_name = "longitude"
ncout_longitude[:] = station_longitudes

ncout_elevation = ncout.createVariable('elevation', 'f4', ('stationindex',))
ncout_elevation.units = "m [amsl]"
ncout_elevation.standard_name = "elevation"
ncout_elevation.long_name = "elevation above mean sea level"
ncout_elevation[:] = station_elevations

ncout_stationfirstyear = ncout.createVariable('firstyear', 'u2', ('stationindex',))
ncout_stationfirstyear.standard_name = "first_year"
ncout_stationfirstyear.long_name = "first_year"
ncout_stationfirstyear[:] = station_firstyears

ncout_stationlastyear = ncout.createVariable('lastyear', 'u2', ('stationindex',))
ncout_stationlastyear.standard_name = "last_year"
ncout_stationlastyear.long_name = "last_year"
ncout_stationlastyear[:] = station_lastyears

ncout_stationsource = ncout.createVariable('source', 'u1', ('stationindex',))
ncout_stationsource.standard_name = "source"
ncout_stationsource.long_name = "GloSAT source code"
ncout_stationsource[:] = station_sources

ncout_stationfirstreliable = ncout.createVariable('firstreliable', 'u2', ('stationindex',))
ncout_stationfirstreliable.standard_name = "first_reliable_year"
ncout_stationfirstreliable.long_name = "first_reliable_year"
ncout_stationfirstreliable[:] = station_firstyears

ncout_tas = ncout.createVariable('tas', 'f4', ('stationindex', 'time'))
ncout_tas.units = 'deg_c'
ncout_tas.standard_name = 'air_temperature'
ncout_tas.long_name = 'air_temperature at approximately 2m'
ncout_tas.cell_methods = 'time: mean (interval: 1 month)'
ncout_tas.fill_value = -1.e+30
ncout_tas[:,:] = tas

ncout_n = ncout.createVariable('n', 'f4', ('stationindex', 'time'))
ncout_n.units = 'deg_c'
ncout_n.standard_name = 'climatological_baseline'
ncout_n.long_name = 'climatological_baseline normal (1961-1990)'
ncout_n.cell_methods = 'time: mean (interval: 1 month)'
ncout_n.fill_value = -1.e+30
ncout_n[:,:] = lek_n

ncout_e = ncout.createVariable('e', 'f4', ('stationindex', 'time'))
ncout_e.units = 'deg_c'
ncout_e.standard_name = 'local_expectation'
ncout_e.long_name = 'local_expectation ( from holdout Kriging )'
ncout_e.cell_methods = 'time: mean (interval: 1 month)'
ncout_e.fill_value = -1.e+30
ncout_e[:,:] = lek_e

ncout_s = ncout.createVariable('s', 'f4', ('stationindex', 'time'))
ncout_s.units = 'deg_c'
ncout_s.standard_name = 'local_expectation_stdev'
ncout_s.long_name = 'local_expectation_stdev'
ncout_s.cell_methods = 'time: mean (interval: 1 month)'
ncout_s.fill_value = -1.e+30
ncout_s[:,:] = lek_s

# CLOSE: netCDF file

ncout.close()
      
#------------------------------------------------------------------------------
print('** END')



    
