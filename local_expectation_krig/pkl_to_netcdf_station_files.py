#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: pkl_to_netcdf_station_files.py
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

# Dataframe libraries:
import numpy as np
import pandas as pd
import xarray as xr
import pickle
from datetime import datetime
import netCDF4
from netCDF4 import Dataset, num2date, date2num

# System libraries:
import os

# System libraries:
import matplotlib.pyplot as plt

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

# EXTRACT: station codes

station_codes = df_temp['stationcode'].unique()

#------------------------------------------------------------------------------
# LOOP: over station archive and construct tas and local expectation timeseries 
#------------------------------------------------------------------------------

for i in range( len(station_codes) ):

    station_code = station_codes[i]
    df = df_temp[df_temp.stationcode==station_code].sort_values(by='year').reset_index(drop=True) # extract station dataframe
    
    station_code = df.stationcode.unique()[0]
    if len(df.stationlat.unique()) == 1:
        j = 0
    else:
        j = 1        
    station_lat = df.stationlat.unique()[j]
    station_lon = df.stationlon.unique()[j]
    station_elevation = df.stationelevation.unique()[j]
    station_name = df.stationname.unique()[j]
    station_country = df.stationcountry.unique()[j]
    station_firstyear = df.stationfirstyear.unique()[j].astype(int)
    station_lastyear = df.stationlastyear.unique()[j].astype(int)
    station_source = df.stationsource.unique()[j].astype(int)
    station_firstreliable = df.stationfirstreliable.unique()[j].astype(int)
              
    tas = np.array(df.groupby('year').mean().iloc[:,0:12]).ravel()
    n = np.array(df.groupby('year').mean().iloc[:,19:31]).ravel()
    e = np.array(df.groupby('year').mean().iloc[:,31:43]).ravel()
    s = np.array(df.groupby('year').mean().iloc[:,43:55]).ravel()      
    time = pd.date_range(start=str(df.year.iloc[0]), periods=len(tas), freq='MS')
        
#    plt.plot(time,tas-n)
#    plt.plot(time,e)    
		
    #------------------------------------------------------------------------------
    # EXPORT: data to netCDF-4
    #------------------------------------------------------------------------------

    # OPEN: netCDF file for writing

    ncout = Dataset('OUT/' + station_code + '.nc', 'w', format='NETCDF4')
    
    # ADD: // global attributes
    
    ncout.comment = 'GloSAT station air temperature at 2m'
    ncout.stationid = station_code
    ncout.stationlat = station_lat
    ncout.stationlon = station_lon
    ncout.stationelevation = station_elevation
    ncout.stationname = station_name
    ncout.stationcountry = station_country
    ncout.stationfirstyear = station_firstyear
    ncout.stationlastyear = station_lastyear
    ncout.stationsource = station_source
    ncout.stationfirstreliable = station_firstreliable
    ncout.history = 'File generated on {} (UTC) by {}'.format(datetime.utcnow().strftime('%c'), os.path.basename(__file__))
    ncout.institution = 'Climatic Research Unit, University of East Anglia/Met Office Hadley Centre/University of York'
    ncout.licence = 'GloSAT is licensed under the Open Government Licence v3.0 except where otherwise stated. To view this licence, visit https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3'
    ncout.reference = 'Osborn, T. J., P. D. Jones, D. H. Lister, C. P. Morice, I. R. Simpson, J. P. Winn, E. Hogan and I. C. Harris (2020), Land surface air temperature variations across the globe updated to 2019: the CRUTEM5 data set, Journal of Geophysical Research: Atmospheres, 126, e2019JD032352. https://doi.org/10.1029/2019JD032352'
    ncout.source = 'CRUTEM5'
    ncout.title = 'GloSAT station series'
    ncout.version = 'GloSAT.p03'
    ncout.Conventions = 'CF-1.7'
            
    # CREATE: dimensions
    
    ncout.createDimension( 'time', len(time) )
    
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

    ncout_tas = ncout.createVariable('tas', 'f4', ('time'))
    ncout_tas.units = 'deg_c'
    ncout_tas.standard_name = 'air_temperature'
    ncout_tas.long_name = 'station air_temperature at approximately 2m'
    ncout_tas.cell_methods = 'time: mean (interval: 1 month)'
    ncout_tas.fill_value = -1.e+30
    ncout_tas[:] = tas

    ncout_n = ncout.createVariable('n', 'f4', ('time'))
    ncout_n.units = 'deg_c'
    ncout_n.standard_name = 'climatological_baseline'
    ncout_n.long_name = 'station climatological_baseline normal (1961-1990)'
    ncout_n.cell_methods = 'time: mean (interval: 1 month)'
    ncout_n.fill_value = -1.e+30
    ncout_n[:] = n

    ncout_e = ncout.createVariable('e', 'f4', ('time'))
    ncout_e.units = 'deg_c'
    ncout_e.standard_name = 'local_expectation'
    ncout_e.long_name = 'station local_expectation ( from holdout Kriging )'
    ncout_e.cell_methods = 'time: mean (interval: 1 month)'
    ncout_e.fill_value = -1.e+30
    ncout_e[:] = e

    ncout_s = ncout.createVariable('s', 'f4', ('time'))
    ncout_s.units = 'deg_c'
    ncout_s.standard_name = 'local_expectation_stdev'
    ncout_s.long_name = 'station local_expectation_stdev'
    ncout_s.cell_methods = 'time: mean (interval: 1 month)'
    ncout_s.fill_value = -1.e+30
    ncout_s[:] = s
    
    # CLOSE: netCDF file
    
    ncout.close()

#------------------------------------------------------------------------------
print('** END')
      
#------------------------------------------------------------------------------
# UNIX: move to 2 digit folders and zip
#------------------------------------------------------------------------------
    
# for i in {01..99}; do mkdir $i; done &
# for i in {01..99}; do cd $i/; mv ../$i*.nc .; cd ../; done &     
# tar -zcvf station_files.tgz *
    
    
