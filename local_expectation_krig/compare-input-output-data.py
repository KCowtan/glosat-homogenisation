#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: compare-input-output-data.py
#------------------------------------------------------------------------------
# Version 0.1
# 7 June, 2022
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
import pickle
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# SETTINGS
#------------------------------------------------------------------------------

filename_input = '../DATA/df_temp.pkl'
filename_output =  'RUNS/FULL/2DIGIT/full_ncycles_0/df_temp_expect.pkl'

#------------------------------------------------------------------------------
# LOAD: dataframes
#------------------------------------------------------------------------------

df_temp = pd.read_pickle( filename_input, compression='bz2' )
df_lek = pd.read_pickle( filename_output, compression='bz2' )
df_hca = pd.read_pickle( filename_clusters, compression='bz2' )

#------------------------------------------------------------------------------
# ARCHIVE: integrity
#------------------------------------------------------------------------------

stationcodes_input = df_temp['stationcode'].unique()
stationcodes_output = df_lek['stationcode'].unique()
stationlats_input = df_temp.groupby('stationcode').mean().stationlat.values
stationlats_output = df_lek.groupby('stationcode').mean().stationlat.values
stationlons_input = df_temp.groupby('stationcode').mean().stationlon.values
stationlons_output = df_lek.groupby('stationcode').mean().stationlon.values
stationyears_input = df_temp.groupby('stationcode').count().year.values
stationyears_output = df_lek.groupby('stationcode').count().year.values

nstations_input = len(stationcodes_input)
nstations_output = len(stationcodes_output)
stations_nodata_input = stationcodes_input[ stationyears_input == 0 ]
stations_nodata_output = stationcodes_output[ stationyears_output == 0 ]
stations_nolats_input = stationcodes_input[ ~np.isfinite( stationlats_input ) ]
stations_nolats_output = stationcodes_output[ ~np.isfinite( stationlats_output ) ]
stations_nolons_input = stationcodes_input[ ~np.isfinite( stationlons_input ) ]
stations_nolons_output = stationcodes_output[ ~np.isfinite( stationlons_output ) ]

print('N(stations) input=', nstations_input)
print('N(stations) output=', nstations_output)
print('stations (missing years) input=', len(stations_nodata_input), ' :', stations_nodata_input)
print('stations (missing years) output=', len(stations_nodata_output), ' :', stations_nodata_output)
print('stations (missing lat) input=', len(stations_nolats_input), ' :',  stations_nolats_input)
print('stations (missing lat) output=', len(stations_nolats_output), ' :',  stations_nolats_output)
print('stations (missing lon) input=', len(stations_nolons_input), ' :',  stations_nolons_input)
print('stations (missing lon) output=', len(stations_nolons_output), ' :',  stations_nolons_output)
 
#------------------------------------------------------------------------------
# COMPUTE: summary year stats
#------------------------------------------------------------------------------

print('YEAR: summary stats: input', df_temp.describe().year.astype(int) )
print('YEAR: summary stats: output', df_lek.describe().year.astype(int) )    
print('HCA:', df_hca.describe() )    
    
#------------------------------------------------------------------------------               
# COMPARE: test station list input-output
#------------------------------------------------------------------------------

test_stations = ['039530', '010010']
use_sorted = True

for s in range(len(test_stations)):
    
    stationcode = test_stations[s]

    if use_sorted == True:
        df_temp_station = df_temp[ df_temp.stationcode==stationcode ].sort_values(['year'], ascending=True).reset_index(drop=True)
        df_lek_station = df_lek[ df_lek.stationcode==stationcode ].sort_values(['year'], ascending=True).reset_index(drop=True)        
        titlestr = stationcode + ':' + df_temp_station.stationname[0] + ' (24m MA): sorted'
        figstr = stationcode + '_sorted.png'
        csvstr = stationcode + '_sorted.csv'
    else:
        df_temp_station = df_temp[ df_temp.stationcode==stationcode ].reset_index(drop=True)
        df_lek_station = df_lek[ df_lek.stationcode==stationcode ].reset_index(drop=True)
        titlestr = stationcode + ':' + df_temp_station.stationname[0] + ' (24m MA): unsorted'
        figstr = stationcode + '_unsorted.png'
        csvstr = stationcode + '_unsorted.csv'

    t_station_input = pd.date_range( start=str(df_temp_station.year.min()), end=str(df_temp_station.year.max()+1), freq='MS')[0:-1]                                                                                                                                          
    t_station_output = pd.date_range( start=str(df_lek_station.year.min()), end=str(df_lek_station.year.max()+1), freq='MS')[0:-1]                                                                                                                                          

    ts_station_input = []    
    for i in range(len(df_temp_station)):            
        monthly = df_temp_station.iloc[i,1:13]
        ts_station_input = ts_station_input + monthly.to_list()    
    ts_station_input = np.array(ts_station_input)   

    ts_station_output = []    
    for i in range(len(df_lek_station)):            
        monthly = df_lek_station.iloc[i,1:13]
        ts_station_output = ts_station_output + monthly.to_list()    
    ts_station_output = np.array(ts_station_output)   

    fig, ax = plt.subplots()
    plt.plot(t_station_input, pd.Series(ts_station_input).rolling(24).mean(), 'o', markersize=5, alpha=0.5, label='input')  
    plt.plot(t_station_output, pd.Series(ts_station_output).rolling(24).mean(), '.', markersize=2,  alpha=0.5, label='output')  
    plt.legend()
    plt.ylabel('Temperature')  
    plt.title( titlestr )
    fig.savefig( figstr, dpi=300)
    
    df_lek_station.to_csv( csvstr )


# input unique stations = 11865
# output unique stations = 10823
# len(dataframe_lek) / (2021-1781+1) == 10823 unique stations --> concurs

#------------------------------------------------------------------------------
print('** END')
