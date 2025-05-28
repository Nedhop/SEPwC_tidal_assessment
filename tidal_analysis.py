#!/usr/bin/env python3
# import the modules you need here
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates
import datetime
import wget
import os
import numpy as np
import uptide
import pytz
import math
import pytest
import scipy
import typing
#def clean(data,column_name):
    #data.replace(to_replace=".*M$",value={column_name:np.nan},regex=True,inplace=True)
    #data.replace(to_replace=".*N$",value={column_name:np.nan},regex=True,inplace=True)
    #data.replace(to_replace=".*T$",value={column_name:np.nan},regex=True,inplace=True)
    #return data
def read_tidal_data(tidal_file):
    """
    Reads the tidal data from a text file and converts/cleans it into a
    usable data frame 
    
    """
    
    try:
        #since the text file is not in csv and has metadata at the start
        #it will have to skip the first rows and have each columns named
        #also have to skip index 10 as its an exampleof what the data is
        tidal_data = pd.read_csv(tidal_file, sep=r'\s+', skiprows=11, header=None)
        tidal_data.columns = ['Cycle', 'Date', 'Time', 'ASLVZZ01', 'Residual']
        columns_to_clean = ['ASLVZZ01', 'Residual']
        for column_name in columns_to_clean:
            tidal_data.replace(to_replace=r'.*[MTN]$', value=np.nan, regex=True, inplace=True)
            tidal_data[column_name] = pd.to_numeric(tidal_data[column_name], errors='coerce')
        tidal_data = tidal_data.rename(columns={'ASLVZZ01': 'Sea Level'})
        tdatetime = tidal_data['Date'] + ' ' + tidal_data['Time']
        dtformat = '%Y/%m/%d %H:%M:%S'
        tidal_data['DateTime'] = pd.to_datetime(tdatetime, format=dtformat)
        tidal_data['DateTimeString'] = tidal_data['DateTime']
        tidal_data = tidal_data.drop(columns=['Cycle', 'Date','Residual'])
        tidal_data = tidal_data.set_index('DateTime')
        tidal_data = tidal_data.sort_index()
        #if i
        #because of my persistent failure in joining due to apparent wrong columns
        print(f"Preview of data from {tidal_file}:")
        print(tidal_data.head())
        print("Data types:")
        print(tidal_data.dtypes)
        print("Column summary:")
        print(tidal_data.describe(include='all'))
        #data["Sea Level"]=data["Sea Level"].astype(float)
        return tidal_data
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at {tidal_file}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
def extract_single_year_remove_mean(year, data):
    """
    calculates the mean from  the selcted year and subtracts 
    it from the data
    """
    year_string_start = str(year)+"0101"
    year_string_end = str(year)+"1231"
    year_data = data.loc[year_string_start:year_string_end, ['Sea Level']]
    mmm = np.mean(year_data['Sea Level'])
    year_data['Sea Level'] -= mmm
    return year_data
def extract_section_remove_mean(start, end, data):
    """
    instead of just the data from a year this does the same proccess but
    for a specific time period
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    try:
        start_time = pd.to_datetime(start, format='%Y%m%d')
        end_time = pd.to_datetime(end, format='%Y%m%d')
        #the amount of data values is lower than expected so make sure all time
        #are included
        end_time = end_time + pd.Timedelta(hours=23, minutes=59, seconds=59)
        section_data = data.loc[start_time:end_time].copy()
        if section_data.empty:
            return section_data
        mean_sea_level = section_data['Sea Level'].mean()
        section_data['Sea Level'] = section_data['Sea Level'] - mean_sea_level
        return section_data
    except KeyError:
        raise KeyError(f"Date range '{start}' to '{end}' not found in data.")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")
def join_data(data1, data2):
    """
    this function combines two different data frames into one
    """
    try:
        joined_data = pd.concat([data1, data2])
        joined_data = joined_data.sort_index()
        return joined_data
    except Exception as e:
        print(f"Error joining dataframes: {e}")
        return None
    #combined = pd.concat([data1, data2]).sort_index()
    #return combined
#data = join_data(data1, data2)
def sea_level_rise(data):
    """
    This function takes the time and sea level data and shows the 
    relationship/slope between the two in the from of linear regression
    a consistant error i have faced is a fail due ot being 0.1 off the 
    desired value
    """
    data.dropna(axis = 0, how = 'any', subset=['Sea Level'], inplace = True)
    x = matplotlib.dates.date2num(data.index.to_pydatetime()) 
    y = data['Sea Level'].values
    slope, _, _, p_value, _ = scipy.stats.linregress(x, y)
    return slope, p_value
def tidal_analysis(data_segment, constituents, start_datetime):
    """
    this function uses harminic analysis to show the aplitude and phases of 
    tidal consituents for a specifc segmentof data
    """
    df = data_segment.dropna(subset=['Sea Level'])
    if df.empty:
        print("No Sea Level data available for tidal analysis.")
        return [], []
    data_segment.dropna(axis = 0, how = 'any', subset=['Sea Level'], inplace = True)
    sea_level = data_segment['Sea Level'].values
    tide = uptide.Tides(constituents)
    tide.set_initial_time(start_datetime)
    print(tide)
    print(sea_level)
    seconds_since = (data_segment.index.astype('int64').to_numpy()/1e9) - start_datetime.timestamp()
    amp,pha = uptide.harmonic_analysis(tide, data_segment['Sea Level'].to_numpy(), seconds_since)
    return amp, pha

def find_longest_contiguous_block(df, timestamp_col='timestamp', freq_minutes=15):
    """
    this function find the longest continous section of data within a
    specified date frame
    """
    print(df)
    df.dropna(axis = 0, how = 'any', subset=['Sea Level'], inplace = True)
    # Ensure datetime and sort
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(by=timestamp_col).reset_index(drop=True)
    print(df)

    expected_delta = pd.Timedelta(minutes=freq_minutes)
    longest_start, longest_len = 0, 1
    current_start, current_len = 0, 1

    for i in range(1, len(df)):
        delta = df.loc[i, timestamp_col] - df.loc[i-1, timestamp_col]
        if delta == expected_delta:
            current_len += 1
        else:
            if current_len > longest_len:
                longest_start, longest_len = current_start, current_len
            current_start = i
            current_len = 1
    # Final check after loop
    if current_len > longest_len:
        longest_start, longest_len = current_start, current_len

    return df.iloc[longest_start:longest_start + longest_len].reset_index(drop=True)

def get_longest_contiguous_data(data):
    """
    this function returns the datetimestring column back into Datetime
    for the longest continouos section of data 
    """
    df=find_longest_contiguous_block(data, timestamp_col='DateTimeString', freq_minutes=15)
    return df
if __name__ == '__main__':
    """
    #The program should print the tidal data, the sea-level rise and the longest contiguous period of data (i.e. without any missing data) from the data loaded.
    
    """
    parser = argparse.ArgumentParser(
                     prog="UK Tidal analysis",
                     description="Calculate tidal constiuents and RSL from tide gauge data",
                     epilog="Copyright 2024, Jon Hill"
                     )
    parser.add_argument("directory",
                    help="the directory containing txt files with data")
    parser.add_argument('-v', '--verbose',
                    action='store_true',
                    default=False,
                    help="Print progress")
    args = parser.parse_args()
    dirname = args.directory
    verbose = args.verbose 
#linear regression test
    gauge_files = ['data/1946ABE.txt', 'data/1947ABE.txt']
    data1 = read_tidal_data(gauge_files[1])
    print(data1)
    data2 = read_tidal_data(gauge_files[0])
    data = join_data(data1, data2)
    print(data1)
    print('=========')
    slope, p_value = sea_level_rise(data)
    print(slope,p_value)
    print(data1)
    print(data1.index)
    year=2000
    for file in os.listdir(dirname):
        if file.endswith(".txt"):
            thisfile = os.path.join(dirname, file)
            thisdata = read_tidal_data(thisfile)
            #test the file has actual data
            df = thisdata.dropna(subset=['Sea Level'])
            if df.empty:
                print("No Sea Level data available for tidal analysis.(2019 Dover)")
                continue
            #https://github.com/jhill1/SEPwC_tidal_assessment
            print(thisfile)
            print("linear regression on sea level (rise)")
            slope, p_value = sea_level_rise(thisdata)
            print("slope,p value")
            print(slope,p_value)
            print("================")
            #contiguous?
            constituents  = ['M2', 'S2']
            tide = uptide.Tides(constituents)
            tz = pytz.timezone("UTC")
            newdf=thisdata
            print(newdf.index[0])
            newdf=thisdata
            print("newdf")
            print(newdf)
            newdf = newdf.set_index('DateTimeString')
            newdf = newdf.sort_index()
            print("newdf")
            print(newdf)
            tide.set_initial_time(datetime.datetime(year,1,1,0,0,0))
            seconds_since = (newdf.index.astype('int64').to_numpy()/1e9) - datetime.datetime(year,1,1,0,0,0,tzinfo=tz).timestamp()
            amp,pha = uptide.harmonic_analysis(tide, newdf['Sea Level'].to_numpy(), seconds_since)
            print(constituents)
            print("amplitutde")
            print(amp)
            print("phase")
            print(pha)
            #import sys
            #sys.exit()
            if year==2000:
                alldata = thisdata
                year=year+1
            else:
                alldata = join_data(alldata,thisdata)
    print("Now we calculate for all data")
    print(alldata)
    slope, p_value = sea_level_rise(alldata)
    print("slope,p value")
    print(slope,p_value)
    print("================")
    print("contiguous data")
    df=find_longest_contiguous_block(alldata, timestamp_col='DateTimeString', freq_minutes=15)
    df['DateTime'] = pd.to_datetime(df['DateTimeString'] , format='%Y/%m/%d %H:%M:%S')
    df = df.set_index('DateTime')
    df = df.sort_index()
    print(df)
    newdf = df #make a copy for M2 S2
    slope, p_value = sea_level_rise(df)
    print("slope,p value")
    print(slope,p_value)
    print("================")
    #need to do it for all data
    newdf=alldata
    # Moving back to Fort Denison
    constituents  = ['M2', 'S2']
    tide = uptide.Tides(constituents)
    tz = pytz.timezone("UTC")
    print(newdf.index[0])
    #import sys
    #sys.exit()
    print("newdf")
    print(newdf)
    newdf = newdf.set_index('DateTimeString')
    newdf = newdf.sort_index()
    print("newdf")
    print(newdf)
    tide.set_initial_time(datetime.datetime(2000,1,1,0,0,0))
    seconds_since = (newdf.index.astype('int64').to_numpy()/1e9) - datetime.datetime(2000,1,1,0,0,0,tzinfo=tz).timestamp()
    amp,pha = uptide.harmonic_analysis(tide, newdf['Sea Level'].to_numpy(), seconds_since)
    print(constituents)
    print("amplitude")
    print(amp)
    print("phase")
    print(pha)
    #now make index of datetimestring
    #df['DateTime'] = pd.to_datetime(df['DateTimeString'] , format='%Y/%m/%d %H:%M:%S')
    #df = df.set_index('DateTime')
    #df = df.sort_index()
    #print(df)
    #data is good.
    #now sea level rise
    #slope, p_value = sea_level_rise(df)
    #print(slope,p_value)
    #The program should print the tidal data, the sea-level rise and the longest contiguous period of data (i.e. without any missing data) from the data loaded.