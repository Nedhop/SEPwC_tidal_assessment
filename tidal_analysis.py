#!/usr/bin/env python3
"""
This code is able to process, analyse and visualise tidal gauge data
"""
import argparse
import datetime
import os
import pandas as pd
import matplotlib.dates
import numpy as np
import uptide
import pytz
import scipy
def read_tidal_data(tidal_file):
    """
    Reads the tidal data from a text file and converts/cleans it into a
    usable data frame 
    
    """
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
    print(f"Preview of data from {tidal_file}:")
    print(tidal_data.head())
    print("Data types:")
    print(tidal_data.dtypes)
    print("Column summary:")
    print(tidal_data.describe(include='all'))
    return tidal_data

def extract_single_year_remove_mean(year, esy_data):
    """
    calculates the mean from  the selcted year and subtracts 
    it from the data
    """
    year_string_start = str(year)+"0101"
    year_string_end = str(year)+"1231"
    year_data = esy_data.loc[year_string_start:year_string_end, ['Sea Level']]
    mmm = np.mean(year_data['Sea Level'])
    year_data['Sea Level'] -= mmm
    return year_data
def extract_section_remove_mean(start, end, es_data):
    """
    instead of just the data from a year this does the same proccess but
    for a specific time period
    """
    if not isinstance(es_data.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    start_time = pd.to_datetime(start, format='%Y%m%d')
    end_time = pd.to_datetime(end, format='%Y%m%d')
    end_time = end_time + pd.Timedelta(hours=23, minutes=59, seconds=59)
    section_data = es_data.loc[start_time:end_time].copy()
    if section_data.empty:
        return section_data
    mean_sea_level = section_data['Sea Level'].mean()
    section_data['Sea Level'] = section_data['Sea Level'] - mean_sea_level
    return section_data
def join_data(dataset1, dataset2):
    """
    this function combines two different data frames into one
    """
    joined_data = pd.concat([dataset1, dataset2])
    joined_data = joined_data.sort_index()
    return joined_data
def sea_level_rise(tidaldata):
    """
    This function takes the time and sea level data and shows the 
    relationship/slope between the two in the from of linear regression
    a consistant error i have faced is a fail due ot being 0.1 off the 
    desired value
    """
    tidaldata.dropna(axis = 0, how = 'any', subset=['Sea Level'], inplace = True)
    #x = matplotlib.dates.date2num(data.index.to_pydatetime())
    #y = data['Sea Level'].values
    slr_slope, _, _, slr_p_value, _ = scipy.stats.linregress(
        matplotlib.dates.date2num(tidaldata.index.to_pydatetime()), data['Sea Level'].values)
    return slr_slope, slr_p_value
def tidal_analysis(data_segment, cons, start_datetime):
    """
    this function uses harminic analysis to show the aplitude and phases of 
    tidal consituents for a specifc segmentof data
    """
    if data_segment.dropna(subset=['Sea Level']).empty:
        print("No Sea Level data available for tidal analysis.")
        return [], []
    data_segment.dropna(axis = 0, how = 'any', subset=['Sea Level'], inplace = True)
    sea_level = data_segment['Sea Level'].values
    ta_tide = uptide.Tides(cons)
    ta_tide.set_initial_time(start_datetime)
    print(ta_tide)
    print(sea_level)
    #ss seconds since
    ta_ss = (data_segment.index.astype('int64').to_numpy()/1e9) - start_datetime.timestamp()
    ta_amp,ta_pha = uptide.harmonic_analysis(ta_tide, data_segment['Sea Level'].to_numpy(), ta_ss)
    return ta_amp, ta_pha

def find_longest_contiguous_block(lcb_df, timestamp_col='timestamp', freq_minutes=15):
    """
    this function find the longest continous section of data within a
    specified date frame
    """
    print(lcb_df)
    lcb_df.dropna(axis = 0, how = 'any', subset=['Sea Level'], inplace = True)
    lcb_df = lcb_df.copy()
    lcb_df[timestamp_col] = pd.to_datetime(lcb_df[timestamp_col])
    lcb_df = lcb_df.sort_values(by=timestamp_col).reset_index(drop=True)
    print(lcb_df)

    expected_delta = pd.Timedelta(minutes=freq_minutes)
    longest_start, longest_len = 0, 1
    current_start, current_len = 0, 1

    for i in range(1, len(lcb_df)):
        delta = lcb_df.loc[i, timestamp_col] - lcb_df.loc[i-1, timestamp_col]
        if delta == expected_delta:
            current_len += 1
        else:
            if current_len > longest_len:
                longest_start, longest_len = current_start, current_len
            current_start = i
            current_len = 1
    if current_len > longest_len:
        longest_start, longest_len = current_start, current_len
    return lcb_df.iloc[longest_start:longest_start + longest_len].reset_index(drop=True)
def get_longest_contiguous_data(lc_data):
    """
    this function returns the datetimestring column back into Datetime
    for the longest continouos section of data 
    """
    return find_longest_contiguous_block(lc_data, timestamp_col='DateTimeString', freq_minutes=15)
if __name__ == '__main__':
    #The program should print the tidal data, sea level rise and longest
    #contiguous period of data for the dat pss through it, it adds each year of data
    #to the total data as it sorts each one until it has every year within it.
    #i faced a consistent error with dover so my code ignores the last year
    #as it was fualty, plus the test fails because it has python3 instead
    #of just python
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
    YEAR=2000
    for file in os.listdir(dirname):
        if file.endswith(".txt"):
            thisfile = os.path.join(dirname, file)
            thisdata = read_tidal_data(thisfile)
            df = thisdata.dropna(subset=['Sea Level'])
            if df.empty:
                print("No Sea Level data available for tidal analysis.(2019 Dover)")
                continue
            print(thisfile)
            print("linear regression on sea level (rise)")
            slope, p_value = sea_level_rise(thisdata)
            print("slope,p value")
            print(slope,p_value)
            print("================")
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
            tide.set_initial_time(datetime.datetime(YEAR,1,1,0,0,0))
            seconds_since = (newdf.index.astype('int64').to_numpy()/1e9) - datetime.datetime(
                YEAR,1,1,0,0,0,tzinfo=tz).timestamp()
            amp,pha = uptide.harmonic_analysis(tide, newdf['Sea Level'].to_numpy(), seconds_since)
            print(constituents)
            print("amplitutde")
            print(amp)
            print("phase")
            print(pha)
            if YEAR==2000:
                alldata = thisdata
                YEAR=YEAR+1
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
    newdf = df
    slope, p_value = sea_level_rise(df)
    print("slope,p value")
    print(slope,p_value)
    print("================")
    newdf=alldata
    constituents  = ['M2', 'S2']
    tide = uptide.Tides(constituents)
    tz = pytz.timezone("UTC")
    print(newdf.index[0])
    print("newdf")
    print(newdf)
    newdf = newdf.set_index('DateTimeString')
    newdf = newdf.sort_index()
    print("newdf")
    print(newdf)
    tide.set_initial_time(datetime.datetime(2000,1,1,0,0,0))
    seconds_since = (newdf.index.astype('int64').to_numpy()/1e9) - datetime.datetime(
        2000,1,1,0,0,0,tzinfo=tz).timestamp()
    amp,pha = uptide.harmonic_analysis(tide, newdf['Sea Level'].to_numpy(), seconds_since)
    print(constituents)
    print("amplitude")
    print(amp)
    print("phase")
    print(pha)
