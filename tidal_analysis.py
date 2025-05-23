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
        tidal_data['DateTime'] = pd.to_datetime(tidal_data['Date'] + ' ' + tidal_data['Time'], format='%Y/%m/%d %H:%M:%S')
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
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    year_data = data[data.index.year == int(year)].copy()
    if year_data.empty:
        return year_data
    mean_sea_level = year_data['Sea Level'].mean()
    year_data['Sea Level'] = year_data['Sea Level'] - mean_sea_level
    return year_data
def extract_section_remove_mean(start, end, data):
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
#start_date = "19460115"
#end_date = "19470310"
#data_segment = extract_section_remove_mean(start_date, end_date, data)
#when testing without gitbash to see where erros occur switch to
#file_path1 = r"C:\Users\Admin\Desktop\Coding\SEPwC_tidal_assessment\data\
    #1946ABE.txt"
#file_path2 = r"C:\Users\Admin\Desktop\Coding\SEPwC_tidal_assessment\data\
    #1947ABE.txt"
#data1 = read_tidal_data(file_path1)
#data2 = read_tidal_data(file_path2)
#gauge_files = ['data/1946ABE.txt', 'data/1947ABE.txt']
#data1 = read_tidal_data(gauge_files[1])
#data2 = read_tidal_data(gauge_files[0])
#file_path1 = r"C:\Users\Admin\Desktop\Coding\SEPwC_tidal_assessment\
    #data\1946ABE.txt"
#file_path2 = r"C:\Users\Admin\Desktop\Coding\SEPwC_tidal_assessment\
    #data\1947ABE.txt"
#data1 = read_tidal_data(file_path1)
#data2 = read_tidal_data(file_path2)
def join_data(data1, data2):
    #time column couldnt be found so instead of joining the data 
    #i am joining th ecolumns in eac hdata
    #data1.columns = [col.strip() for col in data1.columns]
    #data2.columns = [col.strip() for col in data2.columns]
    #data2 = data2.loc["1946-01-01":"1946-12-31 23:00:00"]
    #data1 = data1.loc["1947-01-01":"1947-12-31 23:00:00"]
    #join test continues to fail in the test because the test removes it
    #standard_columns = ['Cycle', 'Date', 'Time', 'Sea Level', 'Residual']
    #if not list(data1.columns) == list(data2.columns):
        #print("Data1 columns:", list(data1.columns))
        #print("Data2 columns:", list(data2.columns))
        #raise ValueError("input DataFrames have incompatable columns.")
    try:
        joined_data = pd.concat([data2, data1])
        #joined_data.dropna(subset=["Sea Level"], inplace=True)
        joined_data = joined_data.sort_index()
        return joined_data
    except Exception as e:
        print(f"Error joining dataframes: {e}")
        return None
    #combined = pd.concat([data1, data2]).sort_index()
    #return combined
#data = join_data(data1, data2)
def sea_level_rise(data):
    slope = 0.0
    p_value = 0.0
    data = data.dropna(subset=["Sea Level"])
    try:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        if 'Sea Level' not in data.columns:
            raise KeyError("DataFrame must contain a 'Sea Level' column.")
        if data.empty:
            print("warning: Input DataFrame is empty.")
            return slope, p_value
        #time_in_seconds = matplotlib.dates.date2num(data.index)
        time_in_seconds = matplotlib.dates.date2num(data.index)
        sea_level = data['Sea Level'].values
        if len(time_in_seconds) < 2:
            return slope, p_value
        if np.all(sea_level == sea_level[0]):
            return slope, p_value
        x_value = time_in_seconds
        y_value = sea_level
        try:
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_value, y_value)
        except Exception as e:
            raise RuntimeError(f"Error during linear regression: {e}")
        if np.isnan(slope):
            return 0.0, 0.0
        return slope, p_value
    except (TypeError, ValueError, KeyError, RuntimeError) as e:
        print(f"Error in sea_level_rise: {e}")
        return slope, p_value
    except Exception as e:
        print(f"Unexpected error in sea_level_rise: {e}")
        return slope, p_value
def tidal_analysis(data_segment, constituents, start_datetime):
    sea_level = data_segment['Sea Level'].values
    time_series = data_segment.index.to_pydatetime()
    tide = uptide.Tides(constituents)
    tide.set_initial_time(start_datetime)
    amp, pha = uptide.harmonic_analysis(tide, sea_level, time_series)
    return amp, pha
def get_longest_contiguous_data(data):
    return
if __name__ == '__main__':
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
    gauge_files = sorted([
        os.path.join(dirname, f)
        for f in os.listdir(dirname)
        if f.endswith('.txt')
        ])
    if len(gauge_files) < 2:
        raise ValueError("Need at least two .txt files in the directory to join data.")
    data1 = read_tidal_data(gauge_files[1])
    data2 = read_tidal_data(gauge_files[0])
    data = join_data(data1, data2)
    if verbose:
        print(f"Read and Joined: {gauge_files[1]} + {gauge_files[0]}")
        print(data.head())
        print(data.tail())
        