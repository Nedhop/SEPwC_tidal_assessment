#!/usr/bin/env python3

# import the modules you need here
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import wget
import os
import numpy as np
import uptide
import pytz
import math
import pytest 

def read_tidal_data(tidal_file):
    try:
        #since the text file is not in csv and has metadata at the start 
        #it will have to skip the first rows and have each columns named
        #also have to skip index 10 as its an exampleof what the data is
        

        tidal_data = pd.read_csv(tidal_file, sep=r'\s+', skiprows=11, header=None)
        tidal_data.columns = ['Cycle', 'Date', 'Time', 'ASLVZZ01', 'Residual']
        def clean_and_convert(value):
                if isinstance(value, str):
                    value = value.replace('M', '')
                try:
                    return float(value)
                except ValueError:
                    return np.nan

        tidal_data['ASLVZZ01'] = tidal_data['ASLVZZ01'].apply(clean_and_convert)
        tidal_data['Residual'] = tidal_data['Residual'].apply(clean_and_convert) 
             
        tidal_data['Sea Level'] = tidal_data['ASLVZZ01'] + tidal_data['Residual']
        tidal_data['DateTime'] = pd.to_datetime(tidal_data['Date'] + ' ' + tidal_data['Time'], format='%Y/%m/%d %H:%M:%S')
        tidal_data = tidal_data.set_index('DateTime')
        tidal_data = tidal_data.sort_index()
        #if i 
        print(tidal_data.head())
        return tidal_data
          

    except FileNotFoundError:
            raise FileNotFoundError(f"Error: File not found at {tidal_file}")
    except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
        
    
gauge_files = ['data/1946ABE.txt', 'data/1947ABE.txt']

data1 = read_tidal_data(gauge_files[1])
data2 = read_tidal_data(gauge_files[0])

    


def join_data(data1, data2):
    #time column couldnt be found so instead of joining the data i am joining th ecolumns in eac hdata 
    if not set(data1.columns) == set(data2.columns):
        raise ValueError("input DataFrames have incompatable columns.")
    try:
        
        joined_data = pd.concat([data1, data2])
        joined_data = joined_data.sort_index()
        return joined_data
    except Exception as e:
        print(f"Error joining dataframes: {e}")
        return None
if data1 is not None and data2 is not None:
    joined_data = join_data(data2, data1)
    if joined_data is not None:
        print("Joined Data:")
        print(joined_data.head())
    else:
        print("Failed to join data.")
else:
    print("Failed to read one or both data files.")
    
data = join_data(data1, data2)    
    
def extract_single_year_remove_mean(year, data):
    if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
    year_data = data[data.index.year == int(year)].copy()
    mean_sea_level = year_data['Sea Level'].mean()
    year_data['Sea Level'] = year_data['Sea Level'] - mean_sea_level
    return year_data
  


def extract_section_remove_mean(start, end, data):
    if not isinstance(data.index, pd.Datetimeindex)
    try:
        start_time = pd.to_datetime(start, format='%Y%m%d')
        end_time = pd.to_datetime(end, format='%Y%m%d')
        section_data = data.loc[start_time:end_time].copy()
        mean_sea_level = section_data['Sea Level'].mean()
        section_data['Sea Level'] = section_data['Sea Level'] - mean_sea_level
        return section_data
    except KeyError as e:
        raise KeyError(f"Date range not found in data: {e}") from e
    except Exception as e:
        raise Exception(f"Error in extract_section_remove_mean: {e}") from e

  




def sea_level_rise(data):

                                                     
    return 

def tidal_analysis(data, constituents, start_datetime):


    return 

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
    


