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
        tidal_data = tidal_data.drop(columns=['Date', 'Time'])
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
        
    

    
file_path1 = r"C:\Users\Admin\Desktop\Coding\SEPwC_tidal_assessment\data\1947ABE.txt"
data1 = read_tidal_data(file_path1)
file_path2 = r"C:\Users\Admin\Desktop\Coding\SEPwC_tidal_assessment\data\1946ABE.txt"
data2 = read_tidal_data(file_path2)

def join_tidal_data(data2, data1):
    try:
        joined_data = pd.concat([data2, data1])
        joined_data = joined_data.sort_index()
        return joined_data
    except Exception as e:
        print(f"Error joining dataframes: {e}")
        return None
if data1 is not None and data2 is not None:
    joined_data = join_tidal_data(data2, data1)
    if joined_data is not None:
        print("Joined Data:")
        print(joined_data.head())
    else:
        print("Failed to join data.")
else:
    print("Failed to read one or both data files.")
    
    
    
def test_join_data():
    """
    Tests the join_data function with sample data.
    """
    gauge_files = ['data/1946ABE.txt', 'data/1947ABE.txt']

    data1 = read_tidal_data(gauge_files[1])
    data2 = read_tidal_data(gauge_files[0])
    data = join_tidal_data(data1, data2)

    assert "Sea Level" in data.columns
    assert type(data.index) == pd.core.indexes.datetimes.DatetimeIndex
    assert data['Sea Level'].size == 8760 * 2

    # check sorting (we join 1947 to 1946, but expect 1946 to 1947)
    assert data.index[0] == pd.Timestamp('1946-01-01 00:00:00')
    assert data.index[-1] == pd.Timestamp('1947-12-31 23:00:00')

    # check you get a fail if two incompatible dfs are given
    data2_copy = data2.copy()  # Create a copy to avoid modifying the original
    data2_copy.drop(columns=["Sea Level"], inplace=True)
    try:
        join_tidal_data(data1, data2_copy)
        print("ValueError was not raised when it should have been.")
    except ValueError:
        print("ValueError correctly raised for incompatible DataFrames.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the test
test_join_data()    
    
    
data = data1 + data2



if data1 is not None:
    assert "Sea Level" in data1.columns
    assert type(data1.index) == pd.core.indexes.datetimes.DatetimeIndex
    assert data1['Sea Level'].size == 8760
    assert pd.Timestamp('1947-01-01 00:00:00') in data1.index
    assert pd.Timestamp('1947-12-31 23:00:00') in data1.index
    assert data1['Sea Level'].isnull().any()
    assert pd.api.types.is_float_dtype(data1['Sea Level'])
    print("All assertions passed!")

data1['Date'] = pd.to_datetime(data1["Date"])
data1['Year'] = data1['Date'].dt.year
data1['Month'] = data1['Date'].dt.month
data1['Day'] = data1['Date'].dt.day
data1['Time'] = pd.to_datetime(data1['Time'], format='%H:%M:%S').dt.time
print(data1['Time'])




    return 0
    
def extract_single_year_remove_mean(year, data):
   

    return 


def extract_section_remove_mean(start, end, data):


    return 


def join_data(data1, data2):

    return 



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
    


