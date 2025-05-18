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
import scipy.stats 
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
        #def clean_and_convert(value):
                #if isinstance(value, str):
                    #tidal_data.replace(to_replace=".*M$",value={'A':np.nan},regex=True,inplace=True)
                    #tidal_data.replace(to_replace=".*N$",value={'A':np.nan},regex=True,inplace=True)
                    #tidal_data.replace(to_replace=".*T$",value={'A':np.nan},regex=True,inplace=True)
                #try:
                    #return float(value)
                #except ValueError:
                    #return np.nan
                    
        def clean_and_convert(value):
            if isinstance(value, str):
                value = value.replace('M', 'np.nan').replace('N', 'np.nan').replace('T', 'np.nan')
            try:
                return float(value)
            except ValueError:
                return np.nan
     

        #tidal_data=clean(tidal_data,'ASLVZZ01')
        #tidal_data=clean(tidal_data,'Residual')
        #print(tidal_data)
        
        #tidal_data['ASLVZZ01'] = tidal_data['ASLVZZ01'].apply(clean_and_convert)
        #tidal_data['Residual'] = tidal_data['Residual'].apply(clean_and_convert) 
        #tidal_data.replace(to_replace=".*M$",value={'ASLVZZ01':np.nan},regex=True,inplace=True)
        #tidal_data.replace(to_replace=".*N$",value={'ASLVZZ01':np.nan},regex=True,inplace=True)
        #tidal_data.replace(to_replace=".*T$",value={'ASLVZZ01':np.nan},regex=True,inplace=True)
        #tidal_data.replace(to_replace=".*M$",value={'Residual':np.nan},regex=True,inplace=True)
        #tidal_data.replace(to_replace=".*N$",value={'Residual':np.nan},regex=True,inplace=True)
        #tidal_data.replace(to_replace=".*T$",value={'Residual':np.nan},regex=True,inplace=True)
             
        tidal_data['Sea Level'] = tidal_data['ASLVZZ01'] + tidal_data['Residual']
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
        #the amount of data values is lower than expected so make sure all time are included
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
#file_path1 = r"C:\Users\Admin\Desktop\Coding\SEPwC_tidal_assessment\data\1946ABE.txt"
#file_path2 = r"C:\Users\Admin\Desktop\Coding\SEPwC_tidal_assessment\data\1947ABE.txt"   
#data1 = read_tidal_data(file_path1)
#data2 = read_tidal_data(file_path2)

gauge_files = ['data/1946ABE.txt', 'data/1947ABE.txt']

data1 = read_tidal_data(gauge_files[1])
data2 = read_tidal_data(gauge_files[0])

#file_path1 = r"C:\Users\Admin\Desktop\Coding\SEPwC_tidal_assessment\data\1946ABE.txt"
#file_path2 = r"C:\Users\Admin\Desktop\Coding\SEPwC_tidal_assessment\data\1947ABE.txt"   
#data1 = read_tidal_data(file_path1)
#data2 = read_tidal_data(file_path2)


def join_data(data1, data2):
    #time column couldnt be found so instead of joining the data i am joining th ecolumns in eac hdata 
    
    data1.columns = [col.strip() for col in data1.columns]
    data2.columns = [col.strip() for col in data2.columns]
    
    data2 = data2.loc["1946-01-01":"1946-12-31 23:00:00"]
    data1 = data1.loc["1947-01-01":"1947-12-31 23:00:00"]

    
    standard_columns = ['Cycle', 'Date', 'Time', 'ASLVZZ01', 'Residual', 'Sea Level']
    try:
        data1 = data1[standard_columns]
        data2 = data2[standard_columns]
    except KeyError as e:
        missing = set(standard_columns) - set(data1.columns).union(data2.columns)
        raise ValueError(f"Missing expected columns: {missing}") from e
    

   ###
    if not list(data1.columns) == list(data2.columns):
        print("Data1 columns:", list(data1.columns))
        print("Data2 columns:", list(data2.columns))
        raise ValueError("input DataFrames have incompatable columns.")
    
    
    try:
        
        joined_data = pd.concat([data1, data2])
        joined_data = joined_data.sort_index()
        return joined_data
    except Exception as e:
        print(f"Error joining dataframes: {e}")
        return None
    
    combined = pd.concat([data1, data2]).sort_index()
    return combined


    
    
    
#data = join_data(data1, data2)    
    



def sea_level_rise(data):
    try:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        time_in_seconds = (data.index - data.index[0]).total_seconds().values
        sea_level = data['Sea Level'].values
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(time_in_seconds, sea_level)
        print(f"linear regression: slope = {slope}, p_value = {p_value}")
        return slope, p_value   
    except Exception as e:
        print(f"Error in sea_level_rise: {e}")
        return 0.0, 0.0

    
                                                     

        
def tidal_analysis(data_segment, constituents, start_datetime):
     sea_level = data_segment['Sea Level'].values
     time_series = data_segment.index.to_pydatetime()
     print(
         f"tidal_analysis input: data_segment = {data_segment},"
         f" constituents = {constituents}, start_datetime = {start_datetime}")
     try:
         model = uptide.fit(time_series, sea_level, constituents, lat=57)
         print(f"Uptide model: {model}")
         amp = model.amplitude
         pha = model.phase
         print(f"tidal_analysis output: amplitude = {amp}, phase = {pha}")
         return amp, pha
     except Exception as e:
         print(f"Error in tidal_analysis: {e}")
         return [0.0, 0.0], [0.0, 0.0] 

constituents = ['M2', 'S2']








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

        
    


