#!/bin/env python
# Created on March 25, 2020
#  by Keith Cherkauer
#Updated 4/22/20 by wagne216

# This script servesa as the solution set for assignment-10 on descriptive
# statistics and environmental informatics.  See the assignment documention 
# and repository at:
# https://github.com/Environmental-Informatics/assignment-10.git for more
# details about the assignment.
#
# The goals of this script are: 
#   1. Import river datasets
#   2. Removes invalid data
#   3. Clips to data range 10/1/1969 to 9/30/2019
#   4. Outputs stats to .csv and tab-delim'ed files


import pandas as pd
import scipy.stats as stats
import numpy as np
from scipy.stats import skew 

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""

    # INPUT: DF, 2 dates
    # redefine MV to make it local since it's not a direct input to this function
    MissingValues = DataDF["Discharge"].isna().sum() 
    
    # given the dict, want to apply clip to each dataframe
    b = DataDF.index < startDate # before
    a = DataDF.index > endDate # after
    DataDF = DataDF[startDate:endDate] # clip
    # OUTPUT 1: Dataframe (clipped)
    # must add to missing val's based on filename, but then also redefine as the output
    MissingValues = MissingValues + b.sum() + a.sum() # add count clipped
    # OUTPUT 2: add to mv dictionary for specific file

    return( DataDF, MissingValues )

def CalcTqmean(Qvalues):
    """This function computes the Tqmean of a series of data, typically
       a 1 year time series of streamflow, after filtering out NoData
       values.  Tqmean is the fraction of time that daily streamflow
       exceeds mean streamflow for each year. Tqmean is based on the
       duration rather than the volume of streamflow. The routine returns
       the Tqmean value for the given data array."""

    # INPUT: SERIES
    a = Qvalues.index[Qvalues > Qvalues.mean()]
    Tqmean = len(a)/len(Qvalues)
    # OUTPUT: SINGLE VALUE
           
    return ( Tqmean )

def CalcRBindex(Qvalues):
    """This function computes the Richards-Baker Flashiness Index
       (R-B Index) of an array of values, typically a 1 year time
       series of streamflow, after filtering out the NoData values.
       The index is calculated by dividing the sum of the absolute
       values of day-to-day changes in daily discharge volumes
       (pathlength) by total discharge volumes for each year. The
       routine returns the RBindex value for the given data array."""   

    # INPUT: SERIES
    a = Qvalues.diff()
    RBindex = a.abs().sum()/Qvalues.sum()
    # OUTPUT: SINGLE VALUE
       
    return ( RBindex )

def Calc7Q(Qvalues):
    """This function computes the seven day low flow of an array of 
       values, typically a 1 year time series of streamflow, after 
       filtering out the NoData values. The index is calculated by 
       computing a 7-day moving average for the annual dataset, and 
       picking the lowest average flow in any 7-day period during
       that year.  The routine returns the 7Q (7-day low flow) value
       for the given data array."""
          
    # INPUT: SERIES
    a = Qvalues.rolling(window=7).mean()
    val7Q = np.min(a)   
    # OUTPUT: SINGLE VALUE

    return ( val7Q )

def CalcExceed3TimesMedian(Qvalues):
    """This function computes the number of days with flows greater 
       than 3 times the annual median flow. The index is calculated by 
       computing the median flow from the given dataset (or using the value
       provided) and then counting the number of days with flow greater than 
       3 times that value.   The routine returns the count of events greater 
       than 3 times the median annual flow value for the given data array."""

    # INPUT: SERIES
    a = Qvalues.median()
    b = Qvalues.index[Qvalues> 3*a]
    median3x = len(b)  
    # OUTPUT: SINGLE VALUE
    
    return ( median3x )

def GetAnnualStatistics(DataDF):
    """This function calculates annual descriptive statistcs and metrics for 
    the given streamflow time series.  Values are retuned as a dataframe of
    annual values for each water year.  Water year, as defined by the USGS,
    starts on October 1."""
    
    # Resample data to water years to create new df  with stats as series  w. water year as index
    mean =  DataDF['Discharge'].resample('AS-OCT').mean()
    peak = DataDF['Discharge'].resample('AS-OCT').max()
    med = DataDF['Discharge'].resample('AS-OCT').median()
    # for coeff of var. div std dev/mean then x100
    sd = DataDF['Discharge'].resample('AS-OCT').std()
    cov = sd/mean*100
    skw = DataDF['Discharge'].resample('AS-OCT').apply(skew)
    tq = DataDF['Discharge'].resample('AS-OCT').apply(CalcTqmean) # DF of series will be the Qvalues input
    rb = DataDF['Discharge'].resample('AS-OCT').apply(CalcRBindex)
    sq = DataDF['Discharge'].resample('AS-OCT').apply(Calc7Q)
    tm = DataDF['Discharge'].resample('AS-OCT').apply(CalcExceed3TimesMedian)
    
    WYDataDF = pd.DataFrame({'Mean Flow':mean,'Peak Flow':peak,'Median Flow':med,\
            'Coeff Var':cov,'Skew':skw,'Tqmean':tq,'R-B Index':rb,'7Q':sq,'3xMedian':tm})
  
    return ( WYDataDF )

def GetMonthlyStatistics(DataDF):
    """This function calculates monthly descriptive statistics and metrics 
    for the given streamflow time series.  Values are returned as a dataframe
    of monthly values for each year."""

    # INPUT: DF
    mean =  DataDF['Discharge'].resample('MS').mean()
    sd = DataDF['Discharge'].resample('MS').std()
    cov = sd/mean*100
    tq = DataDF['Discharge'].resample('MS').apply(CalcTqmean) # DF of series will be the Qvalues input
    rb = DataDF['Discharge'].resample('MS').apply(CalcRBindex)

    MoDataDF = pd.DataFrame({'Mean Flow':mean,'Coeff Var':cov,'Tqmean':tq,'R-B Index':rb})
    # OUTPUT:  NEW DF
    
    return ( MoDataDF )

def GetAnnualAverages(WYDataDF):
    """This function calculates annual average values for all statistics and
    metrics.  The routine returns an array of mean values for each metric
    in the original dataframe."""
    
    # INPUT: DF
    AnnualAverages = pd.Series({'Mean Flow':WYDataDF['Mean Flow'].mean(),\
                                   'Peak Flow':WYDataDF['Peak Flow'].mean(),\
                                   'Median Flow':WYDataDF['Median Flow'].mean(),\
                                   'Coeff Var':WYDataDF['Coeff Var'].mean(),\
                                   'Skew':WYDataDF['Skew'].mean(),\
                                   'Tqmean':WYDataDF['Tqmean'].mean(),\
                                   'R-B Index':WYDataDF['R-B Index'].mean(),\
                                   '7Q':WYDataDF['7Q'].mean(),\
                                   '3xMedian':WYDataDF['3xMedian'].mean()})
    # OUTPUT:  SERIES 
    
    return( AnnualAverages )

def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics.  The routine returns an array of mean values 
    for each metric in the original dataframe."""
    
    # INPUT: DF
    # group data by the month and then take mean of all columns
    MonthlyAverages = MoDataDF.groupby(by=[MoDataDF.index.month]).mean()
    # OUTPUT:  SERIES    
    
    return( MonthlyAverages )

# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define filenames as a dictionary
    # NOTE - you could include more than jsut the filename in a dictionary, 
    #  such as full name of the river or gaging site, units, etc. that would
    #  be used later in the program, like when plotting the data.
    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    
    # define blank dictionaries (these will use the same keys as fileName)
    DataDF = {}
    MissingValues = {}
    WYDataDF = {}
    MoDataDF = {}
    AnnualAverages = {}
    MonthlyAverages = {}
    
    # process input datasets
    for file in fileName.keys():
        
        print( "\n", "="*50, "\n  Working on {} \n".format(file), "="*50, "\n" )
        
        DataDF[file], MissingValues[file] = ReadData(fileName[file])
        print( "-"*50, "\n\nRaw data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # clip to consistent period
        DataDF[file], MissingValues[file] = ClipData( DataDF[file], '1969-10-01', '2019-09-30' )
        print( "-"*50, "\n\nSelected period data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # calculate descriptive statistics for each water year
        WYDataDF[file] = GetAnnualStatistics(DataDF[file])
        
        # calcualte the annual average for each stistic or metric
        AnnualAverages[file] = GetAnnualAverages(WYDataDF[file])
        
        print("-"*50, "\n\nSummary of water year metrics...\n\n", WYDataDF[file].describe(), "\n\nAnnual water year averages...\n\n", AnnualAverages[file])

        # calculate descriptive statistics for each month
        MoDataDF[file] = GetMonthlyStatistics(DataDF[file])

        # calculate the annual averages for each statistics on a monthly basis
        MonthlyAverages[file] = GetMonthlyAverages(MoDataDF[file])
        
        print("-"*50, "\n\nSummary of monthly metrics...\n\n", MoDataDF[file].describe(), "\n\nAnnual Monthly Averages...\n\n", MonthlyAverages[file])
        
    # output 4 files: - after all other functions performed to import (&etc.) but inside final 'if'
    
        # add col for station name based on the keys : 
        MonthlyAverages[file]['Station'] = file
        # add col for station name: 
        AnnualAverages[file]['Station'] = file

    # MONTHLY
    # stack elements in dictionary to create one dataframe
    mo_stack = pd.concat([MonthlyAverages['Wildcat'],MonthlyAverages['Tippe']],axis=0)
    # 1. csv
    mo_stack.to_csv('Average_Monthly_Metrics.csv',sep=',')
    # 1. tab
    mo_stack.to_csv('Average_Monthly_Metrics.txt',sep='\t')
        
    # ANNUALLY
    # stack elements in dictionary to create one dataframe
    yr_stack = pd.concat([AnnualAverages['Wildcat'],AnnualAverages['Tippe']],axis=1)
    # 1. csv
    yr_stack.to_csv('Average_Annual_Metrics.csv',sep=',')
    # 1. tab
    yr_stack.to_csv('Average_Annual_Metrics.txt',sep='\t')





        