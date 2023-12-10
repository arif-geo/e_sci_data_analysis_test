from scipy.interpolate import griddata
import pandas as pd
from obspy import UTCDateTime, read
import numpy as np
import matplotlib.pyplot as plt
import utm
import pdb
import glob
from geopy.distance import geodesic
import os

def convert_lat_lon_to_xy(latitude, longitude):
    """
        Convert latitude and longitude to x, y
    Input:
        latitude: np.array or float
        longitude: np.array or float
    Output:
        x: np.array or int, km
        y: np.array or int, km
    """
    # Convert latitude and longitude to UTM coordinates (eastings, northings, zone number, zone letter)
    clat, clon = 40.5, -124
    utm_centers = utm.from_latlon(clat, clon)
    easting_center, northing_center = utm_centers[0], utm_centers[1]
    utm_coords = utm.from_latlon(latitude, longitude)
    easting, northing = utm_coords[0], utm_coords[1]
    easting, northing = (easting - easting_center) / 1e3, (northing - northing_center) / 1e3
    return easting, northing

#==================================================================================================

def read_phase_picks(filename_picks, folder_station):
    """
    Input:
        filename_picks: filename of the phasepicks
        filename_station: filename of the station information
    Output:
        phasepicks: np.array((npicks, 4))
            4 columns are: station x position,
                           station y position,
                           phasetype (1 for P, 2 for S),
                           demeaned arrival time
        mean_arrival_time: UTCDateTime format
            mean of all the phase pick times
    """
    sta_files = glob.glob('{}/*.txt'.format(folder_station))
    sta_loc = {}
    sta_xy = np.zeros((len(sta_files), 2))
    for i in range(0, len(sta_files)):
        df = pd.read_csv(sta_files[i], 
                         sep='|', 
                         header=0,
                         usecols=[1, 4, 5])                         
        staname, slat, slon = df['Station'][0], df['Latitude'][0], df['Longitude'][0]
        sx, sy = convert_lat_lon_to_xy(slat, slon)
        sta_loc[str(staname)] = [sx, sy]
        sta_xy[i, :] = [slon, slat]
    print('x: {} {}'.format(np.min(sta_xy[:, 0]), np.max(sta_xy[:, 0])))
    print('y: {} {}'.format(np.min(sta_xy[:, 1]), np.max(sta_xy[:, 1])))
    
    df_picks = pd.read_csv(filename_picks, sep='\s+', header=None, skiprows=1, usecols=[1, 2, 4, 8],
                     names=['date', 'time', 'sta', 'phase'],
                     dtype={'date': str, 'time': str, 'sta': str, 'phase': str})
    npick = df_picks.shape[0]
    phasepicks = np.zeros((npick, 4))
    arrival_time = []
    for i in range(0, npick):
        stastr = df_picks['sta'][i]
        staname = stastr.split('.')[1]
        sx, sy = sta_loc[staname]
        if df_picks['phase'][i] == 'P':
            phasetype = 1
        else:
            phasetype = 2
        arrival_time.append(UTCDateTime('{}T{}'.format(df_picks['date'][i], df_picks['time'][i])))
        phasepicks[i, 0:3] = [sx, sy, phasetype]

    mean_arrival_time = get_mean_arrival(arrival_time)
    for i in range(0, npick):
        phasepicks[i, 3] = arrival_time[i] - mean_arrival_time
    return phasepicks, mean_arrival_time

#==================================================================================================

def get_traveltime_points_value(filename):
    """
    Input:
        filename: filename of the P or S traveltime table
    Output:
        points and values required by the scipy.interpolate.griddata function
        points: np.array(N_grid, 2)
            1st column distance, 2nd column dep
        value: np.array(N_grid, )
            corresponding traveltime from the traveltime table
    Please check the lecture notes!!!
    """
    dep = np.arange(0, 80.1, 0.5)
    dist = np.arange(0, 150.1, 0.5)
    df = pd.read_csv(filename, skiprows=3, sep='\s+', header=None)
    matrix = df.values
    ndist, ndep = df.shape[0], df.shape[1] - 1
    points = np.zeros((ndist * ndep, 2))
    values = np.zeros((ndist * ndep,))
    count = 0
    for i in range(0, ndist):
        for j in range(0, ndep):
            points[count, :] = [dist[i], dep[j]]
            values[count] = matrix[i, j + 1]
            count += 1
    return points, values

#==================================================================================================

def get_misfit(array1, array2):
    """
    Input:
        array1: np.array or float
        array2: np.array or float
        Here, array1 is the demeaned predicted traveltime,
              array2 is the demeaned arrival time
    Output:
        misfit: np.array or float
    Calculate norm 1 or norm 2 misfit between array1 and array2

    """
    misfit = np.sum(np.abs(array1-array2))
    return misfit

#==================================================================================================

def get_mean_arrival(arrival_time):
    n = len(arrival_time)
    min_arrival_time = np.min(arrival_time)
    dt = np.zeros((n, ))
    for i in range(0, n):
        dt[i] = arrival_time[i] - min_arrival_time
    return min_arrival_time + np.mean(dt)

#==================================================================================================

def get_sta_lat_lon(filename_picks, folder_station):
    """
    For plotting
    """
    sta_files = glob.glob('{}/*.txt'.format(folder_station))
    sta_loc = {}
    sta_xy = np.zeros((len(sta_files), 2))
    for i in range(0, len(sta_files)):
        df = pd.read_csv(sta_files[i], 
                         sep='|', 
                         header=0,
                         usecols=[1, 4, 5])                         
        staname, slat, slon = df['Station'][0], df['Latitude'][0], df['Longitude'][0]
        sta_loc[str(staname)] = [slat, slon]

    df_picks = pd.read_csv(filename_picks, sep='\s+', header=None, skiprows=1, usecols=[1, 2, 4, 8],
                     names=['date', 'time', 'sta', 'phase'],
                     dtype={'date': str, 'time': str, 'sta': str, 'phase': str})
    npick = df_picks.shape[0]
    sta_position = np.zeros((npick, 2))
    for i in range(0, npick):
        stastr = df_picks['sta'][i]
        staname = stastr.split('.')[1]
        slat, slon = sta_loc[staname]
        sta_position[i, :] = [slat, slon]
    return sta_position[:,0], sta_position[:,1]

#==================================================================================================

def get_sta_dist(filename_picks, elat, elon):
    """
    For plotting
    """
    sta_files = glob.glob('{}/*.txt'.format(folder_station))
    stanames = []
    distances = []
    for i in range(0, len(sta_files)):
        df = pd.read_csv(sta_files[i], 
                         sep='|', 
                         header=0,
                         usecols=[1, 4, 5])                         
        staname, slat, slon = df['Station'][0], df['Latitude'][0], df['Longitude'][0]
        if staname in stanames:
            continue
        stanames.append(staname)
        distances.append(geodesic((elat, elon), (slat, slon)).km)
    distances = np.array(distances)
    return stanames, distances

#==================================================================================================

def get_phasepicks_list(filename):
    # [phasetime]
    df = pd.read_csv(filename, sep='\s+', header=None, skiprows=1, usecols=[1,2,4,8], \
                     names=['date','time','sta','phase'],\
                     dtype={'date':str,'time':str,'sta':str,'phase':str})
    phase_list = {}
    for i in range(0, df.shape[0]):
        dd, tt, stastr, phasetype = df['date'][i], df['time'][i], df['sta'][i], df['phase'][i]
        phasetime = UTCDateTime('{}T{}'.format(dd, tt))
        sta = stastr.split('.')[1]
        phase_list['{}-{}'.format(sta, phasetype)] = phasetime
    return phase_list

#==================================================================================================

def get_polarity(filename, folder_station):
    sta_loc = get_sta_loc(folder_station)
    df = pd.read_csv(filename, sep='\s+', header=None, skiprows=1, usecols=[4,8,9], \
                     names=['sta', 'phase', 'polarity'],\
                     dtype={'sta':str,'phase':str, 'polarity': str})
    polarities = []
    stanames = []
    for i in range(0, df.shape[0]):
        stastr, phasetype, polarity = df['sta'][i], df['phase'][i], df['polarity'][i]
        if polarity == '1' and phasetype == 'P':
            sta = stastr.split('.')[1]
            slat, slon = sta_loc[sta][0], sta_loc[sta][1]
            polarities.append([1, slat, slon])
            stanames.append(sta)
        elif polarity == '-1' and phasetype == 'P':
            sta = stastr.split('.')[1]
            slat, slon = sta_loc[sta][0], sta_loc[sta][1]
            polarities.append([-1, slat, slon])
            stanames.append(sta)
    polarities = np.array(polarities)
    return polarities, stanames

#==================================================================================================
    
def get_sta_loc(folder_station):
    sta_files = glob.glob('{}/*.txt'.format(folder_station))
    sta_loc = {}
    for i in range(0, len(sta_files)):
        df = pd.read_csv(sta_files[i], 
                         sep='|', 
                         header=0,
                         usecols=[1, 4, 5])                         
        staname, slat, slon = df['Station'][0], df['Latitude'][0], df['Longitude'][0]
        sta_loc[str(staname)] = [slat, slon]    
    return sta_loc

#==================================================================================================