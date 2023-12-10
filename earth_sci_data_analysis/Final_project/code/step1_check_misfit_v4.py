"""
Homework version

Main variables and their explanations
phasepicks: np.array((npicks, 4))
4 columns are: station x, station y, phasetype (1 for P, 2 for S), traveltime(demean)

mean_arrival_time

Task:
1. find a function that can convert latitude, longitude to x,y in km
2.
"""

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
    # Convert latitude and longitude to UTM coordinates (eastings, northings, zone number, zone letter)
    clat, clon = 40.5, -124
    utm_centers = utm.from_latlon(clat, clon)
    easting_center, northing_center = utm_centers[0], utm_centers[1]
    utm_coords = utm.from_latlon(latitude, longitude)
    easting, northing = utm_coords[0], utm_coords[1]
    easting, northing = (easting - easting_center) / 1e3, (northing - northing_center) / 1e3
    return easting, northing

def read_phase_picks(filename_picks, folder_station):
    """
    :param filename_picks:
    :param filename_station:
    :return:
    # see the requirement for phasepicks
    # return phasepicks, mean_arrival_time
    phasepicks: np.array((npicks, 4))
    4 columns are: station x, station y, phasetype (1 for P, 2 for S), traveltime(demean)

    create a sta_loc dictionary, with station x/y position
    e.g. sta_loc={'ABC':[10,10]}
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

def get_traveltime_points_value(filename):
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
    # pass

def get_misfit(array1, array2):
    misfit = np.sum(np.abs(array1-array2))
    return misfit
    # pass

def get_mean_arrival(arrival_time):
    n = len(arrival_time)
    min_arrival_time = np.min(arrival_time)
    dt = np.zeros((n, ))
    for i in range(0, n):
        dt[i] = arrival_time[i] - min_arrival_time
    return min_arrival_time + np.mean(dt)


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

################ main code ################

######### Part 1. Read phase picks #########
folder_station = '/Users/mdaislam/Library/CloudStorage/OneDrive-IndianaUniversity/Research/FocalMechanism/OutputData/nc73201181'
filename_picks = f'{folder_station}/Markers/P_S_picks_v2.txt'

phasepicks, mean_arrival_time = read_phase_picks(filename_picks, folder_station)
npick = phasepicks.shape[0]

######### Part 2. Points and value for P/S traveltime interpolation #########
P_points, P_value = get_traveltime_points_value('tt_MTJ_comploc_p')
S_points, S_value = get_traveltime_points_value('tt_MTJ_comploc_s')

######### define grids #########
lat_range = np.arange(39.5, 41.5, 0.02)
lon_range = np.arange(-124.5, -123.0, 0.02)
dep_range = np.arange(0, 40, 0.5)
nx, ny, nz = len(lon_range), len(lat_range), len(dep_range)
elat, elon, edep, t_origin = 40.274, -124.300, 9.4, UTCDateTime('2019-06-23T03:53:02')


grids_lat, grids_lon, grids_z = np.zeros((nx*ny*nz, )), np.zeros((nx*ny*nz, )), np.zeros((nx*ny*nz, ))
for grid_index in range(0, nx*ny*nz):
    lon_index, lat_index, dep_index = np.unravel_index(grid_index, (nx, ny, nz))
    grids_lon[grid_index] = lon_range[lon_index]
    grids_lat[grid_index] = lat_range[lat_index]
    grids_z[grid_index] = dep_range[dep_index]
grid_x, grid_y = convert_lat_lon_to_xy(grids_lat, grids_lon)

######### loops #########
distance = np.zeros((nx*ny*nz, npick))
traveltime = np.zeros((nx*ny*nz, npick))
mean_traveltime = np.zeros((nx*ny*nz, ))
for ipick in range(0, npick):
    distance[:, ipick] = np.sqrt((phasepicks[ipick, 0] - grid_x)**2 +
                                 (phasepicks[ipick, 1] - grid_y)**2)
    xi = np.array([distance[:, ipick], grids_z]).T
    if phasepicks[ipick, 2] == 1:
        traveltime[:, ipick] = griddata(P_points, P_value, xi, method='linear')
    else:
        traveltime[:, ipick] = griddata(S_points, S_value, xi, method='linear')

# Remove mean for traveltime array
traveltime_demean = np.zeros((nx*ny*nz, npick))
mean_traveltime = np.zeros((nx*ny*nz, ))
misfit = np.zeros((nx*ny*nz, ))
for grid_index in range(0, nx*ny*nz):
    traveltime_demean[grid_index, :] = traveltime[grid_index, :] - np.mean(traveltime[grid_index, :])
    misfit[grid_index] = get_misfit(traveltime_demean[grid_index, :], phasepicks[:,3])
    mean_traveltime[grid_index] = np.mean(traveltime[grid_index, :])

min_index = np.nanargmin(misfit)
min_index_x, min_index_y, min_index_z = np.unravel_index(min_index, (nx, ny, nz))
best_lon, best_lat, best_dep = lon_range[min_index_x], lat_range[min_index_y], dep_range[min_index_z]
origin_time = mean_arrival_time - mean_traveltime[min_index]
print('Best location and orign time estimates')
print('lat:{:.2f}, lon:{:.2f}, dep:{:.2f}'.format(best_lat, best_lon, best_dep))
print('Origin time:{}'.format(origin_time))




################### plot misfit ################### 
if not os.path.exists('png'):
    os.mkdir('png')
slat, slon = get_sta_lat_lon(filename_picks, folder_station)
misfit_matrix = np.reshape(misfit, (nx, ny, nz))
vmin, vmax = np.nanmin(misfit), np.nanmax(misfit)
vmax = (vmax - vmin) * 0.1 + vmin
cmap = 'rainbow'

fig = plt.figure(figsize=(14, 4.5))
ax = fig.add_subplot(1, 3, 1)
i = min_index_x
ax.imshow(misfit_matrix[i, :, :].T, vmin=vmin, vmax=vmax, origin='upper', cmap=cmap,
          extent=(lat_range[0], lat_range[-1], dep_range[-1], dep_range[0]), aspect='auto')
ax.plot(grids_lat[min_index], grids_z[min_index], 'o', color='w')
ax.plot(elat, edep, 'o', color='r')
ax.set_xlabel('Lat')
ax.set_ylabel('Dep')
ax = fig.add_subplot(1, 3, 2)
i = min_index_y
ax.imshow(misfit_matrix[:, i, :].T, vmin=vmin, vmax=vmax, origin='upper', cmap=cmap,
          extent=(lon_range[0], lon_range[-1], dep_range[-1], dep_range[0]), aspect='auto')
ax.plot(grids_lon[min_index], grids_z[min_index], 'o', color='w')
ax.plot(elon, edep, 'o', color='r')
ax.set_xlabel('Lon')
ax.set_ylabel('Dep')
ax = fig.add_subplot(1, 3, 3)
i = min_index_z
im = ax.imshow(misfit_matrix[:, :, i].T, vmin=vmin, vmax=vmax, origin='lower', cmap=cmap,
          extent=(lon_range[0], lon_range[-1], lat_range[0], lat_range[-1]), aspect='auto')
ax.plot(grids_lon[min_index], grids_lat[min_index], 'o', color='w')
ax.plot(elon, elat, 'o', color='r')
ax.plot(slon, slat, '^', color='w')
plt.colorbar(im)
ax.set_xlabel('Lon')
ax.set_ylabel('Lat')
plt.tight_layout()
fig.savefig('png/Misfit.png')
plt.show()


################### plot waveform and picks ################### 
phaselist = get_phasepicks_list(filename_picks)
stanames, distances = get_sta_dist(filename_picks, elat, elon)
index = np.argsort(distances)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(1,1,1)
for i in range(0, len(stanames)):
    j = index[i]
    sta = stanames[j]        
    mseed = glob.glob(f'{folder_station}/*{sta}*.mseed')
    if len(mseed) != 1:
        print('mseed problem {}'.format(sta))
#         pdb.set_trace()
    st = read(mseed[0])
    chans = []
    for k in range(0, len(st)):
        chans.append(st[k].stats.channel)
    Z_channel = -1
    for k in range(0, len(chans)):
        if chans[k][-1] == 'Z':
            Z_channel = k
            break
    if Z_channel < 0:
        Z_channel = 0        
    tr = st[Z_channel]
    tr.detrend('demean')
    tr.filter('bandpass', freqmin=2, freqmax=20)
    tr.taper(0.05)
    tr.data = tr.data/np.max(tr.data)
    print(tr.stats.npts)
    t = (tr.stats.starttime-t_origin) + np.arange(0, tr.stats.npts)/tr.stats.sampling_rate
    ax.plot(t, tr.data+i, lw=1)
    if '{}-P'.format(sta) in phaselist:
        dt = phaselist['{}-P'.format(sta)]-t_origin
        ax.plot([dt, dt], [i-0.8, i+0.8],'k')
    if '{}-S'.format(sta) in phaselist:
        dt = phaselist['{}-S'.format(sta)]-t_origin
        ax.plot([dt, dt], [i-0.8, i+0.8],'k')
    ax.text(0, i, sta)
ax.set_xlim([-5, 60])
plt.tight_layout()
fig.savefig('png/picks.png')
plt.show()




################### plot polarity ################### 
polarities, stanames = get_polarity(filename_picks, folder_station)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(grids_lon[min_index], grids_lat[min_index], 'o', color='w')
ax.plot(elon, elat, 'o', color='r')
for i in range(0, len(stanames)):
    if polarities[i, 0] == 1:
        ax.plot(polarities[i, 2], polarities[i, 1], '^', color='k', mec='k')
    elif polarities[i, 0] == -1:
        ax.plot(polarities[i, 2], polarities[i, 1], '^', color='w', mec='k')
ax.set_xlabel('Lon')
ax.set_ylabel('Lat')
plt.tight_layout()
fig.savefig('png/Polarity.png')
plt.show()