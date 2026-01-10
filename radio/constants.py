from astropy.coordinates import Angle, EarthLocation
    
# Location (NYSC, Goheung)
obs_lat_s, obs_lon_s = "34d31m58s", "127d28m06s"
obs_height = 160 #m
obs_lat, obs_lon = Angle(obs_lat_s), Angle(obs_lon_s)
observatory = EarthLocation.from_geodetic(lat=obs_lat, lon=obs_lon, height=obs_height)

# sigma clipping
N_SIG = 3. 

# SDR acquisition setting
N_DAQ = 2**18
N_FFT = 2**10 
fc_def   = 1420e6 # center frequency
fs_def   = 3e6    # sampling rate
gain_def = 50