import astropy.constants as const
from astropy.coordinates import Angle, EarthLocation

# astrophysical constants
f0 = 1420.40575177 # rest frame frequency of HI [MHz]
c  = 299792.458    # lightspeed [km/s]

# Location (NYSC, Goheung)
obs_lat_s, obs_lon_s = "36d22m10s", "127d21m51s"
obs_height = 50 # [m]
obs_lat, obs_lon = Angle(obs_lat_s), Angle(obs_lon_s)
observatory = EarthLocation.from_geodetic(lat=obs_lat, lon=obs_lon, height=obs_height)

# sigma clipping
N_SIG = 3. 

# SDR acquisition setting
N_DAQ = 2**18
N_FFT = 2**10 
fc_def   = 1420.2e6 # center frequency [Hz]
fs_def   = 3e6      # sampling rate [Hz] (maximum 3MHz)
gain_def = 50