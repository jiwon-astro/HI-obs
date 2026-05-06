import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, Angle, LSR

from .constants import observatory

"""
Time conversion
"""
def isotime(time: str = None) -> str:
    """UTC time in ISO format"""
    time = time or Time(Time.now(), format="iso", scale="utc", location=observatory)
    if isinstance(time, str):  # e.g. "2025-08-03 12:00:00"
        fmt = "isot" if "T" in time else "iso"
        time = Time(time, format=fmt, scale="utc", location=observatory)
    return time.isot.split(".")[0]  # trim milliseconds

def obs_date_from_isot(time: str) -> str:
    # "2025-08-03T12:00:00" -> "2025-08-03"
    return time.split("T")[0]

# utility function (":" in filename isn't incompatible for windows environment)
def clean_isot(time: str) -> str:
    """for windows"""
    return time.replace(":", "-")

def restore_isot(s: str) -> str:
    date, time = s.split("T")
    time = time.replace("-", ":")
    t = f"{date}T{time}"
    return t

"""
Coordinate transform
"""
def ra_dec_to_l_b(ra: str | float, dec: float, time: str = None):
    """
    ra: str (hms) or float (deg)
    dec: float
    time: str or Time
    """
    time = time or isotime()

    angle_ra = Angle(ra, unit=u.hourangle if isinstance(ra, str) else u.deg)
    angle_dec = Angle(dec, unit=u.deg)

    coord_icrs = SkyCoord(ra=angle_ra, dec=angle_dec, frame="icrs", obstime=time, location=observatory)
    coord_gal = coord_icrs.galactic

    return coord_gal.l.value, coord_gal.b.value

def alt_az_to_ra_dec(alt, az, time: str = None):
    time = time or isotime()
    
    coord_aa = SkyCoord(alt=alt * u.deg, az=az * u.deg, frame="altaz", obstime=time, location=observatory)
    coord_eq = coord_aa.icrs
    print(f"(ra, dec) = {coord_eq.to_string('hmsdms')}")
    return coord_eq.ra.value, coord_eq.dec.value

def alt_az_to_l_b(alt, az, time: str = None):
    time = time or isotime()

    coord_aa = SkyCoord(alt=alt * u.deg, az=az * u.deg, frame="altaz", obstime=time, location=observatory)
    coord_gal = coord_aa.galactic  # transform to Galactic
    print(f"(l, b) = {coord_gal.to_string('dms')}")
    return coord_gal.l.value, coord_gal.b.value

def l_b_to_alt_az(l, b, time: str = None):
    time = time or isotime()

    coord_gal = SkyCoord(l=l * u.deg, b=b * u.deg, frame="galactic", obstime=time, location=observatory)
    coord_aa = coord_gal.transform_to("altaz")
    return coord_aa.alt.value, coord_aa.az.value

"""
LSR velocity conversion
"""
def LSR_correction(l, b, obstime, observatory = observatory):
    coord = SkyCoord(l=l * u.deg, b=b * u.deg, frame="galactic", obstime=obstime, location=observatory)
    # Barycentric correction
    bary_corr = coord.radial_velocity_correction("barycentric").to(u.km / u.second).value
    # Peculiar velocity of sun
    V_pec = LSR().v_bary.d_xyz.value
    l_rad, b_rad = np.deg2rad(l), np.deg2rad(b)
    pointing = np.array([np.cos(l_rad)*np.cos(b_rad),
                         np.sin(l_rad)*np.cos(b_rad),
                         np.sin(b_rad)])
    peculiar_corr = np.dot(V_pec, pointing)
    return bary_corr, peculiar_corr

"""
Power spectrum calculation
"""
# -> replace to GPU version!
def calc_psd(sample, fs, fc=0, N_fft=1024, overlap=0, window_func=np.hamming,
            offset_correction = True):

    # Welch method - PSD averaging
    # overlap: between windows fraction
    # N_fft: length of each segments (higher value leads higher spectral resolution)

    win = window_func(N_fft)  # window function
    U = np.mean(win**2)  # power of window function

    step = int(N_fft * (1 - overlap))  # window step
    freq = np.fft.fftshift(np.fft.fftfreq(N_fft, d=1/fs)) + fc  # frequency grid
    if offset_correction: sample -= np.mean(sample)
        
    segments = []
    for i in range(0, len(sample) - N_fft + 1, step):
        seg = sample[i : i + N_fft]
        segments.append(seg * win)  # convolution in time domain
    segments = np.array(segments)

    # FFT for each segments
    F_seg = np.fft.fftshift(np.fft.fft(segments, axis=1), axes=1) / N_fft
    psd_seg = np.abs(F_seg) ** 2 / U
    psd = np.mean(psd_seg, axis=0)

    return freq, psd

def plot_psd(freq, P, ax=None, **kwargs):
    # Power in linear scale
    P_db = 10*np.log10(P)
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(8,4))
    ax.plot(freq, P_db, **kwargs)
    ax.set_ylabel("Power [dB/MHz]")
    ax.set_xlabel("Frequency [MHz]")
    ax.set_xlim(freq[0], freq[-1])
    return ax