import numpy as np
from dataclasses import dataclass
from astropy.stats import sigma_clip
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from .constants import f0
from .utils import radial_vel, LSR_correction

@dataclass
class Observation:
    fname: str
    obstime: str
    az: float
    alt: float
    l: float
    b: float
    fc: float
    bandwidth: float  
    V_bary: float
    V_pec: float
    Y: float
    T_sys: float
    freq: np.ndarray
    P_src: np.ndarray
    P_amb: np.ndarray
    T_ant: np.ndarray
    
    def V_r(self, f0=f0):
        """Topocentric radial velocity [km/s]."""
        return radial_vel(self.freq, f0)

    def V_lsr(self, f0=f0):
        """LSR velocity [km/s]."""
        return self.V_r(f0) + self.V_bary + self.V_pec

    @property
    def label(self):
        return rf"($\ell,\;b$) = ({self.l:.0f}, {self.b:.0f})"

    @property
    def color_value(self, scale=0.3):
        return 0.5 + scale * (self.l - 180) / 360
    
# =======================
# Preprocessing
# =======================
def spike_rejection(freq, powers, width=2):
    """
    Removing the artificial spikes in the spectrum.
    """
    powers_red = powers.copy()
    # RFI spikes
    pidx = find_peaks(powers, prominence=(5, None), width=(None,8))[0]
    # masking
    mask = np.ones_like(freq, dtype=bool)
    for w in range(-width, width+1): mask[pidx+w] = 0
    fpowers = interp1d(freq[mask], powers_red[mask], 
                       kind=1, fill_value='extrapolate')
    return fpowers(freq)

def calibrate_antenna_temperature(freq, P_src, P_amb, T_amb=290, T_sky=10, 
                                  enable_rejection=True, width=1):
    """
    Calibrating the measured power to antenna temperature through Y-factor method.
    The calibration procedure requires:
    (1) source spectrum (subjected to calibration)
    (2) ambient spectrum (spectrum acquired by pointing towards to radio absorbers under ambient temperature)

    Return:
    Y: Y-factor (=P_hot/P_cold)
    T_sys: System noise temperature
    T_src: Antenna temperature
    """
    # Y-factor
    Y = np.median(P_amb / P_src)
    P_sky = P_amb / Y
    T_sys = (T_amb - Y*T_sky)/(Y-1) # system temperature
    # antenna temperature
    T_ant = (P_src - P_sky) / (P_amb - P_sky) * T_amb
    if enable_rejection:
        T_ant = spike_rejection(freq, T_ant, width=width)
    return Y, T_sys, T_ant

def reduce_spectrum(src_hdr, freq, P_src, P_amb, **kwargs):
    # antenna temperature
    Y, T_sys, T_ant = calibrate_antenna_temperature(freq, P_src, P_amb, **kwargs)
    # LSR velocity correction
    bary_corr, peculiar_corr = LSR_correction(l =src_hdr.l, b =src_hdr.b,
                                              obstime = src_hdr.obstime)
    return Observation(fname=str(src_hdr.fname), obstime=str(src_hdr.obstime), fc=float(src_hdr.fc),
                       bandwidth=float(src_hdr.fs), az=float(src_hdr.az), alt=float(src_hdr.alt),
                       l=float(src_hdr.l), b=float(src_hdr.b), V_bary=float(bary_corr), V_pec=float(peculiar_corr),
                       Y=float(Y), T_sys=float(T_sys), freq=freq, P_src=P_src, P_amb=P_amb, T_ant=T_ant)