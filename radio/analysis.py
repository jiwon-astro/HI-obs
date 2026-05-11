import numpy as np
from astropy.stats import sigma_clip
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from . import constants

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

def calibrate_antenna_temperature(P_src, P_amb, enable_rejection=True):
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
