import numpy as np
from rtlsdr import RtlSdr
from tqdm.notebook import tqdm
from astropy.stats import sigma_clipped_stats

from .constants import N_SIG, N_DAQ, N_FFT, fs_def, fc_def, gain_def, observatory
from .utils import isotime, alt_az_to_ra_dec, alt_az_to_l_b
from .io import save_spectrum, write_obs_log

try:
    from kraken.gpu import calc_psd
except ModuleNotFoundError:
    from .utils import calc_psd

# simple SDR acquisition
def expose_sdr(n_samples, sample_rate=fs_def, center_freq=fc_def, gain=gain_def,
               ch=0):
    sdr = RtlSdr(device_index=ch)
    sdr.device_index=ch
    sdr.sample_rate = sample_rate  # equals bandwidth (Hz) in complex (IQ) sampling (Nyquist-Shannon)
    sdr.center_freq = center_freq  # center frequency (Hz)
    sdr.freq_correction = 1  # no need to correct
    sdr.gain = gain  # find the highest value before saturation

    samples = sdr.read_samples(n_samples)
    sdr.close()
    return samples

# Exposure module
class Exposure:
    """Exposures whose pointing and obstime can be considered the same"""
    def __init__(self, idx=0, n_obs=10, exposure_type = None, n_fft = N_FFT,
                center_freq = fc_def, sample_rate = fs_def, gain = gain_def,
                observatory = observatory, offset_correction = True):
        self.device_idx = idx # device index
        self.n_obs = n_obs # number of iterations 
        self.n_fft = n_fft # number of the spectral channels (N_DAQ/fn_fft frame averaged)
        self.exposure_type = exposure_type # e.g., sky, gnd
        
        self.sample_rate = sample_rate  # equals bandwidth (Hz) in complex (IQ) sampling (Nyquist-Shannon)
        self.center_freq = center_freq  # center frequency (Hz)
        self.freq_correction = 1  # no need to correct
        self.gain = gain  # find the highest value before saturation

        self.offset_correction = offset_correction # substract DC component for calculating the PSD
        
        self.observatory = observatory
        self.time = isotime() # current time
        
        print(f"Exposure = {exposure_type}")
        print("------------------------------------------------------------")
        # get Alt/Az position
        if exposure_type == "gnd":
            self.alt = None; self.az = None
            self.l = None; self.b = None
            print(f"Data will be saved at [{self.time}]")
            
        else:
            alt = float(input("Altitude [deg] = "))
            az  = float(input("Azimuthal angle [deg] = "))
            print(alt, az, type(alt))
            self.alt = alt; self.az = az

            # convert to (ra, dec) & (l, b)
            self.ra, self.dec = alt_az_to_ra_dec(self.alt, self.az, self.time)
            print(self.ra, self.dec, self.time)
            self.l, self.b    = alt_az_to_l_b(self.alt, self.az, self.time)
            print(self.l, self.b, self.time)
            print(f"Data will be saved at [{self.time}] with l={self.l:.0f}, b={self.b:.0f}")
            
        # create stoarge
        self.freq = None
        self.powers = np.empty((self.n_obs, self.n_fft))
        self.power_stacked = None

    def __repr__(self):
        return f"Exposure object with n_obs={self.n_obs}, l={self.l}, b={self.b}, time={self.time}"
    
    def config(self, sample_rate = 2.5e6, center_freq = 1400e6, gain = 50):
        sdr = self.sdr
        # configuration
        sdr.sample_rate = sample_rate
        sdr.center_freq = center_freq
        # SDR gain
        #sdr.set_agc_mode(False)  # RTL2832 digital AGC OFF (preventing the jump in power level)
        sdr.gain = gain # manual gain 
        # frequency correction
        sdr.freq_correction = 1  # no need to correct

        # initialization
        sdr.reset_buffer()
        _ = sdr.read_samples(N_DAQ) # discard initial frame (possibly transient signals)
    
    def run(self):
        self.sdr = RtlSdr(device_index=self.device_idx)
        self.config(sample_rate= self.sample_rate,
                    center_freq= self.center_freq,
                    gain= self.gain)
        try:
            for i in tqdm(range(self.n_obs), desc = 'Iteration'):
                samples = self._expose(
                        n_samples= N_DAQ,
                        save_raw=False,
                )
                freq, power = self._get_spectrum(i, samples) # frequency in MHz
        finally: # except?
            n_obs_act = i + 1
            self.sdr.close()
            
        t_exp = n_obs_act*N_DAQ/self.sample_rate # integration time
        _, power_median, _ = sigma_clipped_stats(self.powers, axis = 0, sigma = N_SIG) # frame-averaged power spectrum
        self.power_stacked = power_median
        
        # save averaged spectrum
        data_path = save_spectrum(self.freq, self.power_stacked, time=self.time, x=self.l, y=self.b, 
                                  suffix=f"{t_exp:.1f}s_"+self.exposure_type)
        # record the log
        log_path = write_obs_log(
            time=self.time,
            exposure_type=self.exposure_type,
            device_idx=self.device_idx,
            observatory = self.observatory,
            alt=getattr(self, "alt", None),
            az=getattr(self, "az", None),
            l=getattr(self, "l", None),
            b=getattr(self, "b", None),
            center_freq=float(self.center_freq/1e6),
            sample_rate=float(self.sample_rate/1e6),
            gain=float(self.gain),
            n_obs_act=int(n_obs_act),
            t_exp=float(t_exp),
            data_filename=data_path.name
        )
        print(f"Exposure finished, {n_obs_act} frame accumulated (t_exp = {t_exp:.2f}s)")
        print("------------------------------------------------------------")
        
    def _expose(self, n_samples, save_raw = False):
        """unload raw time-series data from memory immediately, and only keep the power spectrum"""
        sdr = self.sdr  
        if n_samples > N_DAQ:
            print(f'Requested the number of samples exceeded maximum size of buffer')
            n_samples = N_DAQ
        samples = sdr.read_samples(n_samples) 
        """
        if save_raw:
            # fname = unique_filename(DATA_DIR / f"{self.time}_{self.l}_{self.b}_raw.npy", always_add_counter=True)
            time = clean_isot(self.time)
            fname = unique_filename(DATA_DIR / f"{time}_{self.l}_{self.b}_raw.npy", always_add_counter=True)
            np.save(fname, samples)
        """
        return samples

    def _get_spectrum(self, i, samples):
        if self.offset_correction: samples -= np.mean(samples)
        freq, power = calc_psd(samples, fs = self.sample_rate, fc = self.center_freq,
                               N_fft= self.n_fft, overlap=0, window_func=np.hamming)
        if self.freq is None: self.freq = freq
        #assert np.array_equal(self.freq, freq)  # ensure the same frequency grid is used for all exposures
        self.powers[i] = power
        return freq, power
