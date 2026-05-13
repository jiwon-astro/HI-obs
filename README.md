# HI-obs
Python code for galactic HI line observation with SDR-based radio telescope
This repository contains Python-based tools and workflows for radio astronomy observations using a portable **RTL-SDR** setup. 
The project was specifically refined for the **Radio Astronomy** course at KAIST.

## 1. Overview
The project enables users to acquire, process, and analyze radio signals, with a focus on the 21cm neutral hydrogen (HI) line. It provides a full pipeline from data acquisition to calibration, including antenna temperature and LSR (Local Standard of Rest) frame conversion.

### Key Features
- **SDR Data Acquisition**: Direct interface with RTL-SDR dongles for IQ sampling and power spectrum calculation.
- **Coordinate Conversion**: Real-time conversion between Equatorial (RA/Dec), Horizontal (Alt/Az), and Galactic (l/b) coordinate systems using `astropy`.
- **LSR Velocity Correction**: Calculation of radial velocity corrections to account for the Earth's and Sun's motion relative to the Local Standard of Rest.
- **Flux Calibration**: Implementation of the Y-factor method using ground (ambient) and sky observations to determine antenna temperature ($T_A$).
- **Data Management**: Automated logging of observation metadata and structured storage of spectral data in CSV format.
- **Sky-map and Footprint Visualization**: Plot observed beams on an Alt/Az sky map with Galactic coordinate grids, bright star / radio-source catalog overlays, and Galactic-plane footprints. (*Credit: Hakjin Lee (KAIST)*)


## 2. Repository Structure
- `radio/`
    - `constants.py`: Defines observatory location, sampling rates, and SDR configurations.
    - `sdr.py`: Contains the `Exposure` class to manage observation runs and hardware interaction.
     - `analysis.py`: Data reduction (pre-processing/calibration)
    - `utils.py`: Utility functions for time conversion, coordinate transforms, power spectral density (PSD) calculation, LSR corrections, and visualization functions.
    - `io.py`: Handles saving/loading of spectra and observation logs.
    - `config.py`: Manages directory paths for data storage and logging.
- `data/`
    - `catalog/`: Optional catalog tables used by the visualization helpers (default: *Hiparcos* (https://cdsarc.cds.unistra.fr/viz-bin/cat/I/239), *3CR catalog* (https://cdsarc.cds.unistra.fr/viz-bin/cat/VIII/1A))


## 3. Installation
### Pre-requisites:
- Python 3.8 or higher is recommended.
- RTL-SDR Hardware: You need a compatible RTL-SDR USB dongle.
- USB Driver - **Zadig** (https://zadig.akeo.ie/)
  - Select Options -> List All Devices.
  - Select `Bulk-In, Interface (Interface 0)`.
  - Click `Replace Driver` with the WinUSB driver.

The following Python libraries are required:
- `numpy`
- `scipy`
- `astropy`
- `pyrtlsdr`
- `matplotlib`
- `tqdm`

### Installation
You can install the `HI-obs` package by

    $ git clone https://github.com/jiwon-astro/HI-obs.git
    $ cd HI-obs
    $ pip install -e .


## 4. Usage
For guidance on using the package, refer to the example code in `RadioLab.ipynb`.

1. **Setup**: Connect your RTL-SDR device.
2. **Configuration**: Set your root directory and observatory coordinates in `constants.py` or through the `set_root_dir()` function.
3. **Observation**: Use the `Exposure` class in `sdr.py` to start taking data:
    1. *idx*: SDR channel (for multi-channel SDRs, default = 0)
   2. *n_obs*: number of frames to observe (each frame acquires $N_{\rm sample}=2^{18}$ IQ samples, and the series of frames are converted to power spectrum via FFT and accumulated to $N_{\rm FFT}=2^{10}$ frequency channels)
    3. *exposure_type*: `gnd` or `sky`
        - `gnd`: mode for observing the ambient frame (e.g., radio absorbers, ground..)
        - `sky`: mode for observing the specific direction in the sky. It requires pointing information, including the elevation/azimuth angle in degrees (Caution: Elevation should be strictly below 90 degrees.). <br> <br>

   ```python
   from radio.sdr import Exposure
   # Observe the sky at a specific target
   obs = Exposure(idx = 0, n_obs=10, exposure_type='sky')
   obs.run()
   ```
4. **Visualizing observed fields**: After loading an observation log, use `plot_skymap()` to inspect where previous beams fall on the current sky, or `plot_footprints()` to see the accumulated Galactic-coordinate coverage.

   ```python
   from astropy.time import Time
   from radio.utils import plot_skymap, plot_footprints

   # Current Alt/Az sky map
   curr_time = Time.now()
   fig, ax = plot_skymap(log, curr_time, timezone="KST")

   # Galactic footprints of the all-sky observations 
   fig, ax = plot_footprints(log, show_idx=True)
   ```
   <p align="center">
   <img width="70%" alt="Image" src="https://github.com/user-attachments/assets/9b62a9db-46ff-4e71-addc-c8a3bfaec3bf" />
   </p>
5. **Calibration** By acquiring both source and ambient spectra, the observed quantities can be calibrated into physical units.
    <p align="center">
   <img width="100% alt="Image" src="https://github.com/user-attachments/assets/c464780f-7b3b-4565-9ab5-c797aa933fbc" />
   </p>

    1. `calibrate_antenna_temperature` can calibrate the SDR raw power spectrum to antenna temperature with the Y-factor method. `enable_rejection` option allows automatic detection of RFI peaks and produces an RFI-rejected spectrum.
    2. `LSR_correction` calculates the barycentric correction and solar peculiar motion with respect to LSR at a given observation time and Galactic coordinates $(l, b)$.
    3. `reduce_spectrum` combines both preprocessing functions sequentially, yielding the `Observation` instance. <br> <br>

    ```python
    from radio.analysis import calibrate_antenna_temperature, reduce_spectrum
    from radio.utils import LSR_correction

    T_amb = 290 # Ambient temeperature [K]
    T_sky = 10  # Sky background temperature [K]
    
    # Antenna temperature calibration    
    Y, T_sys, T_ant = calibrate_antenna_temperature(freq, P_src, P_amb, T_amb=T_amb, T_sky=T_sky,
                                                    enable_rejection=True)

    # LSR velocity correction
    bary_corr, peculiar_corr = LSR_correction(l =src_hdr.l, b =src_hdr.b,
                                              obstime = src_hdr.obstime)
    
    # integrated pre-processing (Antenna temperature + LSR correction)
    summary = reduce_spectrum(src_hdr, freq, P_src, P_amb)
    ```

## 5. Credit
- Revision: **Jiwon Jang (KAIST)**, Hakjin Lee (KAIST)
- Original Development: TAs of the 2024/2025 SNU Natural Science Camp (Donghwan Hyeon, Jiwon Jang, Wooseok Kang, Chanjin Lee, Wonhyeong Lee).
