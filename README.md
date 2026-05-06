# HI-obs
Python code for galactic HI line observation with SDR-based radio telescope
This repository contains Python-based tools and workflows for radio astronomy observations using a portable **RTL-SDR** setup. 
The project was specifically refined for the **Radio Astronomy** course at KAIST.

## 1. Overview
The project enables users to acquire, process, and analyze radio signals, with a focus on the 21cm neutral hydrogen (HI) line. It provides a full pipeline from data acquisition to calibration, including antenna temperature and LSR (Local Standard of Rest) velocity.

### Key Features
- **SDR Data Acquisition**: Direct interface with RTL-SDR dongles for IQ sampling and power spectrum calculation.
- **Coordinate Conversion**: Real-time conversion between Equatorial (RA/Dec), Horizontal (Alt/Az), and Galactic (l/b) coordinate systems using `astropy`.
- **LSR Velocity Correction**: Calculation of radial velocity corrections to account for the Earth's and Sun's motion relative to the Local Standard of Rest.
- **Flux Calibration**: Implementation of the Y-factor method using ground (ambient) and sky observations to determine antenna temperature ($T_A$).
- **Data Management**: Automated logging of observation metadata and structured storage of spectral data in CSV format.

## 2. Repository Structure
- `radio/`
    - `constants.py`: Defines observatory location, default sampling rates, and SDR configurations.
    - `sdr.py`: Contains the `Exposure` class to manage observation runs and hardware interaction.
    - `utils.py`: Utility functions for time conversion, coordinate transforms, PSD (Power Spectral Density) calculation, and LSR corrections.
    - `io.py`: Handles saving/loading of spectra and observation logs.
    - `config.py`: Manages directory paths for data storage and logging.

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

## 5. Credit
- Revision: Jiwon Jang (KAIST)
- Original Development: TAs of the 2024/2025 SNU Natural Science Camp (Donghwan Hyeon, Jiwon Jang, Wooseok Kang, Chanjin Lee, Wonhyeong Lee).
