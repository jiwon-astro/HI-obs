import csv
from astropy.table import Table
from dataclasses import dataclass

from .config import get_data_dir, get_log_dir
from .utils import isotime, clean_isot, obs_date_from_isot

# Data I/O
def save_spectrum(freq, power, time=None, x=None, y=None, suffix=None, overwrite = True):
    # (x, y): (Alt, Az) or (l, b)
    if x is None and y is None:
        filename = f"{time}_{suffix}.csv" if suffix else f"{time}.csv"
    else:
        filename = f"{time}_{x:.0f}_{y:.0f}_{suffix}.csv" if suffix else f"{time}_{x:.0f}_{y:.0f}.csv"

    filename = clean_isot(filename)

    data_path = get_data_dir() / filename
    tbl = Table()
    tbl["frequency"] = freq
    tbl["power"] = power
    tbl.write(data_path, format="csv", delimiter="\t", overwrite = overwrite)
    
    return data_path

def load_data(filename):
    if filename.suffix == ".npy":
        return np.load(filename)
    return Table.read(filename, format="csv", delimiter="\t")

def load_spectra_from_list(log, exposure_type = None,
                           alt=None, az=None, l=None, b=None,
                           data_dir = None, mask = None):
    # log = astropy.Table
    data_dir = data_dir or get_data_dir()
    print(f"Data directory = {data_dir}")
    
    mask = (log['type'] == exposure_type)
    if (alt is not None) and (az is not None):
        mask &= (log['alt [deg]'] == alt) & (log['az [deg]'] == az)
    elif (l is not None) and (b is not None):
        mask &= (log['l [deg]'] == alt) & (log['b [deg]'] == az)
    
    flist = list(log[mask]['filename'].data)
    
    loaded  = {}
    missing = []
    print("------ loaded -------")
    for fname in flist:
        p = data_dir / str(fname)
        if not p.exists():
            missing.append(str(fname))
            continue
        loaded[str(fname)] = load_data(p)
        print(f"{fname}")
    
    if missing:
        print("------ missing -------")
        for fname in missing: print(f"{fname}")
    print("")
    return loaded, missing

@dataclass
class header:
    obstime: str
    ftype: str
    l: float
    b: float
    fc: float
    texp: float
        
def load_source_header(log, fname):
    tbl = log[log['filename']==fname][0]
    hdr = header(obstime=tbl['UT'],ftype=tbl['type'],
                 l=tbl['l [deg]'], b=tbl['b [deg]'], 
                 fc=tbl['fcen [MHz]'], texp=tbl['t_exp [s]'])
    return hdr
    
def write_obs_log(time: str, exposure_type: str, observatory,
                   alt: float | None, az: float | None, l: float | None, b: float | None, 
                   center_freq: float, sample_rate: float, gain: float, n_obs_act: int,
                   t_exp: float, data_filename: str):
    
    obsdate = obs_date_from_isot(time)
    obs_lat, obs_lon = observatory.lat.value, observatory.lon.value
    log_path = get_log_dir() / f'log_{obsdate}.csv'
    
    fieldnames = [
        "UT", "type", "obs_lat [deg]", "obs_lon [deg]",
        "alt [deg]", "az [deg]", "l [deg]", "b [deg]", 
        "fcen [MHz]", "fsampl [MHz]", "gain", 
        "n_obs", "t_exp [s]", "filename"
    ]

    row = {
        "UT": time,
        "type": exposure_type,
        "obs_lat [deg]": f"{obs_lat:.8f}",
        "obs_lon [deg]": f"{obs_lon:.8f}",
        "alt [deg]": "NaN" if alt is None else f"{alt:.2f}",
        "az [deg]": "NaN" if az  is None else f"{az:.2f}",
        "l [deg]": "NaN" if l   is None else f"{l:.2f}",
        "b [deg]": "NaN" if b   is None else f"{b:.2f}",
        "fcen [MHz]": f"{center_freq:.3f}",
        "fsampl [MHz]": f"{sample_rate:.3f}",
        "gain": f"{gain:.3f}",
        "n_obs": int(n_obs_act),
        "t_exp [s]": f"{t_exp:.6f}",
        "filename": data_filename,
    }

    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if write_header:
            print(f"Log file created at {log_path}")
            w.writeheader()
        w.writerow(row)

    return log_path