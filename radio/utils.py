import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import astropy.units as u
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import AltAz, SkyCoord, Angle, LSR

from .config import get_data_dir
from .constants import observatory, c, BEAM_SIZE, DEFAULT_STAR_CAT, DEFAULT_RADIO_CAT

# =======================
# Time Conversion
# =======================
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


# =======================
# Coordinate Transform
# =======================
def eq2altaz(ra: str | float, dec: float, time: str = None):
    time = time or isotime()

    angle_ra = Angle(ra, unit=u.hourangle if isinstance(ra, str) else u.deg)
    angle_dec = Angle(dec, unit=u.deg)

    coord_eq = SkyCoord(ra=angle_ra, dec=angle_dec, frame="icrs", obstime=time, location=observatory)
    coord_aa = coord_eq.transform_to("altaz") # transform to AltAz
    return coord_aa.alt.deg, coord_aa.az.deg

def altaz2eq(alt, az, time: str = None):
    time = time or isotime()
    
    coord_aa = SkyCoord(alt=alt * u.deg, az=az * u.deg, frame="altaz", obstime=time, location=observatory)
    coord_eq = coord_aa.icrs # transform to equatorial
    print(f"(ra, dec) = {coord_eq.to_string('hmsdms')}")
    return coord_eq.ra.deg, coord_eq.dec.deg

def altaz2gal(alt, az, time: str = None):
    time = time or isotime()

    coord_aa = SkyCoord(alt=alt * u.deg, az=az * u.deg, frame="altaz", obstime=time, location=observatory)
    coord_gal = coord_aa.galactic  # transform to Galactic
    print(f"(l, b) = {coord_gal.to_string('dms')}")
    return coord_gal.l.deg, coord_gal.b.deg

def gal2altaz(l, b, time: str = None):
    time = time or isotime()

    coord_gal = SkyCoord(l=l * u.deg, b=b * u.deg, frame="galactic", obstime=time, location=observatory)
    coord_aa = coord_gal.transform_to("altaz") # transform to AltAz
    return coord_aa.alt.deg, coord_aa.az.deg

# =======================
# Cart Plane Calibration
# =======================
def cartesian_vector(alpha, beta, deg=True):
    """
    alpha_rad: azimuthal angle from x-axis (counter-clockwise direction)
    beta_rad: elevation angle from xy plane
    """
    if deg: 
        alpha, beta = np.deg2rad(alpha), np.deg2rad(beta)
    return np.array([np.cos(beta)*np.cos(alpha),
                   np.cos(beta)*np.sin(alpha),
                   np.sin(beta)])

def rotation_cart(alpha, beta, delta):
    """
    Build mobile cart base plane rotation matrix
    alpha : angle from ground x to proj_xy(x_c)
    beta  : elevation angle of x_c
    delta : elevation angle of y_c
    """
    alpha_rad, beta_rad, delta_rad = np.deg2rad([alpha, beta, delta])
    gamma_rad = alpha_rad + np.arccos(-np.tan(beta_rad)*np.tan(delta_rad)) # angle(x, proj_xy(y_c))
    gamma = np.rad2deg(gamma_rad)
    # cart plane basis
    xc = cartesian_vector(alpha, beta)
    yc = cartesian_vector(gamma, delta)
    zc = np.cross(xc, yc)
    # rotation matrix
    return np.column_stack([xc, yc, zc])


# =======================
# Sky Footprints
# =======================
def plotConvertAltAz(alt, az, projection=None, deg=True):
    """
    Project Alt/Az to polar-plot coordinates.
    """
    projection = projection if projection is not None else "equidistant"
    if deg: 
        alt, az = np.deg2rad(alt), np.deg2rad(az)
    theta = az + np.pi/2 # North : upside
    z = np.pi/2 - alt
    if projection == 'equidistant': 
        r = z/(np.pi/2)
    elif projection == 'stereographic': 
        r = np.sin(z)/(1+np.cos(z)) # Map Projection
    else: raise NotImplementedError(f"This mode is not implemented: {projection}")
    return theta, r

def _visible_mask(alt, radius):
    return (
        np.isfinite(alt)
        & np.isfinite(radius)
        & (alt >= 0)
        & (radius >= 0)
        & (radius <= 1)
    )

def galGrid(ax, altaz_frame, grid_unit=(10, 10), label=True, projection=None):
    """
    Plot Galactic Coordinates Grid
    """
    dl, db = grid_unit
    Nl = 360//dl + 1
    Nb = 180//db + 1

    # b-Grid
    for b in np.linspace(-90, 90, Nb):
        l_temp = np.linspace(0, 360, 10*Nl)
        c_gal = SkyCoord(
            l=l_temp*u.deg,
            b=b*np.ones_like(l_temp)*u.deg,
            frame='galactic'
        )
        c_altaz = c_gal.transform_to(altaz_frame)
        alt_temp, az_temp = c_altaz.alt.deg, c_altaz.az.deg
        theta_temp, r_temp = plotConvertAltAz(alt_temp, az_temp, projection=projection)

        lw = 0.9 if b % 30 == 0 else 0.4
        ax.plot(theta_temp, r_temp, linestyle='dotted', linewidth=lw, 
                color='orangered', alpha=0.4)

        if label and b % 30 == 0 and b != 0:
            visible = _visible_mask(alt_temp, r_temp)
            idx = np.flatnonzero(visible)
            if len(idx):
                k = idx[len(idx) // 2]
                ax.annotate(
                    f"b={b:.0f}°",
                    xy=(theta_temp[k], r_temp[k]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=6,
                    color='orangered',
                    alpha=0.8
                )

    # l-Grid
    for l in np.linspace(0, 360, Nl):
        b_temp = np.linspace(-90, 90, 10*Nb)
        c_gal = SkyCoord(
            l=l*np.ones_like(b_temp)*u.deg,
            b=b_temp*u.deg,
            frame='galactic'
        )
        c_altaz = c_gal.transform_to(altaz_frame)
        alt_temp, az_temp = c_altaz.alt.deg, c_altaz.az.deg
        theta_temp, r_temp = plotConvertAltAz(alt_temp, az_temp, projection=projection)

        lw = 0.9 if l % 60 == 0 else 0.4
        ax.plot(theta_temp, r_temp, linestyle='dotted', linewidth=lw, color='orangered', alpha=0.4)

        if label and l % 60 == 0:
            visible = _visible_mask(alt_temp, r_temp)
            idx = np.flatnonzero(visible)
            if len(idx):
                k = idx[len(idx) // 2]
                ax.annotate(
                    f"l={l:.0f}°",
                    xy=(theta_temp[k], r_temp[k]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=6,
                    color='orangered',
                    alpha=0.8
                )

    # Galactic Equator
    l_temp = np.linspace(0, 360, 10*Nl)
    c_gal = SkyCoord(
        l=l_temp*u.deg,
        b=np.zeros_like(l_temp)*u.deg,
        frame='galactic'
    )
    c_altaz = c_gal.transform_to(altaz_frame)
    alt_temp, az_temp = c_altaz.alt.deg, c_altaz.az.deg
    theta_temp, r_temp = plotConvertAltAz(alt_temp, az_temp, projection=projection)
    ax.plot(theta_temp, r_temp, linestyle='dashed', linewidth=0.7, color='orangered', alpha=0.7, label='Galactic Equator')

def _load_catalog(catalog):
    print(catalog)
    if catalog is None:
        return None
    if isinstance(catalog, Table):
        return catalog

    fpath = Path(catalog)
    if not fpath.exists():
        return None
    return Table.read(fpath, format="csv")

def _format_skymap_axes(ax, obstime, location=observatory, timezone="KST"):
    # set timezone
    timezone = timezone.upper()
    tz_offset = 0
    if timezone == "UT": pass
    elif timezone == "KST": tz_offset = 9
    else: raise NotImplementedError(f"unknown timezone: {timezone}")

    # local times (standard time / LST)
    local_time = obstime + tz_offset * u.hour
    lst = obstime.sidereal_time("mean", longitude=location.lon)

    # plot setting
    ax.set_ylim(0, 1)
    ax.set_title(f"{local_time.to_value('iso', subfmt='date_hm')} (LST={lst.hour:.2f} h)", 
                 va="bottom", fontsize=18)
    ax.set_yticks([])
    ax.set_xticks(
        np.arange(0, 2 * np.pi, np.pi / 4),
        labels=["W", "NW", "N", "NE", "E", "SE", "S", "SW"],
    )
    ax.invert_xaxis()
    ax.grid(visible=False)

def plot_star_cat(ax, star_cat, altaz_frame, params=None, 
                  projection=None):
    """
    Visualize bright stars brighter than the magnitude limit.
    """
    mag_lim = params if params is not None else 5.0
    star_cat = _load_catalog(star_cat)
    if star_cat is None:
        return

    ra = np.asarray(star_cat['RAJ2000'])
    dec = np.asarray(star_cat['DEJ2000'])
    mag = np.asarray(star_cat['Vmag'])

    idx = mag < mag_lim
    # Converting the catalog equatorial coordinates to current Alt/Az frame
    c_eq = SkyCoord(ra=ra[idx]*u.deg, dec=dec[idx]*u.deg, frame='icrs')
    c_altaz = c_eq.transform_to(altaz_frame) 
    alt, az = c_altaz.alt.deg, c_altaz.az.deg

    theta_temp, r_temp = plotConvertAltAz(alt, az, projection=projection)
    visible = _visible_mask(alt, r_temp)

    star_size = 3.5 ** (0.40 * (6 - mag[idx][visible])) - 0.7

    ax.scatter(theta_temp[visible], r_temp[visible], s=star_size,
               color='black', alpha=0.5, marker='.')

def plot_radio_cat(ax, radio_cat, altaz_frame, params=None,
                   projection=None):
    """
    Visualize the bright radio sources exceeding flux percentile 
    """
    label, percentile, color = params if params is not None else (None, 85, 'gold')
    radio_cat = _load_catalog(radio_cat)
    if radio_cat is None:
        return
    
    ra = np.asarray(radio_cat['ra_j2000_deg'])
    dec = np.asarray(radio_cat['dec_j2000_deg'])
    flux = np.asarray(radio_cat['flux_178mhz_jy'])

    flux_q = np.nanpercentile(flux, percentile)
    flux_lim = flux_q

    idx = flux > flux_lim
    # Converting the catalog equatorial coordinates to current Alt/Az frame
    c_eq = SkyCoord(ra=ra[idx]*u.deg, dec=dec[idx]*u.deg, frame='icrs')
    c_altaz = c_eq.transform_to(altaz_frame) 
    alt, az = c_altaz.alt.deg, c_altaz.az.deg

    theta_temp, r_temp = plotConvertAltAz(alt, az, projection=projection)
    visible = _visible_mask(alt, r_temp)

    object_size = 20

    ax.scatter(theta_temp[visible], r_temp[visible], s=object_size,
               color=color, alpha=0.5, marker='o', label=label)

def _beam_mask(lon_grid, lat_grid, lon0, lat0, radius, deg=True):
    """
    Return a mask for points inside an angular radius on a sphere.

    lon_grid, lat_grid : grid coordinates 
    lon0, lat0         : beam center 
    radius             : beam radius
    """
    if deg:
        # presuming all angles in degree
        lon_grid, lat_grid = np.deg2rad([lon_grid, lat_grid])
        lon0, lat0 = np.deg2rad([lon0, lat0])
        radius =  np.deg2rad(radius)

    cos_d = (
        np.sin(lat0) * np.sin(lat_grid)
        + np.cos(lat0) * np.cos(lat_grid) * np.cos(lon_grid - lon0)
    )
    return cos_d >= np.cos(radius)

def plot_beam(ax, lon0, lat0, beam_size=BEAM_SIZE, idx=None, projection=None, 
              frame="altaz", highlight=False, fill=True, **kwargs):
    """Draw single beam coverage on polar skymap"""
    beam_radius = beam_size / 2
    color = 'orangered' if highlight else 'royalblue'
    label = kwargs.pop("label") if "label" in kwargs else None

    if fill:
        lat_lb = 0 if frame == 'altaz' else -90
        lat_grid = np.linspace(lat_lb, 90, 360+1)
        lon_grid = np.linspace(0, 360, 720+1)
        LAT, LON = np.meshgrid(lat_grid, lon_grid)

        mask = _beam_mask(lon_grid=LON, lat_grid=LAT, 
                        lon0=lon0, lat0=lat0, radius=beam_radius)
        if frame == 'altaz':
            theta_beam, r_beam = plotConvertAltAz(LAT, LON, projection=projection)
        else: theta_beam, r_beam = LON, LAT

        ax.contourf(theta_beam, r_beam, mask.astype(float),
                    levels=[0.5, 1.5], colors=color, **kwargs)

    else:
        pa = np.linspace(0, 360, 20*int(360/beam_size))
        center = SkyCoord(lon0 * u.deg, lat0 * u.deg, frame=frame)
        edge = center.directional_offset_by(pa * u.deg, beam_radius * u.deg)
        if frame == "galactic":
            x, y = edge.l.deg, edge.b.deg
        elif frame == "altaz":
            alt, az = edge.alt.deg, edge.az.deg
            x, y = plotConvertAltAz(alt, az, projection=projection)
        else:
            raise NotImplementedError(f"unknown frame: {frame}")
        ax.plot(x, y, color=color, **kwargs)

    if frame == 'altaz':
        theta0, r0 = plotConvertAltAz(lat0, lon0, projection=projection)
    else: theta0, r0 = lon0, lat0
    ax.plot(theta0, r0, '.', markersize=3, color=color, label=label)

    if idx is not None:
        ax.annotate(
            f"{idx}",
            xy=(theta0, r0),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            color=color
        )

def plot_skymap(log:Table, curr_time=None, 
                beam_size=BEAM_SIZE, location=observatory, 
                star_cat = DEFAULT_STAR_CAT,
                radio_cat = DEFAULT_RADIO_CAT,
                enable_beam = True,
                timezone='KST', projection="equidistant", 
                **kwargs):
    
    curr_time = Time(curr_time) if curr_time is not None else Time.now()
    sky_mask = (log['type']=='sky') # select the sky frames
    time_mask = (Time(log['UT']) < curr_time) # presuming current time as UT
    log_sky = log[sky_mask & time_mask]

    altaz_frame = AltAz(obstime=curr_time, location=location)
    
    # Convert the beam pointings in current Alt/Az 
    gal_beams = SkyCoord(l=log_sky['l [deg]'] * u.deg, 
                         b=log_sky['b [deg]'] * u.deg, frame='galactic')
    altaz_beams_curr = gal_beams.transform_to(altaz_frame)
    beam_points = np.vstack([altaz_beams_curr.alt.deg,
                             altaz_beams_curr.az.deg]).T
    
    star_params, radio_params = None, None
    if "params_star" in kwargs:
        star_params = kwargs.pop("params_star")
    if "params_radio" in kwargs:
        radio_params = kwargs.pop("params_radio")

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"}, dpi=300)
    _format_skymap_axes(ax, curr_time, location=location, timezone=timezone)
    
    galGrid(ax, altaz_frame, grid_unit=(10, 10))
    plot_star_cat(ax, star_cat, altaz_frame, params=star_params)
    plot_radio_cat(ax, radio_cat, altaz_frame, params=radio_params)
    
    if enable_beam:
        alpha = min(0.2,5/len(beam_points))
        for i, (alt, az) in enumerate(beam_points):
            plot_beam(ax, lon0=az, lat0=alt, beam_size=beam_size, idx=i+1, 
                    frame='altaz', projection=projection, 
                    highlight=(i==beam_points.shape[0]-1), alpha=alpha) # highlight latest beam
        
    handles, labels = ax.get_legend_handles_labels()
    if handles: ax.legend(loc="upper right", frameon=True)
    return fig, ax


def plot_footprints(log, beam_size=BEAM_SIZE, 
                    timezone='KST', projection="equidistant", 
                    xlim=None, ylim=None, label=None, show_idx=False):
    
    log_sky = log[log['type'] == 'sky']  # select the sky frames

    # Galactic coordinates
    l = np.asarray(log_sky['l [deg]'])
    b = np.asarray(log_sky['b [deg]'])

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    alpha = min(0.5,10/len(l))
    # Beam coverage in Galactic coordinates
    for i, (l0, b0) in enumerate(zip(l, b)):
        label0 = None
        if i == 0:
            suffix = "" if label is None else f" {label}"
            label0=f"Observed Points{suffix}"

        plot_beam(ax, lon0=l0, lat0=b0, beam_size=beam_size, 
                  idx=(i + 1 if show_idx else None),
                  frame="galactic", projection=projection,
                  highlight=(i == len(l) - 1), fill=False, 
                  linewidth=0.8, alpha=alpha, label=label0)
    
    # Auto zoom around observed region
    if xlim is None:
        margin_l = max(beam_size, 5)
        xlim = (np.nanmax(l) + margin_l, np.nanmin(l) - margin_l)

    if ylim is None:
        margin_b = max(beam_size / 2, 5)
        ylim = (max(-90, np.nanmin(b) - margin_b),
                min(90, np.nanmax(b) + margin_b))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_xlabel('Galactic longitude $l$ [deg]')
    ax.set_ylabel('Galactic latitude $b$ [deg]')
    ax.axhline(0, color='k', alpha=0.3)
    ax.scatter(180, 0, marker='+', color='orange', s=200, zorder=2, label='Anticenter')
    ax.legend(loc='upper right', fontsize=15)

    return fig, ax


# =======================
# Velocity Conversion
# =======================
def radial_vel(freq, f0):
    # convert frequency to radial velocity
    return c * (f0 - freq) / freq

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


# ========================
# Power Spectrum
# ========================
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