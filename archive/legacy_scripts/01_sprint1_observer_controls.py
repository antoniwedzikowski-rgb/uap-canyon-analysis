#!/usr/bin/env python3
"""
Sprint 1: Kill-or-Confirm — Observer Infrastructure Controls
=============================================================
Tests whether the UAP-canyon proximity signal (OR=5.30 at 10km, OR=3.10 at 25km)
survives after adding observer-infrastructure covariates:
  - Port/marina proximity and density
  - Coastline morphology (complexity)

If the canyon distance coefficient remains significant after these controls,
the finding survives. If it collapses, the signal was an observer artifact.

Based on Phase B v2 pipeline (uap_ocean_phase_b_v2.py).
"""

import os
import warnings
import time
import json
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.stats import chi2 as chi2_dist
from scipy.spatial import cKDTree
from scipy.ndimage import label as ndimage_label
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import requests
import ssl
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()
# Workaround for macOS Python missing root certificates (common with brew/pyenv installs).
# Only affects OSM port data download, which is cached after first run.
ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# CONFIGURATION
# ============================================================
# REPO_DIR = this repo's root (one level up from notebooks/)
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# DATA_DIR = raw data files (included in repo under data/)
DATA_DIR = os.path.join(REPO_DIR, "data")
# Outputs go into repo
FIG_DIR = os.path.join(REPO_DIR, "figures")
RESULTS_DIR = os.path.join(REPO_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

R_EARTH = 6371.0  # km
CANYON_GRADIENT_THRESHOLD = 20.0  # m/km
COASTAL_BAND_KM = 200
N_CONTROL = 20000
PORT_CACHE_FILE = os.path.join(DATA_DIR, "port_coords_cache.npz")

print("=" * 70)
print("SPRINT 1: KILL-OR-CONFIRM — OBSERVER INFRASTRUCTURE CONTROLS")
print("=" * 70)
t_start = time.time()


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Accepts scalars or arrays."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def compute_or_at_thresholds(uap_dists, ctrl_dists, thresholds=[10, 25, 50, 75, 100]):
    """Compute odds ratios at distance thresholds."""
    results = []
    for t in thresholds:
        uap_near = int((uap_dists < t).sum())
        uap_far = len(uap_dists) - uap_near
        ctrl_near = int((ctrl_dists < t).sum())
        ctrl_far = len(ctrl_dists) - ctrl_near
        if ctrl_near > 0 and uap_near > 0 and uap_far > 0 and ctrl_far > 0:
            OR = (uap_near * ctrl_far) / (ctrl_near * uap_far)
            table = np.array([[uap_near, uap_far], [ctrl_near, ctrl_far]])
            _, p = stats.chi2_contingency(table)[:2]
        else:
            OR, p = np.nan, np.nan
        results.append({'threshold': t, 'OR': float(OR), 'p': float(p),
                        'uap_n': uap_near, 'ctrl_n': ctrl_near})
    return results


# ============================================================
# SECTION 1: DATA LOADING
# ============================================================
print("\n[SECTION 1] Loading data...")

# --- NUFORC ---
nuforc_cols = ['datetime', 'city', 'state', 'country', 'shape', 'duration_sec',
               'duration_text', 'comments', 'date_posted', 'latitude', 'longitude']
df_raw = pd.read_csv(os.path.join(DATA_DIR, "nuforc_reports.csv"),
                      names=nuforc_cols, header=None, low_memory=False)
print(f"  Raw reports: {len(df_raw)}")

df_raw['latitude'] = pd.to_numeric(df_raw['latitude'], errors='coerce')
df_raw['longitude'] = pd.to_numeric(df_raw['longitude'], errors='coerce')
df = df_raw.dropna(subset=['latitude', 'longitude']).copy()
df = df[(df['latitude'] != 0) & (df['longitude'] != 0)]
df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]
df = df.drop_duplicates(subset=['latitude', 'longitude', 'datetime'])

df_us = df[(df['latitude'] >= 20) & (df['latitude'] <= 55) &
           (df['longitude'] >= -135) & (df['longitude'] <= -55)].copy()
print(f"  After cleaning (US region): {len(df_us)}")

# --- ETOPO Bathymetry ---
print("  Loading ETOPO 2022 bathymetry...")
ds = xr.open_dataset(os.path.join(DATA_DIR, "etopo_subset.nc"))
elev = ds['z'].values
elev_lats = ds['lat'].values
elev_lons = ds['lon'].values
print(f"  Bathymetry grid: {elev.shape}")

# --- County Centroids (US Census 2020) ---
df_counties = pd.read_csv(os.path.join(DATA_DIR, "county_centroids_pop.csv"))
counties_lat = df_counties['lat'].values
counties_lon = df_counties['lon'].values
counties_pop = df_counties['pop'].values
county_tree = cKDTree(np.column_stack([counties_lat, counties_lon]))
print(f"  Counties: {len(df_counties)}, total pop: {counties_pop.sum():,.0f}")

# --- Military Bases (full 171-base CSV) ---
df_bases = pd.read_csv(os.path.join(DATA_DIR, "military_bases_us.csv"))
bases_lat = df_bases['lat'].values
bases_lon = df_bases['lon'].values
base_tree = cKDTree(np.column_stack([bases_lat, bases_lon]))
print(f"  Military bases: {len(df_bases)}")

print(f"  Data loaded ({time.time() - t_start:.1f}s)")

# ============================================================
# SECTION 2: COASTLINE & CANYON DETECTION
# ============================================================
print("\n[SECTION 2] Coastline & canyon detection...")

ocean_mask = elev < 0
land_mask = elev >= 0

# Ocean points
ocean_rows, ocean_cols = np.where(ocean_mask)
ocean_lats = elev_lats[ocean_rows]
ocean_lons = elev_lons[ocean_cols]
ocean_depths = elev[ocean_rows, ocean_cols]
ocean_tree = cKDTree(np.column_stack([ocean_lats, ocean_lons]))

# Coastline (land cells adjacent to ocean)
coast_mask = np.zeros_like(elev, dtype=bool)
for di in [-1, 0, 1]:
    for dj in [-1, 0, 1]:
        if di == 0 and dj == 0:
            continue
        shifted = np.roll(np.roll(ocean_mask, di, axis=0), dj, axis=1)
        coast_mask |= (land_mask & shifted)

coast_rows, coast_cols = np.where(coast_mask)
coast_lats = elev_lats[coast_rows]
coast_lons = elev_lons[coast_cols]
coast_tree = cKDTree(np.column_stack([coast_lats, coast_lons]))
print(f"  Coastline cells: {len(coast_lats)}")

# BallTree on coast points for morphology queries (Section 6)
coast_coords_rad = np.radians(np.column_stack([coast_lats, coast_lons]))
coast_ball_tree = BallTree(coast_coords_rad, metric='haversine')

# --- Shelf canyon detection (gradient-based) ---
shelf_mask = (elev < 0) & (elev > -500)
lat_res_km = np.abs(np.diff(elev_lats).mean()) * 111.0
mid_lat = 37.0
lon_res_km = np.abs(np.diff(elev_lons).mean()) * 111.0 * np.cos(np.radians(mid_lat))

grad_y, grad_x = np.gradient(elev.astype(float))
grad_mag = np.sqrt((grad_y / lat_res_km) ** 2 + (grad_x / lon_res_km) ** 2)

shelf_canyon_mask = shelf_mask & (grad_mag > CANYON_GRADIENT_THRESHOLD)

# Clean isolated pixels (keep clusters >= 3 cells)
labeled, n_features = ndimage_label(shelf_canyon_mask)
canyon_sizes = np.bincount(labeled.ravel())
for sl in np.where(canyon_sizes < 3)[0]:
    shelf_canyon_mask[labeled == sl] = False

canyon_rows, canyon_cols = np.where(shelf_canyon_mask)
canyon_lats = elev_lats[canyon_rows]
canyon_lons = elev_lons[canyon_cols]
canyon_gradients = grad_mag[canyon_rows, canyon_cols]
canyon_tree = cKDTree(np.column_stack([canyon_lats, canyon_lons]))

print(f"  Canyon cells (grad>{CANYON_GRADIENT_THRESHOLD} m/km): {len(canyon_lats)}")
print(f"  Mean gradient: {np.mean(canyon_gradients):.1f} m/km")

# ============================================================
# SECTION 3: COASTAL FILTERING & CONTROL POINTS
# ============================================================
print("\n[SECTION 3] Coastal filtering & control point generation...")

uap_lats = df_us['latitude'].values
uap_lons = df_us['longitude'].values

# Coast distance for all reports
coast_dists_deg, coast_idxs = coast_tree.query(
    np.column_stack([uap_lats, uap_lons]), k=1)
df_us['dist_to_coast_km'] = np.array([
    haversine_km(uap_lats[i], uap_lons[i],
                 coast_lats[coast_idxs[i]], coast_lons[coast_idxs[i]])
    for i in range(len(uap_lats))
])

# Filter to coastal band
df_coastal = df_us[df_us['dist_to_coast_km'] <= COASTAL_BAND_KM].copy()
coastal_lats = df_coastal['latitude'].values
coastal_lons = df_coastal['longitude'].values
print(f"  Coastal reports (0-{COASTAL_BAND_KM}km): {len(df_coastal)}")

# --- Metrics for coastal UAP reports ---
# Ocean depth
ocean_dists_deg, ocean_idxs = ocean_tree.query(
    np.column_stack([coastal_lats, coastal_lons]), k=1)
df_coastal['depth_nearest_ocean'] = ocean_depths[ocean_idxs]

# Canyon distance
canyon_dists_deg, canyon_idxs = canyon_tree.query(
    np.column_stack([coastal_lats, coastal_lons]), k=1)
df_coastal['dist_to_canyon_km'] = np.array([
    haversine_km(coastal_lats[i], coastal_lons[i],
                 canyon_lats[canyon_idxs[i]], canyon_lons[canyon_idxs[i]])
    for i in range(len(coastal_lats))
])

# Military distance (from full 171-base CSV)
base_dists_deg, _ = base_tree.query(
    np.column_stack([coastal_lats, coastal_lons]), k=1)
df_coastal['dist_to_military_km'] = base_dists_deg * 111.0

# Population density proxy
uap_county_dists, uap_county_idx = county_tree.query(
    np.column_stack([coastal_lats, coastal_lons]), k=5)
uap_pop = np.zeros(len(coastal_lats))
for k in range(5):
    d_km = uap_county_dists[:, k] * 111.0 + 1.0
    uap_pop += counties_pop[uap_county_idx[:, k]] / (d_km ** 2)
df_coastal['pop_density_proxy'] = uap_pop

# --- Generate control points (county-weighted, same as v2) ---
print("  Generating control points...")
grid_lat = np.linspace(22, 52, 300)
grid_lon = np.linspace(-130, -60, 600)
glat, glon = np.meshgrid(grid_lat, grid_lon, indexing='ij')
glat_flat = glat.flatten()
glon_flat = glon.flatten()

coast_dists_grid, _ = coast_tree.query(
    np.column_stack([glat_flat, glon_flat]), k=1)
coast_km_grid = coast_dists_grid * 111.0
coastal_grid_mask = coast_km_grid <= COASTAL_BAND_KM

glat_coastal = glat_flat[coastal_grid_mask]
glon_coastal = glon_flat[coastal_grid_mask]

# Kernel interpolation from county centroids
grid_county_dists, grid_county_idx = county_tree.query(
    np.column_stack([glat_coastal, glon_coastal]), k=10)
weights = np.zeros(len(glat_coastal))
for k in range(10):
    d_km = grid_county_dists[:, k] * 111.0 + 1.0
    weights += counties_pop[grid_county_idx[:, k]] / (d_km ** 2)

# Land/ocean weighting
lat_idx = np.clip(np.searchsorted(elev_lats, glat_coastal), 0, len(elev_lats) - 1)
lon_idx = np.clip(np.searchsorted(elev_lons, glon_coastal), 0, len(elev_lons) - 1)
grid_elev = elev[lat_idx, lon_idx]
land_weight = np.where(grid_elev >= 0, 3.0, 0.05)
weights *= land_weight
weights = weights / weights.sum()

chosen = np.random.choice(len(glat_coastal), size=N_CONTROL, p=weights, replace=True)
jitter = 0.12
ctrl_lats = glat_coastal[chosen] + np.random.uniform(-jitter, jitter, N_CONTROL)
ctrl_lons = glon_coastal[chosen] + np.random.uniform(-jitter, jitter, N_CONTROL)

# Filter control to coastal band
ctrl_coast_dists, ctrl_coast_idx = coast_tree.query(
    np.column_stack([ctrl_lats, ctrl_lons]), k=1)
ctrl_coast_km = np.array([
    haversine_km(ctrl_lats[i], ctrl_lons[i],
                 coast_lats[ctrl_coast_idx[i]], coast_lons[ctrl_coast_idx[i]])
    for i in range(len(ctrl_lats))
])
ctrl_mask = ctrl_coast_km <= COASTAL_BAND_KM
ctrl_lats = ctrl_lats[ctrl_mask]
ctrl_lons = ctrl_lons[ctrl_mask]
ctrl_coast_km = ctrl_coast_km[ctrl_mask]
N_CONTROL = len(ctrl_lats)
print(f"  Control points after coastal filter: {N_CONTROL}")

# Control metrics
ctrl_ocean_dists, ctrl_ocean_idx = ocean_tree.query(
    np.column_stack([ctrl_lats, ctrl_lons]), k=1)
ctrl_depths = ocean_depths[ctrl_ocean_idx]

ctrl_canyon_dists, ctrl_canyon_idx = canyon_tree.query(
    np.column_stack([ctrl_lats, ctrl_lons]), k=1)
ctrl_canyon_km = np.array([
    haversine_km(ctrl_lats[i], ctrl_lons[i],
                 canyon_lats[ctrl_canyon_idx[i]], canyon_lons[ctrl_canyon_idx[i]])
    for i in range(N_CONTROL)
])

ctrl_base_dists, _ = base_tree.query(
    np.column_stack([ctrl_lats, ctrl_lons]), k=1)
ctrl_base_km = ctrl_base_dists * 111.0

ctrl_county_dists, ctrl_county_idx = county_tree.query(
    np.column_stack([ctrl_lats, ctrl_lons]), k=5)
ctrl_pop = np.zeros(N_CONTROL)
for k in range(5):
    d_km = ctrl_county_dists[:, k] * 111.0 + 1.0
    ctrl_pop += counties_pop[ctrl_county_idx[:, k]] / (d_km ** 2)

print(f"  Section 3 done ({time.time() - t_start:.1f}s)")

# ============================================================
# SECTION 4: PORT/MARINA DATA ACQUISITION
# ============================================================
print("\n[SECTION 4] Acquiring port/marina data...")


def fetch_ports_overpass():
    """Query OSM Overpass for marinas, ports, harbours in CONUS bbox."""
    query = """
    [out:json][timeout:180];
    (
      node["leisure"="marina"](20,-135,55,-55);
      node["industrial"="port"](20,-135,55,-55);
      node["harbour"="yes"](20,-135,55,-55);
      node["seamark:type"="harbour"](20,-135,55,-55);
      way["leisure"="marina"](20,-135,55,-55);
      way["industrial"="port"](20,-135,55,-55);
    );
    out center;
    """
    url = "https://overpass-api.de/api/interpreter"
    resp = requests.post(url, data={'data': query}, timeout=200)
    resp.raise_for_status()
    data = resp.json()
    coords = []
    for elem in data['elements']:
        if 'lat' in elem and 'lon' in elem:
            coords.append((elem['lat'], elem['lon']))
        elif 'center' in elem:
            coords.append((elem['center']['lat'], elem['center']['lon']))
    return np.array(coords), 'overpass_osm'


def fetch_ports_noaa_tides():
    """NOAA tide/current stations as port proxy."""
    url = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
    params = {'type': 'tidepredictions', 'units': 'metric'}
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    stations = resp.json()['stations']
    coords = []
    for s in stations:
        lat, lon = float(s['lat']), float(s['lng'])
        if 20 <= lat <= 55 and -135 <= lon <= -55:
            coords.append((lat, lon))
    return np.array(coords), 'noaa_tide_stations'


# Try cache first, then network fallbacks
port_coords, port_source = None, None

if os.path.exists(PORT_CACHE_FILE):
    data = np.load(PORT_CACHE_FILE, allow_pickle=True)
    port_coords = data['port_coords']
    port_source = str(data['source'])
    print(f"  Loaded {len(port_coords)} ports from cache ({port_source})")
else:
    for fetch_fn, name in [
        (fetch_ports_overpass, "Overpass OSM"),
        (fetch_ports_noaa_tides, "NOAA Tide Stations"),
    ]:
        try:
            print(f"  Trying {name}...")
            port_coords, port_source = fetch_fn()
            if len(port_coords) >= 50:
                print(f"  SUCCESS: {len(port_coords)} ports from {name}")
                np.savez(PORT_CACHE_FILE, port_coords=port_coords, source=np.array(port_source))
                break
            else:
                print(f"  WARNING: Only {len(port_coords)} ports, trying next...")
                port_coords = None
        except Exception as e:
            print(f"  FAILED ({name}): {e}")

if port_coords is None or len(port_coords) < 50:
    print("  FATAL: Could not acquire port data from any source.")
    raise RuntimeError("Port data acquisition failed")

print(f"  Port data: {len(port_coords)} locations from {port_source}")

# ============================================================
# SECTION 5: PORT DENSITY METRICS (BallTree)
# ============================================================
print("\n[SECTION 5] Computing port density metrics...")

port_coords_rad = np.radians(port_coords)
port_tree = BallTree(port_coords_rad, metric='haversine')

# --- UAP points ---
uap_coords_rad = np.radians(np.column_stack([coastal_lats, coastal_lons]))

port_dists_rad, _ = port_tree.query(uap_coords_rad, k=1)
df_coastal['dist_to_nearest_port'] = port_dists_rad.flatten() * R_EARTH

counts_25_uap = port_tree.query_radius(uap_coords_rad, r=25.0 / R_EARTH, count_only=True)
df_coastal['port_count_25km'] = counts_25_uap

counts_50_uap = port_tree.query_radius(uap_coords_rad, r=50.0 / R_EARTH, count_only=True)
df_coastal['port_count_50km'] = counts_50_uap

# --- Control points ---
ctrl_coords_rad = np.radians(np.column_stack([ctrl_lats, ctrl_lons]))

ctrl_port_dists_rad, _ = port_tree.query(ctrl_coords_rad, k=1)
ctrl_port_km = ctrl_port_dists_rad.flatten() * R_EARTH

ctrl_port_count_25 = port_tree.query_radius(ctrl_coords_rad, r=25.0 / R_EARTH, count_only=True)
ctrl_port_count_50 = port_tree.query_radius(ctrl_coords_rad, r=50.0 / R_EARTH, count_only=True)

print(f"  UAP: mean port dist={df_coastal['dist_to_nearest_port'].mean():.1f}km, "
      f"mean count@25km={df_coastal['port_count_25km'].mean():.1f}")
print(f"  Ctrl: mean port dist={np.mean(ctrl_port_km):.1f}km, "
      f"mean count@25km={np.mean(ctrl_port_count_25):.1f}")
print(f"  Port metrics computed ({time.time() - t_start:.1f}s)")

# ============================================================
# SECTION 6: COASTLINE MORPHOLOGY PROXY
# ============================================================
print("\n[SECTION 6] Computing coastline morphology (bearing variance, 25km window)...")


def compute_coast_complexity(query_lats, query_lons, coast_bt, c_lats, c_lons,
                             radius_km=25.0, batch_size=5000):
    """
    Circular standard deviation of bearings from query point to nearby coastline points.
    High circ_std = complex coast (bays, inlets), Low = straight coast.
    """
    query_rad = np.radians(np.column_stack([query_lats, query_lons]))
    n = len(query_lats)
    complexities = np.full(n, np.nan)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_rad = query_rad[start:end]
        indices = coast_bt.query_radius(batch_rad, r=radius_km / R_EARTH)

        for j, idx in enumerate(indices):
            i = start + j
            if len(idx) < 3:
                continue

            # Bearings from query point to each coast point
            qlat_r = np.radians(query_lats[i])
            qlon_r = np.radians(query_lons[i])
            clat_r = np.radians(c_lats[idx])
            clon_r = np.radians(c_lons[idx])
            dlon = clon_r - qlon_r

            x = np.cos(clat_r) * np.sin(dlon)
            y = (np.cos(qlat_r) * np.sin(clat_r) -
                 np.sin(qlat_r) * np.cos(clat_r) * np.cos(dlon))
            bearings = np.arctan2(x, y)

            # Circular standard deviation: sqrt(-2 * ln(R_bar))
            mean_cos = np.mean(np.cos(bearings))
            mean_sin = np.mean(np.sin(bearings))
            R_bar = np.sqrt(mean_cos ** 2 + mean_sin ** 2)
            R_bar = np.clip(R_bar, 1e-10, 1.0)
            complexities[i] = np.sqrt(-2.0 * np.log(R_bar))

    return complexities


t6 = time.time()
coast_complexity_uap = compute_coast_complexity(
    coastal_lats, coastal_lons, coast_ball_tree, coast_lats, coast_lons)
df_coastal['coast_complexity'] = coast_complexity_uap

coast_complexity_ctrl = compute_coast_complexity(
    ctrl_lats, ctrl_lons, coast_ball_tree, coast_lats, coast_lons)

n_nan_uap = np.isnan(coast_complexity_uap).sum()
n_nan_ctrl = np.isnan(coast_complexity_ctrl).sum()
print(f"  UAP: mean complexity={np.nanmean(coast_complexity_uap):.3f}, NaN={n_nan_uap}")
print(f"  Ctrl: mean complexity={np.nanmean(coast_complexity_ctrl):.3f}, NaN={n_nan_ctrl}")
print(f"  Coastline morphology computed ({time.time() - t6:.1f}s)")

# ============================================================
# SECTION 7: FEATURE MATRIX ASSEMBLY
# ============================================================
print("\n[SECTION 7] Assembling feature matrix...")

# Log-transform port counts (right-skewed, add 1 to handle zeros)
df_coastal['log_port_count_25km'] = np.log1p(df_coastal['port_count_25km'])
ctrl_log_port_count_25 = np.log1p(ctrl_port_count_25)

# UAP features
feature_cols = ['dist_to_canyon_km', 'dist_to_coast_km', 'dist_to_military_km',
                'pop_density_proxy', 'depth_nearest_ocean',
                'dist_to_nearest_port', 'log_port_count_25km', 'coast_complexity']

X_uap = df_coastal[feature_cols].copy()
y_uap_full = np.ones(len(X_uap))

# Control features
X_ctrl = pd.DataFrame({
    'dist_to_canyon_km': ctrl_canyon_km,
    'dist_to_coast_km': ctrl_coast_km,
    'dist_to_military_km': ctrl_base_km,
    'pop_density_proxy': ctrl_pop,
    'depth_nearest_ocean': ctrl_depths,
    'dist_to_nearest_port': ctrl_port_km,
    'log_port_count_25km': ctrl_log_port_count_25,
    'coast_complexity': coast_complexity_ctrl,
})
y_ctrl_full = np.zeros(len(X_ctrl))

# Combine and drop NaN
X_all_raw = pd.concat([X_uap, X_ctrl], ignore_index=True)
y_all_raw = np.concatenate([y_uap_full, y_ctrl_full])

valid_mask = X_all_raw.notna().all(axis=1)
X_all_clean = X_all_raw[valid_mask].reset_index(drop=True)
y_all = y_all_raw[valid_mask.values]
n_dropped = (~valid_mask).sum()
n_uap_valid = int(y_all[:len(X_uap)].sum()) if len(y_all) > 0 else 0

print(f"  Feature matrix: {len(X_all_clean)} rows ({n_dropped} dropped for NaN)")
print(f"  UAP: {int(y_all.sum())}, Control: {int(len(y_all) - y_all.sum())}")

# Z-score all features
scaler = StandardScaler()
feature_names = list(X_all_clean.columns)
X_scaled = pd.DataFrame(scaler.fit_transform(X_all_clean), columns=feature_names)

# --- VIF check ---
print("\n  VIF check (multicollinearity):")
X_vif = sm.add_constant(X_scaled)
vif_table = {}
for i, col in enumerate(X_vif.columns):
    if col == 'const':
        continue
    vif_val = variance_inflation_factor(X_vif.values, i)
    vif_table[col] = vif_val
    flag = ' ***' if vif_val > 10 else ' **' if vif_val > 5 else ''
    print(f"    VIF({col}): {vif_val:.2f}{flag}")

# Auto-drop features with VIF > 10
dropped_features = []
active_features = list(feature_names)
while True:
    X_check = sm.add_constant(X_scaled[active_features])
    vifs = []
    for i in range(len(active_features)):
        vifs.append(variance_inflation_factor(X_check.values, i + 1))
    max_vif_idx = np.argmax(vifs)
    if vifs[max_vif_idx] > 10:
        dropped = active_features.pop(max_vif_idx)
        dropped_features.append((dropped, float(vifs[max_vif_idx])))
        print(f"    DROPPING {dropped} (VIF={vifs[max_vif_idx]:.1f})")
    else:
        break

if dropped_features:
    print(f"  Dropped {len(dropped_features)} features for VIF > 10")
else:
    print(f"  All features retained (max VIF < 10)")

print(f"  Active features for regression: {active_features}")

# ============================================================
# SECTION 8: LOGISTIC REGRESSION (statsmodels)
# ============================================================
print("\n[SECTION 8] Logistic regression...")

# --- Original model (v2 features only) ---
orig_features = ['depth_nearest_ocean', 'dist_to_coast_km', 'dist_to_canyon_km',
                 'pop_density_proxy', 'dist_to_military_km']
X_orig = sm.add_constant(X_scaled[orig_features])
model_orig = sm.Logit(y_all, X_orig).fit(disp=0, maxiter=1000)

print("\n  ORIGINAL MODEL (v2 features only):")
print(f"  Pseudo R²: {model_orig.prsquared:.4f}, AIC: {model_orig.aic:.1f}, LL: {model_orig.llf:.1f}")
ci_orig = model_orig.conf_int()
print(f"  {'Feature':<25} {'Coef':>10} {'95% CI':>25} {'p':>12}")
print(f"  {'-' * 75}")
for name in orig_features:
    coef = model_orig.params[name]
    lo, hi = ci_orig.loc[name]
    p = model_orig.pvalues[name]
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"  {name:<25} {coef:>+10.4f} [{lo:>+.4f}, {hi:>+.4f}] {p:>12.2e} {sig}")

# --- Full model (with observer controls) ---
X_full = sm.add_constant(X_scaled[active_features])
model_full = sm.Logit(y_all, X_full).fit(disp=0, maxiter=1000)

print(f"\n  FULL MODEL (with observer infrastructure controls):")
print(f"  Pseudo R²: {model_full.prsquared:.4f}, AIC: {model_full.aic:.1f}, LL: {model_full.llf:.1f}")
ci_full = model_full.conf_int()
print(f"  {'Feature':<25} {'Coef':>10} {'95% CI':>25} {'p':>12}")
print(f"  {'-' * 75}")
for name in active_features:
    coef = model_full.params[name]
    lo, hi = ci_full.loc[name]
    p = model_full.pvalues[name]
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"  {name:<25} {coef:>+10.4f} [{lo:>+.4f}, {hi:>+.4f}] {p:>12.2e} {sig}")

# ============================================================
# SECTION 9: NESTED MODEL LR TEST & KILL-OR-CONFIRM VERDICT
# ============================================================
print("\n[SECTION 9] Nested model LR test & verdict...")

# Reduced model: full model MINUS dist_to_canyon_km
reduced_features = [f for f in active_features if f != 'dist_to_canyon_km']
X_reduced = sm.add_constant(X_scaled[reduced_features])
model_reduced = sm.Logit(y_all, X_reduced).fit(disp=0, maxiter=1000)

lr_stat = 2 * (model_full.llf - model_reduced.llf)
lr_df = 1
lr_p = chi2_dist.sf(lr_stat, lr_df)

print(f"\n  NESTED LR TEST (canyon effect after observer controls):")
print(f"  Full model LL:    {model_full.llf:.2f}")
print(f"  Reduced model LL: {model_reduced.llf:.2f}")
print(f"  LR statistic:     {lr_stat:.4f}")
print(f"  df:               {lr_df}")
print(f"  p-value:          {lr_p:.2e}")

# Canyon coefficient comparison
canyon_coef_orig = model_orig.params['dist_to_canyon_km']
canyon_ci_orig = ci_orig.loc['dist_to_canyon_km'].values

canyon_coef_full = model_full.params['dist_to_canyon_km']
canyon_ci_full = ci_full.loc['dist_to_canyon_km'].values

attenuation = 1 - abs(canyon_coef_full) / abs(canyon_coef_orig) if canyon_coef_orig != 0 else 0.0

print(f"\n  CANYON COEFFICIENT COMPARISON:")
print(f"  Original (v2 features):   {canyon_coef_orig:+.4f} [{canyon_ci_orig[0]:+.4f}, {canyon_ci_orig[1]:+.4f}]")
print(f"  Full (+ observer ctrls):  {canyon_coef_full:+.4f} [{canyon_ci_full[0]:+.4f}, {canyon_ci_full[1]:+.4f}]")
print(f"  Attenuation: {attenuation * 100:.1f}%")

# Kill-or-confirm verdict
ci_excludes_zero = (canyon_ci_full[0] * canyon_ci_full[1]) > 0

if lr_p < 0.01 and ci_excludes_zero:
    if attenuation < 0.30:
        verdict = "SURVIVES — canyon effect robust to observer controls"
        verdict_short = "SURVIVES"
    elif attenuation < 0.60:
        verdict = f"SURVIVES (attenuated) — canyon effect reduced by {attenuation * 100:.0f}% but still significant"
        verdict_short = "SURVIVES_ATTENUATED"
    else:
        verdict = f"INCONCLUSIVE — canyon effect reduced by {attenuation * 100:.0f}%, nominally significant but heavily attenuated"
        verdict_short = "INCONCLUSIVE"
elif lr_p > 0.05 or not ci_excludes_zero:
    verdict = "KILLED — canyon effect not significant after observer controls"
    verdict_short = "KILLED"
else:
    verdict = f"INCONCLUSIVE — LR p={lr_p:.2e}, CI={'excludes' if ci_excludes_zero else 'includes'} zero, attenuation={attenuation * 100:.0f}%"
    verdict_short = "INCONCLUSIVE"

print(f"\n  {'=' * 50}")
print(f"  VERDICT: {verdict}")
print(f"  {'=' * 50}")

# ============================================================
# SECTION 10: STRATIFIED ORs BY PORT-DENSITY TERCILES
# ============================================================
print("\n[SECTION 10] Stratified ORs by port-density terciles...")

uap_canyon_dists = df_coastal['dist_to_canyon_km'].values
uap_port_25 = df_coastal['port_count_25km'].values

all_port_count = np.concatenate([uap_port_25, ctrl_port_count_25])
tercile_bounds = np.percentile(all_port_count, [33.3, 66.7])

tercile_names = [
    f'Low ports (0-{tercile_bounds[0]:.0f})',
    f'Medium ports ({tercile_bounds[0]:.0f}-{tercile_bounds[1]:.0f})',
    f'High ports ({tercile_bounds[1]:.0f}+)',
]

print(f"  Tercile bounds: {tercile_bounds[0]:.0f}, {tercile_bounds[1]:.0f}")
print(f"\n  {'Tercile':<35} {'OR@10km':>10} {'OR@25km':>10} {'UAP_n':>8} {'Ctrl_n':>8}")
print(f"  {'-' * 75}")

stratified_results = []
for t_idx, t_name in enumerate(tercile_names):
    if t_idx == 0:
        uap_mask_t = uap_port_25 <= tercile_bounds[0]
        ctrl_mask_t = ctrl_port_count_25 <= tercile_bounds[0]
    elif t_idx == 1:
        uap_mask_t = ((uap_port_25 > tercile_bounds[0]) &
                      (uap_port_25 <= tercile_bounds[1]))
        ctrl_mask_t = ((ctrl_port_count_25 > tercile_bounds[0]) &
                       (ctrl_port_count_25 <= tercile_bounds[1]))
    else:
        uap_mask_t = uap_port_25 > tercile_bounds[1]
        ctrl_mask_t = ctrl_port_count_25 > tercile_bounds[1]

    uap_canyon_t = uap_canyon_dists[uap_mask_t]
    ctrl_canyon_t = ctrl_canyon_km[ctrl_mask_t]

    if len(uap_canyon_t) < 30 or len(ctrl_canyon_t) < 30:
        print(f"  {t_name:<35} (insufficient: UAP={len(uap_canyon_t)}, Ctrl={len(ctrl_canyon_t)})")
        stratified_results.append({
            'tercile': t_name, 'OR_10': np.nan, 'OR_25': np.nan,
            'uap_n': int(uap_mask_t.sum()), 'ctrl_n': int(ctrl_mask_t.sum())
        })
        continue

    ors = compute_or_at_thresholds(uap_canyon_t, ctrl_canyon_t, thresholds=[10, 25])
    or_10 = ors[0]['OR'] if ors else np.nan
    or_25 = ors[1]['OR'] if len(ors) > 1 else np.nan

    print(f"  {t_name:<35} {or_10:>10.3f} {or_25:>10.3f} {int(uap_mask_t.sum()):>8} {int(ctrl_mask_t.sum()):>8}")
    stratified_results.append({
        'tercile': t_name, 'OR_10': float(or_10), 'OR_25': float(or_25),
        'uap_n': int(uap_mask_t.sum()), 'ctrl_n': int(ctrl_mask_t.sum()),
        'or_details_10': ors[0] if ors else None,
        'or_details_25': ors[1] if len(ors) > 1 else None,
    })

# ============================================================
# SECTION 11: SENSITIVITY — CANYON DEFINITION PARAMETERS
# ============================================================
print("\n[SECTION 11] Sensitivity analysis (gradient x depth grid)...")

gradient_thresholds = [15.0, 20.0, 30.0]
depth_ranges = [(-500, 0), (-200, 0)]

# Precompute valid indices for UAP and control in the feature matrix
# (so we can rebuild just the canyon column)
uap_valid_mask = X_uap.notna().all(axis=1).values
ctrl_valid_mask = X_ctrl.notna().all(axis=1).values

sensitivity_results = []

for grad_thresh in gradient_thresholds:
    for depth_lo, depth_hi in depth_ranges:
        label = f"grad>{grad_thresh}, depth={depth_lo}..{depth_hi}"
        print(f"  Running: {label}...")

        # Recompute canyon mask
        shelf_mask_s = (elev < depth_hi) & (elev > depth_lo)
        canyon_mask_s = shelf_mask_s & (grad_mag > grad_thresh)

        # Clean clusters < 3
        labeled_s, _ = ndimage_label(canyon_mask_s)
        sizes_s = np.bincount(labeled_s.ravel())
        for sl in np.where(sizes_s < 3)[0]:
            canyon_mask_s[labeled_s == sl] = False

        c_rows, c_cols = np.where(canyon_mask_s)
        if len(c_rows) < 10:
            print(f"    Skipped: only {len(c_rows)} canyon cells")
            sensitivity_results.append({
                'gradient_threshold': grad_thresh,
                'depth_range': f'{depth_lo} to {depth_hi}',
                'n_canyon_cells': len(c_rows),
                'canyon_coef': np.nan, 'canyon_ci_lo': np.nan,
                'canyon_ci_hi': np.nan, 'canyon_p': np.nan,
            })
            continue

        c_lats_s = elev_lats[c_rows]
        c_lons_s = elev_lons[c_cols]
        c_tree_s = cKDTree(np.column_stack([c_lats_s, c_lons_s]))

        # Recompute canyon distances for UAP
        _, idx_u = c_tree_s.query(np.column_stack([coastal_lats, coastal_lons]), k=1)
        uap_can_s = np.array([
            haversine_km(coastal_lats[i], coastal_lons[i],
                         c_lats_s[idx_u[i]], c_lons_s[idx_u[i]])
            for i in range(len(coastal_lats))
        ])

        # Recompute canyon distances for control
        _, idx_c = c_tree_s.query(np.column_stack([ctrl_lats, ctrl_lons]), k=1)
        ctrl_can_s = np.array([
            haversine_km(ctrl_lats[i], ctrl_lons[i],
                         c_lats_s[idx_c[i]], c_lons_s[idx_c[i]])
            for i in range(N_CONTROL)
        ])

        # Rebuild canyon column in scaled feature matrix
        # Concatenate UAP (valid) + ctrl (valid)
        uap_can_valid = uap_can_s[uap_valid_mask]
        ctrl_can_valid = ctrl_can_s[ctrl_valid_mask]
        canyon_vals = np.concatenate([uap_can_valid, ctrl_can_valid])

        # Need to align with X_all_clean
        if len(canyon_vals) != len(X_all_clean):
            # Fallback: use valid_mask from Section 7
            canyon_all = np.concatenate([uap_can_s, ctrl_can_s])
            canyon_vals = canyon_all[valid_mask.values]

        if len(canyon_vals) != len(X_all_clean):
            print(f"    Size mismatch, skipping")
            sensitivity_results.append({
                'gradient_threshold': grad_thresh,
                'depth_range': f'{depth_lo} to {depth_hi}',
                'n_canyon_cells': len(c_rows),
                'canyon_coef': np.nan, 'canyon_ci_lo': np.nan,
                'canyon_ci_hi': np.nan, 'canyon_p': np.nan,
            })
            continue

        # Z-score the new canyon distance
        canyon_z = (canyon_vals - canyon_vals.mean()) / (canyon_vals.std() + 1e-10)

        X_s = X_scaled[active_features].copy()
        X_s['dist_to_canyon_km'] = canyon_z
        X_s_full = sm.add_constant(X_s)

        try:
            model_s = sm.Logit(y_all, X_s_full).fit(disp=0, maxiter=1000)
            canyon_coef_s = model_s.params['dist_to_canyon_km']
            canyon_ci_s = model_s.conf_int().loc['dist_to_canyon_km']
            canyon_p_s = model_s.pvalues['dist_to_canyon_km']

            sig = '***' if canyon_p_s < 0.001 else '**' if canyon_p_s < 0.01 else '*' if canyon_p_s < 0.05 else ''
            print(f"    Cells={len(c_rows):>6}, coef={canyon_coef_s:+.4f} "
                  f"[{canyon_ci_s.iloc[0]:+.4f}, {canyon_ci_s.iloc[1]:+.4f}] p={canyon_p_s:.2e} {sig}")

            sensitivity_results.append({
                'gradient_threshold': float(grad_thresh),
                'depth_range': f'{depth_lo} to {depth_hi}',
                'n_canyon_cells': int(len(c_rows)),
                'canyon_coef': float(canyon_coef_s),
                'canyon_ci_lo': float(canyon_ci_s.iloc[0]),
                'canyon_ci_hi': float(canyon_ci_s.iloc[1]),
                'canyon_p': float(canyon_p_s),
            })
        except Exception as e:
            print(f"    Model failed: {e}")
            sensitivity_results.append({
                'gradient_threshold': float(grad_thresh),
                'depth_range': f'{depth_lo} to {depth_hi}',
                'n_canyon_cells': int(len(c_rows)),
                'canyon_coef': np.nan, 'canyon_ci_lo': np.nan,
                'canyon_ci_hi': np.nan, 'canyon_p': np.nan,
            })

print(f"  Sensitivity done ({time.time() - t_start:.1f}s)")

# ============================================================
# SECTION 12: VISUALIZATIONS
# ============================================================
print("\n[SECTION 12] Generating visualizations...")

# --- Plot 1: Forest plot — model comparison ---
print("  sprint1_model_comparison.png...")
fig, ax = plt.subplots(figsize=(11, 8))

# Collect coefficients from both models for shared features
all_feature_names = list(set(orig_features + active_features))
all_feature_names.sort()

y_positions = []
y_labels = []
y_pos = 0
colors_orig = '#e74c3c'
colors_full = '#3498db'

for feat in all_feature_names:
    # Full model
    if feat in active_features:
        coef_f = model_full.params[feat]
        ci_f = ci_full.loc[feat].values
        ax.errorbar(coef_f, y_pos, xerr=[[coef_f - ci_f[0]], [ci_f[1] - coef_f]],
                    fmt='s', color=colors_full, markersize=8, capsize=4, linewidth=2,
                    label='Full model' if y_pos == 0 else None)
    y_positions.append(y_pos)
    y_pos += 0.4

    # Original model
    if feat in orig_features:
        coef_o = model_orig.params[feat]
        ci_o = ci_orig.loc[feat].values
        ax.errorbar(coef_o, y_pos, xerr=[[coef_o - ci_o[0]], [ci_o[1] - coef_o]],
                    fmt='o', color=colors_orig, markersize=8, capsize=4, linewidth=2,
                    label='Original model' if y_pos == 0.4 else None)
    y_positions.append(y_pos)
    y_pos += 0.8
    y_labels.append(feat)

# Feature labels
label_positions = [y_positions[i * 2] + 0.2 for i in range(len(all_feature_names))]
ax.set_yticks(label_positions)
ax.set_yticklabels(all_feature_names, fontsize=10)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Standardized Coefficient', fontsize=12)
ax.set_title(f'Sprint 1: Model Comparison — Original vs Full (with observer controls)\n'
             f'Canyon attenuation: {attenuation * 100:.1f}% | Verdict: {verdict_short}',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.2, axis='x')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'sprint1_model_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()

# --- Plot 2: Sensitivity heatmap ---
print("  sprint1_sensitivity.png...")
fig, ax = plt.subplots(figsize=(8, 6))

grad_labels = [f'{g} m/km' for g in gradient_thresholds]
depth_labels = ['0 to -500m', '0 to -200m']
matrix = np.full((len(gradient_thresholds), len(depth_ranges)), np.nan)
annot_matrix = np.empty((len(gradient_thresholds), len(depth_ranges)), dtype=object)

for sr in sensitivity_results:
    g_idx = gradient_thresholds.index(sr['gradient_threshold'])
    d_idx = depth_ranges.index(tuple(map(int, sr['depth_range'].replace(' to ', ',').split(','))) if ',' in sr['depth_range'].replace(' to ', ',') else (-500, 0))
    matrix[g_idx, d_idx] = sr['canyon_coef']
    p = sr['canyon_p']
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    annot_matrix[g_idx, d_idx] = f"{sr['canyon_coef']:.4f}\n({sig})\nn={sr['n_canyon_cells']}"

# Rebuild matrix more carefully
matrix = np.full((len(gradient_thresholds), len(depth_ranges)), np.nan)
annot_matrix = [['' for _ in depth_ranges] for _ in gradient_thresholds]

for sr in sensitivity_results:
    g_idx = gradient_thresholds.index(sr['gradient_threshold'])
    dr_str = sr['depth_range']
    for d_idx, (dlo, dhi) in enumerate(depth_ranges):
        if f'{dlo} to {dhi}' == dr_str:
            matrix[g_idx, d_idx] = sr['canyon_coef']
            p = sr['canyon_p']
            if np.isnan(p):
                sig = ''
            else:
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            n_cells = sr['n_canyon_cells']
            coef = sr['canyon_coef']
            annot_matrix[g_idx][d_idx] = f"{coef:.4f}\n{sig}\nn={n_cells}"
            break

sns.heatmap(matrix, ax=ax, cmap='RdBu_r', center=0,
            xticklabels=depth_labels, yticklabels=grad_labels,
            annot=np.array(annot_matrix), fmt='', annot_kws={'fontsize': 10},
            linewidths=2, linecolor='white', cbar_kws={'label': 'Canyon coefficient (std.)'})
ax.set_xlabel('Depth Range', fontsize=12)
ax.set_ylabel('Gradient Threshold', fontsize=12)
ax.set_title('Sensitivity: Canyon Coefficient Across Definitions\n(with observer infrastructure controls)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'sprint1_sensitivity.png'), dpi=200, bbox_inches='tight')
plt.close()

# --- Plot 3: Stratified OR bar chart ---
print("  sprint1_stratified_or.png...")
fig, ax = plt.subplots(figsize=(10, 6))

valid_strat = [s for s in stratified_results if not np.isnan(s.get('OR_10', np.nan))]
if valid_strat:
    x = np.arange(len(valid_strat))
    width = 0.35
    or10_vals = [s['OR_10'] for s in valid_strat]
    or25_vals = [s['OR_25'] for s in valid_strat]
    labels = [s['tercile'] for s in valid_strat]

    bars1 = ax.bar(x - width / 2, or10_vals, width, label='OR @ 10km',
                   color='#e74c3c', alpha=0.8, edgecolor='darkred')
    bars2 = ax.bar(x + width / 2, or25_vals, width, label='OR @ 25km',
                   color='#3498db', alpha=0.8, edgecolor='darkblue')

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='OR = 1 (no effect)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Odds Ratio (canyon proximity)', fontsize=12)
    ax.set_title('Canyon OR by Port-Density Tercile\n(controls for observer infrastructure confound)',
                 fontsize=13, fontweight='bold')

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.05,
                f'{h:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.05,
                f'{h:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
else:
    ax.text(0.5, 0.5, 'No valid stratified OR data', transform=ax.transAxes,
            ha='center', va='center', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'sprint1_stratified_or.png'), dpi=200, bbox_inches='tight')
plt.close()

print("  Visualizations saved!")

# ============================================================
# SECTION 13: EXPORT RESULTS
# ============================================================
print("\n[SECTION 13] Exporting results...")

# Recompute final VIF for active features
X_final_vif = sm.add_constant(X_scaled[active_features])
final_vifs = {}
for i, col in enumerate(active_features):
    final_vifs[col] = float(variance_inflation_factor(X_final_vif.values, i + 1))

# Build full results dict
results_out = {
    'metadata': {
        'script': 'sprint1_observer_controls.py',
        'n_uap_coastal': int(y_all.sum()),
        'n_control': int(len(y_all) - y_all.sum()),
        'port_source': port_source,
        'n_ports': int(len(port_coords)),
        'n_military_bases': int(len(df_bases)),
        'canyon_gradient_threshold': CANYON_GRADIENT_THRESHOLD,
        'n_canyon_cells': int(len(canyon_lats)),
        'runtime_seconds': round(time.time() - t_start, 1),
    },
    'original_model': {
        'features': orig_features,
        'canyon_coef': float(canyon_coef_orig),
        'canyon_ci': [float(canyon_ci_orig[0]), float(canyon_ci_orig[1])],
        'canyon_p': float(model_orig.pvalues['dist_to_canyon_km']),
        'pseudo_r2': float(model_orig.prsquared),
        'aic': float(model_orig.aic),
        'log_likelihood': float(model_orig.llf),
        'all_coefs': {name: float(model_orig.params[name]) for name in orig_features},
        'all_pvalues': {name: float(model_orig.pvalues[name]) for name in orig_features},
    },
    'full_model': {
        'features': active_features,
        'canyon_coef': float(canyon_coef_full),
        'canyon_ci': [float(canyon_ci_full[0]), float(canyon_ci_full[1])],
        'canyon_p': float(model_full.pvalues['dist_to_canyon_km']),
        'pseudo_r2': float(model_full.prsquared),
        'aic': float(model_full.aic),
        'log_likelihood': float(model_full.llf),
        'all_coefs': {name: float(model_full.params[name]) for name in active_features},
        'all_pvalues': {name: float(model_full.pvalues[name]) for name in active_features},
    },
    'nested_lr_test': {
        'lr_stat': float(lr_stat),
        'df': int(lr_df),
        'p': float(lr_p),
    },
    'attenuation_pct': float(attenuation * 100),
    'verdict': verdict,
    'verdict_short': verdict_short,
    'vif': final_vifs,
    'dropped_features_vif': [{'feature': f, 'vif': v} for f, v in dropped_features],
    'stratified_or': stratified_results,
    'sensitivity': sensitivity_results,
}

with open(os.path.join(RESULTS_DIR, 'sprint1_results.json'), 'w') as f:
    json.dump(results_out, f, indent=2, default=str)
print(f"  Saved: sprint1_results.json")

# ============================================================
# FINAL SUMMARY
# ============================================================
elapsed = time.time() - t_start
print("\n" + "=" * 70)
print("SPRINT 1 — FINAL SUMMARY")
print("=" * 70)
print(f"\n  Data: {int(y_all.sum())} UAP + {int(len(y_all) - y_all.sum())} control points")
print(f"  Port data: {len(port_coords)} locations ({port_source})")
print(f"  Military bases: {len(df_bases)} (full CSV)")
print(f"  Canyon cells: {len(canyon_lats)} (grad>{CANYON_GRADIENT_THRESHOLD} m/km)")
print(f"\n  ORIGINAL MODEL (v2 covariates only):")
print(f"    Canyon coef: {canyon_coef_orig:+.4f} [{canyon_ci_orig[0]:+.4f}, {canyon_ci_orig[1]:+.4f}]")
print(f"    Pseudo R²: {model_orig.prsquared:.4f}")
print(f"\n  FULL MODEL (+ port proximity, port density, coast complexity):")
print(f"    Canyon coef: {canyon_coef_full:+.4f} [{canyon_ci_full[0]:+.4f}, {canyon_ci_full[1]:+.4f}]")
print(f"    Pseudo R²: {model_full.prsquared:.4f}")
print(f"    Canyon p-value: {model_full.pvalues['dist_to_canyon_km']:.2e}")
print(f"\n  Nested LR test: chi²={lr_stat:.2f}, df={lr_df}, p={lr_p:.2e}")
print(f"  Attenuation: {attenuation * 100:.1f}%")

print(f"\n  Stratified ORs by port-density tercile:")
for s in stratified_results:
    or10 = s.get('OR_10', np.nan)
    or25 = s.get('OR_25', np.nan)
    if not np.isnan(or10):
        print(f"    {s['tercile']:<35} OR@10km={or10:.2f}, OR@25km={or25:.2f}")

print(f"\n  Sensitivity (canyon definition):")
for sr in sensitivity_results:
    coef = sr['canyon_coef']
    p = sr['canyon_p']
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns' if not np.isnan(p) else ''
    print(f"    {sr['depth_range']:>12} grad>{sr['gradient_threshold']:.0f}: "
          f"coef={coef:+.4f} [{sr['canyon_ci_lo']:+.4f},{sr['canyon_ci_hi']:+.4f}] {sig} "
          f"(n={sr['n_canyon_cells']})")

print(f"\n  {'=' * 50}")
print(f"  VERDICT: {verdict}")
print(f"  {'=' * 50}")
print(f"\n  Runtime: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
print(f"\n  Output files:")
print(f"    sprint1_results.json")
print(f"    figures/sprint1_model_comparison.png")
print(f"    figures/sprint1_sensitivity.png")
print(f"    figures/sprint1_stratified_or.png")
print(f"\n✓ Sprint 1 complete!")
