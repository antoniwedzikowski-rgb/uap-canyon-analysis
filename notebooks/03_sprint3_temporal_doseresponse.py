#!/usr/bin/env python3
"""
Sprint 3: Temporal Clustering + Dose-Response (Canyon Magnitude)
================================================================
Two independent questions using existing data:
  1. Temporal clustering: Do near-canyon UAP reports show episodic bursts
     ("flaps") beyond what their higher rate predicts?
  2. Dose-response: Do steeper canyons predict more excess UAP reports
     than shallow ones?

Both are independent of Sprints 1-2. Each can kill or strengthen the finding.

Uses Model B specification (7 covariates) as base.

Outputs:
  sprint3_results.json
  figures/sprint3_temporal_permutation.png
  figures/sprint3_temporal_sensitivity.png
  figures/sprint3_flap_map.png
  figures/sprint3_dose_response_bins.png
  figures/sprint3_dose_response_gam.png
  figures/sprint3_dose_response_stratified.png
"""

import os
import warnings
import time
import json
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from scipy.ndimage import label as ndimage_label
from scipy.stats import spearmanr, chi2 as chi2_dist
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# pyGAM with fallback
try:
    from pygam import LogisticGAM, s, l as l_term
    HAS_PYGAM = True
except ImportError:
    HAS_PYGAM = False
    print("WARNING: pyGAM not installed. GAM dose-response plot will be skipped.")

warnings.filterwarnings('ignore')
np.random.seed(42)


# ============================================================
# SECTION 0: PRE-REGISTERED ANALYSIS PLAN
# ============================================================
PLAN_TEXT = """
================================================================================
PRE-REGISTERED ANALYSIS PLAN — Sprint 3: Temporal Clustering + Dose-Response
Timestamp: {timestamp}
================================================================================

PRIMARY SPECIFICATIONS (report as main results):

  TEMPORAL:
    Spatial radius:     50 km
    Temporal window:    +/-7 days
    Canyon threshold:   25 km
    Null model:         time permutation within year (1000 iterations)
    Metric:             excess_near - excess_far (observed/expected ratio)

  DOSE-RESPONSE:
    Gradient feature:   p95_gradient_within_25km (95th percentile, not max)
    Transform:          log1p
    Model:              Model B + gradient feature (logistic regression, LR test)
    Primary plot:       OR by gradient bins (6 bins)

SENSITIVITY (report as supplementary):
    Temporal windows:   3, 14, 30 days
    Spatial radii:      25, 100 km
    Canyon thresholds:  10, 50 km
    Gradient features:  max, mean_top10%, count_steep (gradient > 50 m/km)
    BH-FDR correction across sensitivity grid

================================================================================
"""
print(PLAN_TEXT.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = "/Users/antoniwedzikowski/Desktop/UAP research"
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

R_EARTH = 6371.0
CANYON_GRADIENT_THRESHOLD = 20.0
COASTAL_BAND_KM = 200
N_CONTROL = 20000
PORT_CACHE_FILE = os.path.join(DATA_DIR, "port_coords_cache.npz")

# Part A: Temporal clustering
TEMPORAL_WINDOW_PRIMARY = 7       # days
SPATIAL_RADIUS_PRIMARY = 50       # km
CANYON_THRESHOLD_PRIMARY = 25     # km
N_PERM_PRIMARY = 1000
N_PERM_SENSITIVITY = 200
TEMPORAL_WINDOWS = [3, 7, 14, 30]
SPATIAL_RADII = [25, 50, 100]
CANYON_THRESHOLDS = [10, 25, 50]

# Part B: Dose-response
GRADIENT_RADIUS_KM = 25
GRID_RES = 0.1  # degrees for aggregated gradient grid
DOSE_BINS = [0, 0.01, 5, 10, 20, 50, 100, 500]
DOSE_BIN_LABELS = ['0 (no shelf)', '0-5', '5-10', '10-20', '20-50', '50-100', '100+']

# Performance
MAX_NEIGHBORS_CAP = 1500

print("=" * 70)
print("SPRINT 3: TEMPORAL CLUSTERING + DOSE-RESPONSE")
print("=" * 70)
t_start = time.time()


# ============================================================
# SECTION 1: UTILITY FUNCTIONS
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_padded_neighbor_matrix(spatial_neighbors, max_cap=1500):
    """
    Convert ragged list-of-arrays to dense (N, max_neighbors) int32 matrix.
    -1 = padding. Cap at max_cap neighbors per point (random subsample if exceeded).
    Returns: neighbor_matrix, n_actual_neighbors (1D array).
    """
    N = len(spatial_neighbors)
    # Remove self from each neighbor list and compute actual counts
    clean_neighbors = []
    n_actual = np.zeros(N, dtype=np.int32)
    for i in range(N):
        nb = spatial_neighbors[i]
        nb = nb[nb != i]  # exclude self
        if len(nb) > max_cap:
            nb = np.random.choice(nb, max_cap, replace=False)
        clean_neighbors.append(nb)
        n_actual[i] = len(nb)

    max_nb = int(n_actual.max()) if n_actual.max() > 0 else 1
    neighbor_matrix = np.full((N, max_nb), -1, dtype=np.int32)
    for i in range(N):
        nb = clean_neighbors[i]
        if len(nb) > 0:
            neighbor_matrix[i, :len(nb)] = nb

    return neighbor_matrix, n_actual


def vectorized_temporal_density(days_arr, neighbor_matrix, temporal_window):
    """
    Compute temporal density for all points simultaneously.
    days_arr: (N,) float64
    neighbor_matrix: (N, max_neighbors) int32, -1 = padding
    temporal_window: float (days)
    Returns: (N,) count of temporal neighbors within window.
    """
    N = len(days_arr)
    safe_indices = np.clip(neighbor_matrix, 0, N - 1)
    neighbor_days = days_arr[safe_indices]       # (N, max_neighbors)
    self_days = days_arr[:, None]                 # (N, 1)
    time_diffs = np.abs(neighbor_days - self_days)

    # Mask padding positions
    padding_mask = (neighbor_matrix == -1)
    time_diffs[padding_mask] = np.inf

    return np.sum(time_diffs <= temporal_window, axis=1).astype(float)


def compute_excess_near_minus_far(temporal_density, n_spatial, near_mask, far_mask,
                                   temporal_window, days_in_year):
    """
    Compute median(excess_near) - median(excess_far).
    excess_i = observed_neighbors / expected_neighbors
    expected_i = n_spatial_i * (2 * window / days_in_year_i)
    """
    expected = n_spatial * (2 * temporal_window / days_in_year)

    valid = expected > 0
    excess = np.full(len(temporal_density), np.nan)
    excess[valid] = temporal_density[valid] / expected[valid]

    excess_near = excess[near_mask & valid]
    excess_far = excess[far_mask & valid]

    if len(excess_near) == 0 or len(excess_far) == 0:
        return np.nan

    return np.nanmedian(excess_near) - np.nanmedian(excess_far)


def compute_excess_trimmed_mean(temporal_density, n_spatial, near_mask, far_mask,
                                 temporal_window, days_in_year, trim_pct=5):
    """
    Like compute_excess_near_minus_far but uses trimmed mean (5-95%) instead of median.
    More sensitive to heavy-tail structure (flaps).
    """
    expected = n_spatial * (2 * temporal_window / days_in_year)
    valid = expected > 0
    excess = np.full(len(temporal_density), np.nan)
    excess[valid] = temporal_density[valid] / expected[valid]

    excess_near = excess[near_mask & valid]
    excess_far = excess[far_mask & valid]

    if len(excess_near) < 10 or len(excess_far) < 10:
        return np.nan

    from scipy.stats import trim_mean
    return trim_mean(excess_near[~np.isnan(excess_near)], trim_pct / 100.0) - \
           trim_mean(excess_far[~np.isnan(excess_far)], trim_pct / 100.0)


def compute_excess_heavy_tail(temporal_density, n_spatial, near_mask, far_mask,
                               threshold=3):
    """
    Alternative metric: proportion of points with temporal_density >= threshold.
    Tests whether near-canyon has more "heavy tail" clustering.
    """
    frac_near = np.mean(temporal_density[near_mask] >= threshold)
    frac_far = np.mean(temporal_density[far_mask] >= threshold)
    return frac_near - frac_far


def get_p95_gradient_gridded(points, gradient_by_cell, grid_res=0.1, radius_km=25):
    """
    For each point, gather gradient values from nearby grid cells within radius,
    return 95th percentile. Returns 0 if no shelf cells nearby.
    """
    radius_deg = radius_km / 111.0
    p95_gradients = np.zeros(len(points))

    for i in range(len(points)):
        lat, lon = points[i]
        all_grads = []
        for dlat in np.arange(-radius_deg, radius_deg + grid_res, grid_res):
            for dlon in np.arange(-radius_deg, radius_deg + grid_res, grid_res):
                cell = (round((lat + dlat) / grid_res) * grid_res,
                        round((lon + dlon) / grid_res) * grid_res)
                if cell in gradient_by_cell:
                    all_grads.extend(gradient_by_cell[cell])

        if len(all_grads) > 0:
            p95_gradients[i] = np.percentile(all_grads, 95)

        if (i + 1) % 10000 == 0:
            print(f"    Gradient p95 lookup: {i+1}/{len(points)}")

    return p95_gradients


def get_gradient_feature(points, gradient_by_cell, grid_res=0.1, radius_km=25,
                          mode='p95'):
    """
    Generalized gradient feature lookup.
    mode: 'p95', 'max', 'mean_top10', 'count_steep_50'
    """
    radius_deg = radius_km / 111.0
    result = np.zeros(len(points))

    for i in range(len(points)):
        lat, lon = points[i]
        all_grads = []
        for dlat in np.arange(-radius_deg, radius_deg + grid_res, grid_res):
            for dlon in np.arange(-radius_deg, radius_deg + grid_res, grid_res):
                cell = (round((lat + dlat) / grid_res) * grid_res,
                        round((lon + dlon) / grid_res) * grid_res)
                if cell in gradient_by_cell:
                    all_grads.extend(gradient_by_cell[cell])

        if len(all_grads) > 0:
            if mode == 'p95':
                result[i] = np.percentile(all_grads, 95)
            elif mode == 'max':
                result[i] = np.max(all_grads)
            elif mode == 'mean_top10':
                sorted_g = np.sort(all_grads)
                top_n = max(1, len(sorted_g) // 10)
                result[i] = np.mean(sorted_g[-top_n:])
            elif mode == 'count_steep_50':
                result[i] = np.sum(np.array(all_grads) > 50)

        if (i + 1) % 10000 == 0:
            print(f"    Gradient {mode} lookup: {i+1}/{len(points)}")

    return result


# ============================================================
# SECTION 2: DATA LOADING
# ============================================================
print("\n[SECTION 2] Loading data...")

nuforc_cols = ['datetime', 'city', 'state', 'country', 'shape', 'duration_sec',
               'duration_text', 'comments', 'date_posted', 'latitude', 'longitude']
df_raw = pd.read_csv(os.path.join(DATA_DIR, "nuforc_reports.csv"),
                      names=nuforc_cols, header=None, low_memory=False)

df_raw['latitude'] = pd.to_numeric(df_raw['latitude'], errors='coerce')
df_raw['longitude'] = pd.to_numeric(df_raw['longitude'], errors='coerce')
df = df_raw.dropna(subset=['latitude', 'longitude']).copy()
df = df[(df['latitude'] != 0) & (df['longitude'] != 0)]
df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]
df = df.drop_duplicates(subset=['latitude', 'longitude', 'datetime'])

df_us = df[(df['latitude'] >= 20) & (df['latitude'] <= 55) &
           (df['longitude'] >= -135) & (df['longitude'] <= -55)].copy()
print(f"  US reports: {len(df_us)}")

# ETOPO
ds = xr.open_dataset(os.path.join(DATA_DIR, "etopo_subset.nc"))
elev = ds['z'].values
elev_lats = ds['lat'].values
elev_lons = ds['lon'].values

# Counties
df_counties = pd.read_csv(os.path.join(DATA_DIR, "county_centroids_pop.csv"))
counties_lat = df_counties['lat'].values
counties_lon = df_counties['lon'].values
counties_pop = df_counties['pop'].values
county_tree = cKDTree(np.column_stack([counties_lat, counties_lon]))

# Military
df_bases = pd.read_csv(os.path.join(DATA_DIR, "military_bases_us.csv"))
bases_lat = df_bases['lat'].values
bases_lon = df_bases['lon'].values
base_tree = cKDTree(np.column_stack([bases_lat, bases_lon]))

print(f"  Data loaded ({time.time() - t_start:.1f}s)")

# ============================================================
# SECTION 3: COASTLINE & CANYON DETECTION
# ============================================================
print("\n[SECTION 3] Coastline & canyon detection...")

ocean_mask = elev < 0
land_mask = elev >= 0

ocean_rows, ocean_cols = np.where(ocean_mask)
ocean_lats = elev_lats[ocean_rows]
ocean_lons = elev_lons[ocean_cols]
ocean_depths = elev[ocean_rows, ocean_cols]
ocean_tree = cKDTree(np.column_stack([ocean_lats, ocean_lons]))

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

# Canyon detection
shelf_mask = (elev < 0) & (elev > -500)
lat_res_km = np.abs(np.diff(elev_lats).mean()) * 111.0
mid_lat = 37.0
lon_res_km = np.abs(np.diff(elev_lons).mean()) * 111.0 * np.cos(np.radians(mid_lat))

grad_y, grad_x = np.gradient(elev.astype(float))
grad_mag = np.sqrt((grad_y / lat_res_km) ** 2 + (grad_x / lon_res_km) ** 2)

shelf_canyon_mask = shelf_mask & (grad_mag > CANYON_GRADIENT_THRESHOLD)
labeled, n_features = ndimage_label(shelf_canyon_mask)
canyon_sizes = np.bincount(labeled.ravel())
for sl in np.where(canyon_sizes < 3)[0]:
    shelf_canyon_mask[labeled == sl] = False

canyon_rows, canyon_cols = np.where(shelf_canyon_mask)
canyon_lats = elev_lats[canyon_rows]
canyon_lons = elev_lons[canyon_cols]
canyon_tree = cKDTree(np.column_stack([canyon_lats, canyon_lons]))
print(f"  Canyon cells: {len(canyon_lats)}")

# Retain shelf_mask and grad_mag for Part B gradient computation
# (Sprint 2 doesn't need these after canyon detection, but Sprint 3 does)

print(f"  Section 3 done ({time.time() - t_start:.1f}s)")

# ============================================================
# SECTION 4: COASTAL FILTERING & UAP METRICS
# ============================================================
print("\n[SECTION 4] Coastal filtering & UAP metrics...")

uap_lats = df_us['latitude'].values
uap_lons = df_us['longitude'].values

coast_dists_deg, coast_idxs = coast_tree.query(
    np.column_stack([uap_lats, uap_lons]), k=1)
df_us['dist_to_coast_km'] = haversine_km(
    uap_lats, uap_lons, coast_lats[coast_idxs], coast_lons[coast_idxs])

df_coastal = df_us[df_us['dist_to_coast_km'] <= COASTAL_BAND_KM].copy()
coastal_lats = df_coastal['latitude'].values
coastal_lons = df_coastal['longitude'].values
print(f"  Coastal reports: {len(df_coastal)}")

# UAP metrics
_, ocean_idxs = ocean_tree.query(
    np.column_stack([coastal_lats, coastal_lons]), k=1)
df_coastal['depth_nearest_ocean'] = ocean_depths[ocean_idxs]

_, canyon_idxs = canyon_tree.query(
    np.column_stack([coastal_lats, coastal_lons]), k=1)
df_coastal['dist_to_canyon_km'] = haversine_km(
    coastal_lats, coastal_lons, canyon_lats[canyon_idxs], canyon_lons[canyon_idxs])

base_dists_deg, _ = base_tree.query(
    np.column_stack([coastal_lats, coastal_lons]), k=1)
df_coastal['dist_to_military_km'] = base_dists_deg * 111.0

uap_county_dists, uap_county_idx = county_tree.query(
    np.column_stack([coastal_lats, coastal_lons]), k=5)
uap_pop = np.zeros(len(coastal_lats))
for k in range(5):
    d_km = uap_county_dists[:, k] * 111.0 + 1.0
    uap_pop += counties_pop[uap_county_idx[:, k]] / (d_km ** 2)
df_coastal['pop_density_proxy'] = uap_pop

print(f"  UAP metrics done ({time.time() - t_start:.1f}s)")

# ============================================================
# SECTION 5: CONTROL POINTS & METRICS
# ============================================================
print("\n[SECTION 5] Control points...")

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

grid_county_dists, grid_county_idx = county_tree.query(
    np.column_stack([glat_coastal, glon_coastal]), k=10)
weights = np.zeros(len(glat_coastal))
for k in range(10):
    d_km = grid_county_dists[:, k] * 111.0 + 1.0
    weights += counties_pop[grid_county_idx[:, k]] / (d_km ** 2)

lat_idx = np.clip(np.searchsorted(elev_lats, glat_coastal), 0, len(elev_lats) - 1)
lon_idx = np.clip(np.searchsorted(elev_lons, glon_coastal), 0, len(elev_lons) - 1)
grid_elev = elev[lat_idx, lon_idx]
land_weight = np.where(grid_elev >= 0, 3.0, 0.05)
weights *= land_weight
weights = weights / weights.sum()

chosen = np.random.choice(len(glat_coastal), size=N_CONTROL, p=weights, replace=True)
jitter_val = 0.12
ctrl_lats = glat_coastal[chosen] + np.random.uniform(-jitter_val, jitter_val, N_CONTROL)
ctrl_lons = glon_coastal[chosen] + np.random.uniform(-jitter_val, jitter_val, N_CONTROL)

# Filter controls to coastal band
_, ctrl_coast_idx = coast_tree.query(
    np.column_stack([ctrl_lats, ctrl_lons]), k=1)
ctrl_coast_km = haversine_km(ctrl_lats, ctrl_lons,
                              coast_lats[ctrl_coast_idx], coast_lons[ctrl_coast_idx])
ctrl_mask = ctrl_coast_km <= COASTAL_BAND_KM
ctrl_lats = ctrl_lats[ctrl_mask]
ctrl_lons = ctrl_lons[ctrl_mask]
ctrl_coast_km = ctrl_coast_km[ctrl_mask]
N_CONTROL = len(ctrl_lats)
print(f"  Control points: {N_CONTROL}")

# Control metrics
_, ctrl_ocean_idx = ocean_tree.query(
    np.column_stack([ctrl_lats, ctrl_lons]), k=1)
ctrl_depths = ocean_depths[ctrl_ocean_idx]

_, ctrl_canyon_idx = canyon_tree.query(
    np.column_stack([ctrl_lats, ctrl_lons]), k=1)
ctrl_canyon_km = haversine_km(ctrl_lats, ctrl_lons,
                               canyon_lats[ctrl_canyon_idx], canyon_lons[ctrl_canyon_idx])

ctrl_base_dists, _ = base_tree.query(
    np.column_stack([ctrl_lats, ctrl_lons]), k=1)
ctrl_base_km = ctrl_base_dists * 111.0

ctrl_county_dists, ctrl_county_idx = county_tree.query(
    np.column_stack([ctrl_lats, ctrl_lons]), k=5)
ctrl_pop = np.zeros(N_CONTROL)
for k in range(5):
    d_km = ctrl_county_dists[:, k] * 111.0 + 1.0
    ctrl_pop += counties_pop[ctrl_county_idx[:, k]] / (d_km ** 2)

print(f"  Control metrics done ({time.time() - t_start:.1f}s)")

# ============================================================
# SECTION 6: PORT DATA & FEATURE ASSEMBLY
# ============================================================
print("\n[SECTION 6] Port data & feature assembly...")

if not os.path.exists(PORT_CACHE_FILE):
    raise FileNotFoundError(f"Port cache not found: {PORT_CACHE_FILE}")

data = np.load(PORT_CACHE_FILE, allow_pickle=True)
port_coords = data['port_coords']
port_source = str(data['source'])
print(f"  Loaded {len(port_coords)} ports ({port_source})")

port_coords_rad = np.radians(port_coords)
port_tree = BallTree(port_coords_rad, metric='haversine')

# UAP port metrics
uap_coords_rad = np.radians(np.column_stack([coastal_lats, coastal_lons]))
port_dists_rad, _ = port_tree.query(uap_coords_rad, k=1)
df_coastal['dist_to_nearest_port'] = port_dists_rad.flatten() * R_EARTH
counts_25_uap = port_tree.query_radius(uap_coords_rad, r=25.0 / R_EARTH, count_only=True)
df_coastal['log_port_count_25km'] = np.log1p(counts_25_uap)

# Control port metrics
ctrl_coords_rad = np.radians(np.column_stack([ctrl_lats, ctrl_lons]))
ctrl_port_dists_rad, _ = port_tree.query(ctrl_coords_rad, k=1)
ctrl_port_km = ctrl_port_dists_rad.flatten() * R_EARTH
ctrl_port_count_25 = port_tree.query_radius(ctrl_coords_rad, r=25.0 / R_EARTH, count_only=True)
ctrl_log_port_count_25 = np.log1p(ctrl_port_count_25)

# --- Assemble Model B feature matrix ---
feat_B = ['dist_to_canyon_km', 'dist_to_coast_km', 'dist_to_military_km',
          'pop_density_proxy', 'depth_nearest_ocean',
          'dist_to_nearest_port', 'log_port_count_25km']

X_uap_raw = pd.DataFrame({
    'dist_to_canyon_km': df_coastal['dist_to_canyon_km'].values,
    'dist_to_coast_km': df_coastal['dist_to_coast_km'].values,
    'dist_to_military_km': df_coastal['dist_to_military_km'].values,
    'pop_density_proxy': df_coastal['pop_density_proxy'].values,
    'depth_nearest_ocean': df_coastal['depth_nearest_ocean'].values,
    'dist_to_nearest_port': df_coastal['dist_to_nearest_port'].values,
    'log_port_count_25km': df_coastal['log_port_count_25km'].values,
})

X_ctrl_raw = pd.DataFrame({
    'dist_to_canyon_km': ctrl_canyon_km,
    'dist_to_coast_km': ctrl_coast_km,
    'dist_to_military_km': ctrl_base_km,
    'pop_density_proxy': ctrl_pop,
    'depth_nearest_ocean': ctrl_depths,
    'dist_to_nearest_port': ctrl_port_km,
    'log_port_count_25km': ctrl_log_port_count_25,
})

X_raw_pre = pd.concat([X_uap_raw, X_ctrl_raw], ignore_index=True)
y_raw_pre = np.concatenate([np.ones(len(X_uap_raw)), np.zeros(len(X_ctrl_raw))])

# Track valid mask BEFORE dropping NaN — needed to align gradient arrays later
valid_mask = X_raw_pre.notna().all(axis=1)
n_dropped = (~valid_mask).sum()
print(f"  NaN rows dropped: {n_dropped}")

X_raw = X_raw_pre[valid_mask].reset_index(drop=True)
y_raw = y_raw_pre[valid_mask.values]

n_uap = int(y_raw.sum())
n_ctrl = len(y_raw) - n_uap
N_total = len(y_raw)
print(f"  Full sample: {N_total} (UAP={n_uap}, Ctrl={n_ctrl})")

# Store raw port counts for Part B stratification (before NaN drop alignment)
port_count_all_pre = np.concatenate([counts_25_uap.astype(float),
                                      ctrl_port_count_25.astype(float)])
port_count_all = port_count_all_pre[valid_mask.values]

# Fit reference Model B
scaler = StandardScaler()
X_z_all = scaler.fit_transform(X_raw[feat_B].values)
X_z_const = np.column_stack([np.ones(N_total), X_z_all])
model_B_ref = sm.Logit(y_raw, X_z_const).fit(disp=0, maxiter=1000)
ref_beta = model_B_ref.params[1]
ref_llf = model_B_ref.llf
print(f"  Model B reference: canyon_beta={ref_beta:.4f}, p={model_B_ref.pvalues[1]:.2e}")

print(f"\n  Feature assembly done ({time.time() - t_start:.1f}s)")


# ============================================================
# SECTION 7: PART A — TEMPORAL PARSING & GROUP DEFINITION
# ============================================================
print("\n" + "=" * 70)
print("PART A: TEMPORAL CLUSTERING")
print("=" * 70)
t_partA = time.time()

print("\n[SECTION 7] Temporal parsing & groups...")

# Parse datetimes from the coastal UAP DataFrame
df_coastal['datetime_parsed'] = pd.to_datetime(
    df_coastal['datetime'], format='mixed', dayfirst=False, errors='coerce'
)
temporal_valid = df_coastal['datetime_parsed'].notna()
n_temporal_dropped = (~temporal_valid).sum()
print(f"  Datetime parse failures: {n_temporal_dropped} "
      f"({n_temporal_dropped/len(df_coastal)*100:.1f}%)")

df_temporal = df_coastal[temporal_valid].copy()
temporal_lats = df_temporal['latitude'].values
temporal_lons = df_temporal['longitude'].values
N_temporal = len(df_temporal)

# Day since epoch for fast arithmetic
uap_days = (df_temporal['datetime_parsed'] - pd.Timestamp('2000-01-01')).dt.total_seconds() / 86400.0
uap_days = uap_days.values
uap_years = df_temporal['datetime_parsed'].dt.year.values

print(f"  Date range: {df_temporal['datetime_parsed'].min()} to "
      f"{df_temporal['datetime_parsed'].max()}")
print(f"  Reports per year: min={pd.Series(uap_years).value_counts().min()}, "
      f"max={pd.Series(uap_years).value_counts().max()}")
print(f"  Temporal UAP reports: {N_temporal}")

# Canyon distance for temporal subset
temporal_canyon_dist = df_temporal['dist_to_canyon_km'].values
near_mask_primary = temporal_canyon_dist < CANYON_THRESHOLD_PRIMARY
far_mask_primary = ~near_mask_primary
print(f"  Near-canyon (< {CANYON_THRESHOLD_PRIMARY} km): {near_mask_primary.sum()}")
print(f"  Far-canyon (>= {CANYON_THRESHOLD_PRIMARY} km): {far_mask_primary.sum()}")

# Build coordinate array for BallTree
uap_coords_temporal = np.column_stack([temporal_lats, temporal_lons])

# Year groups for within-year permutation
year_groups = {}
for yr in np.unique(uap_years):
    year_groups[yr] = np.where(uap_years == yr)[0]

# Days per year for each report
days_in_year = np.array([
    366 if (yr % 4 == 0 and (yr % 100 != 0 or yr % 400 == 0)) else 365
    for yr in uap_years
], dtype=float)

print(f"  Section 7 done ({time.time() - t_start:.1f}s)")


# ============================================================
# SECTION 8: PART A — PERMUTATION TEST (PRIMARY)
# ============================================================
print("\n[SECTION 8] Primary permutation test...")

# Step 1: Spatial neighbors at primary radius
print(f"  Computing spatial neighbors at {SPATIAL_RADIUS_PRIMARY} km...")
uap_tree_temporal = BallTree(np.radians(uap_coords_temporal), metric='haversine')
spatial_radius_rad = SPATIAL_RADIUS_PRIMARY / R_EARTH
spatial_neighbors_primary = uap_tree_temporal.query_radius(
    np.radians(uap_coords_temporal), r=spatial_radius_rad
)
print(f"  BallTree query done ({time.time() - t_start:.1f}s)")

# Step 2: Build padded neighbor matrix
print(f"  Building padded neighbor matrix (cap={MAX_NEIGHBORS_CAP})...")
neighbor_matrix_primary, n_actual_primary = build_padded_neighbor_matrix(
    spatial_neighbors_primary, max_cap=MAX_NEIGHBORS_CAP
)
max_nb_primary = neighbor_matrix_primary.shape[1]
print(f"  Padded matrix: {neighbor_matrix_primary.shape}, "
      f"memory: {neighbor_matrix_primary.nbytes / 1e6:.0f} MB")
print(f"  Max neighbors: {max_nb_primary}, mean: {n_actual_primary.mean():.1f}, "
      f"median: {np.median(n_actual_primary):.0f}")

# Step 3: Observed temporal density
observed_td = vectorized_temporal_density(
    uap_days, neighbor_matrix_primary, TEMPORAL_WINDOW_PRIMARY
)
print(f"  Observed temporal density: near mean={observed_td[near_mask_primary].mean():.2f}, "
      f"far mean={observed_td[far_mask_primary].mean():.2f}")

# Step 4: Observed statistic
observed_diff = compute_excess_near_minus_far(
    observed_td, n_actual_primary.astype(float), near_mask_primary, far_mask_primary,
    TEMPORAL_WINDOW_PRIMARY, days_in_year
)
print(f"  Observed excess difference (near - far): {observed_diff:.4f}")

# Step 5: Permutation loop
print(f"\n  Running {N_PERM_PRIMARY} permutations (within-year time shuffle)...")
t_perm = time.time()
perm_diffs = np.empty(N_PERM_PRIMARY)

for p in range(N_PERM_PRIMARY):
    perm_days = uap_days.copy()
    for yr, indices in year_groups.items():
        perm_days[indices] = np.random.permutation(perm_days[indices])

    td_perm = vectorized_temporal_density(perm_days, neighbor_matrix_primary,
                                           TEMPORAL_WINDOW_PRIMARY)
    perm_diffs[p] = compute_excess_near_minus_far(
        td_perm, n_actual_primary.astype(float), near_mask_primary, far_mask_primary,
        TEMPORAL_WINDOW_PRIMARY, days_in_year
    )

    if (p + 1) % 100 == 0:
        elapsed = time.time() - t_perm
        rate = (p + 1) / elapsed
        remaining = (N_PERM_PRIMARY - p - 1) / rate
        print(f"    Permutation {p+1}/{N_PERM_PRIMARY} "
              f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

# Corrected p-value: (k+1)/(N+1) to avoid p=0 artifact
k_exceeded = np.sum(perm_diffs >= observed_diff)
p_value_perm = (k_exceeded + 1) / (N_PERM_PRIMARY + 1)
z_score_perm = (observed_diff - perm_diffs.mean()) / perm_diffs.std() if perm_diffs.std() > 0 else np.nan

print(f"\n  Permutation test: observed = {observed_diff:.4f}, "
      f"p = {p_value_perm:.4f} ({N_PERM_PRIMARY} permutations, "
      f"k={k_exceeded} exceeded)")
print(f"  Permutation null: mean = {perm_diffs.mean():.4f}, "
      f"std = {perm_diffs.std():.4f}, "
      f"95th pct = {np.percentile(perm_diffs, 95):.4f}")
print(f"  Z-score: {z_score_perm:.1f} (observed vs null)")
print(f"  Primary permutation done ({time.time() - t_perm:.1f}s)")


# ============================================================
# SECTION 8b: ROBUSTNESS — ALTERNATIVE METRICS + WITHIN-MONTH NULL
# ============================================================
print("\n[SECTION 8b] Robustness checks...")
t_robust = time.time()

# --- Robustness 1: Trimmed mean (5-95%) metric ---
print("  Robustness 1: Trimmed mean (5-95%) metric...")

observed_diff_trimmed = compute_excess_trimmed_mean(
    observed_td, n_actual_primary.astype(float), near_mask_primary, far_mask_primary,
    TEMPORAL_WINDOW_PRIMARY, days_in_year, trim_pct=5
)

perm_diffs_trimmed = np.empty(N_PERM_SENSITIVITY)
for p in range(N_PERM_SENSITIVITY):
    perm_days = uap_days.copy()
    for yr, indices in year_groups.items():
        perm_days[indices] = np.random.permutation(perm_days[indices])
    td_perm = vectorized_temporal_density(perm_days, neighbor_matrix_primary,
                                           TEMPORAL_WINDOW_PRIMARY)
    perm_diffs_trimmed[p] = compute_excess_trimmed_mean(
        td_perm, n_actual_primary.astype(float), near_mask_primary, far_mask_primary,
        TEMPORAL_WINDOW_PRIMARY, days_in_year, trim_pct=5
    )

k_trimmed = np.sum(perm_diffs_trimmed >= observed_diff_trimmed)
p_trimmed = (k_trimmed + 1) / (N_PERM_SENSITIVITY + 1)
z_trimmed = ((observed_diff_trimmed - perm_diffs_trimmed.mean()) / perm_diffs_trimmed.std()
             if perm_diffs_trimmed.std() > 0 else np.nan)
print(f"    Observed (trimmed mean): {observed_diff_trimmed:.4f}, "
      f"p = {p_trimmed:.4f} ({N_PERM_SENSITIVITY} perms), z = {z_trimmed:.1f}")

# --- Robustness 1b: Heavy tail metric (frac with td >= 3) ---
print("  Robustness 1b: Heavy tail metric (frac with td >= 3)...")

observed_diff_tail = compute_excess_heavy_tail(
    observed_td, n_actual_primary.astype(float), near_mask_primary, far_mask_primary,
    threshold=3
)

perm_diffs_tail = np.empty(N_PERM_SENSITIVITY)
for p in range(N_PERM_SENSITIVITY):
    perm_days = uap_days.copy()
    for yr, indices in year_groups.items():
        perm_days[indices] = np.random.permutation(perm_days[indices])
    td_perm = vectorized_temporal_density(perm_days, neighbor_matrix_primary,
                                           TEMPORAL_WINDOW_PRIMARY)
    perm_diffs_tail[p] = compute_excess_heavy_tail(
        td_perm, n_actual_primary.astype(float), near_mask_primary, far_mask_primary,
        threshold=3
    )

k_tail = np.sum(perm_diffs_tail >= observed_diff_tail)
p_tail = (k_tail + 1) / (N_PERM_SENSITIVITY + 1)
z_tail = ((observed_diff_tail - perm_diffs_tail.mean()) / perm_diffs_tail.std()
          if perm_diffs_tail.std() > 0 else np.nan)
print(f"    Observed (heavy tail frac): {observed_diff_tail:.4f}, "
      f"p = {p_tail:.4f} ({N_PERM_SENSITIVITY} perms), z = {z_tail:.1f}")

# --- Robustness 2: Within-month permutation null ---
print("  Robustness 2: Within-month permutation null (preserves seasonality)...")

# Build month groups
uap_months = df_temporal['datetime_parsed'].dt.month.values
month_year_groups = {}
for yr in np.unique(uap_years):
    for mo in range(1, 13):
        indices = np.where((uap_years == yr) & (uap_months == mo))[0]
        if len(indices) > 1:
            month_year_groups[(yr, mo)] = indices

perm_diffs_monthly = np.empty(N_PERM_SENSITIVITY)
for p in range(N_PERM_SENSITIVITY):
    perm_days = uap_days.copy()
    for (yr, mo), indices in month_year_groups.items():
        perm_days[indices] = np.random.permutation(perm_days[indices])

    td_perm = vectorized_temporal_density(perm_days, neighbor_matrix_primary,
                                           TEMPORAL_WINDOW_PRIMARY)
    perm_diffs_monthly[p] = compute_excess_near_minus_far(
        td_perm, n_actual_primary.astype(float), near_mask_primary, far_mask_primary,
        TEMPORAL_WINDOW_PRIMARY, days_in_year
    )

k_monthly = np.sum(perm_diffs_monthly >= observed_diff)
p_monthly = (k_monthly + 1) / (N_PERM_SENSITIVITY + 1)
z_monthly = ((observed_diff - perm_diffs_monthly.mean()) / perm_diffs_monthly.std()
             if perm_diffs_monthly.std() > 0 else np.nan)
print(f"    Within-month null: observed = {observed_diff:.4f}, "
      f"p = {p_monthly:.4f} ({N_PERM_SENSITIVITY} perms), z = {z_monthly:.1f}")
print(f"    Monthly null: mean = {perm_diffs_monthly.mean():.4f}, "
      f"std = {perm_diffs_monthly.std():.4f}")

# Store robustness results
robustness_results = {
    'trimmed_mean_5_95': {
        'observed': float(observed_diff_trimmed),
        'p': float(p_trimmed),
        'z': float(z_trimmed) if not np.isnan(z_trimmed) else None,
        'null_mean': float(perm_diffs_trimmed.mean()),
        'null_std': float(perm_diffs_trimmed.std()),
        'n_perms': N_PERM_SENSITIVITY,
    },
    'heavy_tail_frac_ge3': {
        'observed': float(observed_diff_tail),
        'p': float(p_tail),
        'z': float(z_tail) if not np.isnan(z_tail) else None,
        'null_mean': float(perm_diffs_tail.mean()),
        'null_std': float(perm_diffs_tail.std()),
        'n_perms': N_PERM_SENSITIVITY,
    },
    'within_month_null': {
        'observed': float(observed_diff),
        'p': float(p_monthly),
        'z': float(z_monthly) if not np.isnan(z_monthly) else None,
        'null_mean': float(perm_diffs_monthly.mean()),
        'null_std': float(perm_diffs_monthly.std()),
        'n_perms': N_PERM_SENSITIVITY,
    },
}

print(f"\n  Robustness summary:")
print(f"    Primary (median, within-year): p = {p_value_perm:.4f}, z = {z_score_perm:.1f}")
print(f"    Trimmed mean (5-95%):          p = {p_trimmed:.4f}, z = {z_trimmed:.1f}")
print(f"    Heavy tail (frac td>=3):       p = {p_tail:.4f}, z = {z_tail:.1f}")
print(f"    Within-month null:             p = {p_monthly:.4f}, z = {z_monthly:.1f}")
print(f"  Robustness done ({time.time() - t_robust:.1f}s)")


# ============================================================
# SECTION 9: PART A — SENSITIVITY GRID + BH-FDR
# ============================================================
print("\n[SECTION 9] Sensitivity grid...")
t_sens = time.time()

# Pre-compute spatial neighbors for all 3 radii
print("  Pre-computing spatial neighbors for sensitivity radii...")
spatial_data = {}
for sr in SPATIAL_RADII:
    sr_rad = sr / R_EARTH
    print(f"    Radius {sr} km...")
    if sr == SPATIAL_RADIUS_PRIMARY:
        # Reuse primary
        spatial_data[sr] = (neighbor_matrix_primary, n_actual_primary)
    else:
        sn = uap_tree_temporal.query_radius(
            np.radians(uap_coords_temporal), r=sr_rad
        )
        nm, na = build_padded_neighbor_matrix(sn, max_cap=MAX_NEIGHBORS_CAP)
        spatial_data[sr] = (nm, na)
    nm_s, na_s = spatial_data[sr]
    print(f"      Shape: {nm_s.shape}, mean neighbors: {na_s.mean():.1f}")

# Sensitivity loop
sensitivity_results = {}
total_combos = len(TEMPORAL_WINDOWS) * len(SPATIAL_RADII) * len(CANYON_THRESHOLDS)
combo_i = 0

for tw in TEMPORAL_WINDOWS:
    for sr in SPATIAL_RADII:
        for ct in CANYON_THRESHOLDS:
            combo_i += 1

            if tw == TEMPORAL_WINDOW_PRIMARY and sr == SPATIAL_RADIUS_PRIMARY and ct == CANYON_THRESHOLD_PRIMARY:
                # Use primary result
                sensitivity_results[(tw, sr, ct)] = {
                    'p': float(p_value_perm),
                    'observed_diff': float(observed_diff),
                    'is_primary': True,
                    'n_perms': N_PERM_PRIMARY
                }
                print(f"  [{combo_i}/{total_combos}] tw={tw}, sr={sr}, ct={ct} — PRIMARY (cached)")
                continue

            nm, na = spatial_data[sr]
            near_s = temporal_canyon_dist < ct
            far_s = ~near_s

            if near_s.sum() < 50 or far_s.sum() < 50:
                sensitivity_results[(tw, sr, ct)] = {
                    'p': np.nan,
                    'observed_diff': np.nan,
                    'is_primary': False,
                    'n_perms': 0,
                    'note': f'Insufficient group size (near={near_s.sum()}, far={far_s.sum()})'
                }
                print(f"  [{combo_i}/{total_combos}] tw={tw}, sr={sr}, ct={ct} — SKIPPED (small group)")
                continue

            # Observed
            obs_td_s = vectorized_temporal_density(uap_days, nm, tw)
            obs_diff_s = compute_excess_near_minus_far(
                obs_td_s, na.astype(float), near_s, far_s, tw, days_in_year
            )

            # Permutations
            perm_d_s = np.empty(N_PERM_SENSITIVITY)
            for p in range(N_PERM_SENSITIVITY):
                pd_s = uap_days.copy()
                for yr, indices in year_groups.items():
                    pd_s[indices] = np.random.permutation(pd_s[indices])
                td_p_s = vectorized_temporal_density(pd_s, nm, tw)
                perm_d_s[p] = compute_excess_near_minus_far(
                    td_p_s, na.astype(float), near_s, far_s, tw, days_in_year
                )

            k_s = np.sum(perm_d_s >= obs_diff_s)
            p_val_s = (k_s + 1) / (N_PERM_SENSITIVITY + 1)
            sensitivity_results[(tw, sr, ct)] = {
                'p': float(p_val_s),
                'observed_diff': float(obs_diff_s),
                'is_primary': False,
                'n_perms': N_PERM_SENSITIVITY
            }
            print(f"  [{combo_i}/{total_combos}] tw={tw}, sr={sr}, ct={ct} — "
                  f"diff={obs_diff_s:.4f}, p={p_val_s:.4f}")

# BH-FDR correction
keys_ordered = sorted(sensitivity_results.keys())
all_pvals = []
for k in keys_ordered:
    pv = sensitivity_results[k]['p']
    all_pvals.append(pv if not np.isnan(pv) else 1.0)

_, fdr_pvals, _, _ = multipletests(all_pvals, method='fdr_bh')
for i, k in enumerate(keys_ordered):
    sensitivity_results[k]['fdr_p'] = float(fdr_pvals[i])

n_significant_fdr = sum(1 for fp in fdr_pvals if fp < 0.05)
n_same_sign = sum(1 for v in sensitivity_results.values()
                   if not np.isnan(v['observed_diff']) and
                   np.sign(v['observed_diff']) == np.sign(observed_diff))

print(f"\n  Sensitivity: {n_same_sign}/{len(sensitivity_results)} same sign as primary")
print(f"  Sensitivity: {n_significant_fdr}/{len(sensitivity_results)} significant after BH-FDR at 0.05")
print(f"  Sensitivity grid done ({time.time() - t_sens:.1f}s)")


# ============================================================
# SECTION 10: PART A — FLAP EPISODES (DESCRIPTIVE) + FIGURES
# ============================================================
print("\n[SECTION 10] Flap episodes (descriptive) + Part A figures...")

# --- Flap episode identification ---
flap_episodes = []
nonzero_td = observed_td[observed_td > 0]
if len(nonzero_td) > 0:
    flap_threshold = np.percentile(nonzero_td, 95)
    flap_mask = (observed_td >= flap_threshold) & near_mask_primary
    n_flap = flap_mask.sum()
    print(f"  N reports in top-5% temporal density near canyons: {n_flap}")

    if n_flap >= 10:
        flap_coords = uap_coords_temporal[flap_mask]
        flap_times = uap_days[flap_mask]
        flap_canyon_dists = temporal_canyon_dist[flap_mask]

        # DBSCAN on spatiotemporal features
        flap_features = np.column_stack([
            flap_coords[:, 0],
            flap_coords[:, 1],
            flap_times / 7.0 * 0.45  # 7 days ~ 50 km in clustering space
        ])

        clustering = DBSCAN(eps=0.5, min_samples=3).fit(flap_features)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  Distinct flap episodes (DBSCAN): {n_clusters}")

        for label_id in range(min(n_clusters, 5)):
            members = np.where(labels == label_id)[0]
            lats_ep = flap_coords[members, 0]
            lons_ep = flap_coords[members, 1]
            times_ep = pd.to_datetime(
                flap_times[members], unit='D', origin='2000-01-01'
            )
            canyon_dists_ep = flap_canyon_dists[members]
            episode = {
                'id': label_id + 1,
                'n_reports': int(len(members)),
                'lat_mean': float(lats_ep.mean()),
                'lon_mean': float(lons_ep.mean()),
                'time_start': str(times_ep.min().date()),
                'time_end': str(times_ep.max().date()),
                'nearest_canyon_km': float(canyon_dists_ep.min()),
            }
            flap_episodes.append(episode)
            print(f"    Episode #{label_id+1}: {len(members)} reports, "
                  f"loc={lats_ep.mean():.2f}N/{lons_ep.mean():.2f}W, "
                  f"{times_ep.min().date()} to {times_ep.max().date()}, "
                  f"nearest canyon: {canyon_dists_ep.min():.1f} km")
    else:
        print(f"  Insufficient flap reports ({n_flap}) for DBSCAN")
        n_clusters = 0
else:
    print("  No non-zero temporal density found")
    n_clusters = 0
    flap_threshold = 0
    flap_mask = np.zeros(len(observed_td), dtype=bool)

# --- Figure 1: Permutation null distribution ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(perm_diffs, bins=50, alpha=0.7, color='steelblue', edgecolor='white',
        label='Permutation null')
ax.axvline(observed_diff, color='red', linewidth=2, linestyle='-',
           label=f'Observed = {observed_diff:.4f}')
pct95 = np.percentile(perm_diffs, 95)
ax.axvline(pct95, color='gray', linewidth=1.5, linestyle='--',
           label=f'95th percentile = {pct95:.4f}')
ax.set_xlabel('Excess near - Excess far', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'Temporal Clustering: Permutation Test (n={N_PERM_PRIMARY})\n'
             f'p = {p_value_perm:.4f}', fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'sprint3_temporal_permutation.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("  Saved figures/sprint3_temporal_permutation.png")

# --- Figure 2: Sensitivity heatmap ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ci, ct in enumerate(CANYON_THRESHOLDS):
    ax = axes[ci]
    heatmap = np.full((len(TEMPORAL_WINDOWS), len(SPATIAL_RADII)), np.nan)
    for ti, tw in enumerate(TEMPORAL_WINDOWS):
        for si, sr in enumerate(SPATIAL_RADII):
            key = (tw, sr, ct)
            if key in sensitivity_results:
                heatmap[ti, si] = sensitivity_results[key]['p']

    im = ax.imshow(heatmap, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(SPATIAL_RADII)))
    ax.set_xticklabels([f'{s} km' for s in SPATIAL_RADII])
    ax.set_yticks(range(len(TEMPORAL_WINDOWS)))
    ax.set_yticklabels([f'{t}d' for t in TEMPORAL_WINDOWS])
    ax.set_xlabel('Spatial radius')
    ax.set_ylabel('Temporal window')
    ax.set_title(f'Canyon threshold: {ct} km')

    # Annotate cells
    for ti in range(len(TEMPORAL_WINDOWS)):
        for si in range(len(SPATIAL_RADII)):
            key = (TEMPORAL_WINDOWS[ti], SPATIAL_RADII[si], ct)
            if key in sensitivity_results and not np.isnan(sensitivity_results[key]['p']):
                pv = sensitivity_results[key]['p']
                fdr_pv = sensitivity_results[key].get('fdr_p', 1.0)
                star = '***' if fdr_pv < 0.001 else '**' if fdr_pv < 0.01 else '*' if fdr_pv < 0.05 else ''
                is_primary = sensitivity_results[key].get('is_primary', False)
                txt = f'{pv:.3f}{star}'
                weight = 'bold' if is_primary else 'normal'
                color = 'white' if pv < 0.3 else 'black'
                ax.text(si, ti, txt, ha='center', va='center', fontsize=8,
                        fontweight=weight, color=color)

fig.colorbar(im, ax=axes, shrink=0.8, label='p-value')
fig.suptitle('Temporal Clustering: Sensitivity Grid (p-values, * = significant after BH-FDR)',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'sprint3_temporal_sensitivity.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("  Saved figures/sprint3_temporal_sensitivity.png")

# --- Figure 3: Flap map ---
fig, ax = plt.subplots(figsize=(12, 8))
# Background: all coastal UAP in gray
ax.scatter(uap_coords_temporal[:, 1], uap_coords_temporal[:, 0],
           s=0.3, c='lightgray', alpha=0.3, label='All coastal UAP')
# Canyon cells in blue
canyon_subsample = np.random.choice(len(canyon_lats),
                                     min(5000, len(canyon_lats)), replace=False)
ax.scatter(canyon_lons[canyon_subsample], canyon_lats[canyon_subsample],
           s=0.5, c='blue', alpha=0.2, label='Canyon cells')
# Flap members colored by episode
if len(flap_episodes) > 0:
    colors = plt.cm.Set1(np.linspace(0, 1, max(n_clusters, 1)))
    for ep in flap_episodes:
        label_id = ep['id'] - 1
        members = np.where(labels == label_id)[0]
        ax.scatter(flap_coords[members, 1], flap_coords[members, 0],
                   s=30, c=[colors[label_id % len(colors)]],
                   edgecolors='black', linewidths=0.5, zorder=5,
                   label=f"Episode #{ep['id']} (n={ep['n_reports']})")

ax.set_xlabel('Longitude', fontsize=11)
ax.set_ylabel('Latitude', fontsize=11)
ax.set_title('Temporal Flap Episodes Near Canyons', fontsize=13)
ax.legend(fontsize=8, loc='lower left')

# Watermark
ax.text(0.5, 0.5, 'DESCRIPTIVE — NOT STATISTICAL EVIDENCE',
        transform=ax.transAxes, fontsize=18, color='red', alpha=0.25,
        ha='center', va='center', rotation=30, fontweight='bold')

ax.set_xlim(-130, -60)
ax.set_ylim(22, 52)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'sprint3_flap_map.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("  Saved figures/sprint3_flap_map.png")

# Temporal verdict
if p_value_perm < 0.05:
    temporal_verdict = (f"TEMPORAL CLUSTERING CONFIRMED — near-canyon reports show "
                        f"excess temporal clustering beyond rate difference, "
                        f"p = {p_value_perm:.4f} from time-permutation null")
else:
    temporal_verdict = (f"NO TEMPORAL SIGNAL — permutation null not exceeded "
                        f"(p = {p_value_perm:.4f}), the spatial association "
                        f"reflects a rate difference only without episodic structure")

print(f"\n  TEMPORAL VERDICT: {temporal_verdict}")
print(f"\n  Part A done ({time.time() - t_partA:.1f}s)")


# ============================================================
# SECTION 11: PART B — SHELF GRADIENT COMPUTATION & GRID
# ============================================================
print("\n" + "=" * 70)
print("PART B: DOSE-RESPONSE (CANYON MAGNITUDE)")
print("=" * 70)
t_partB = time.time()

print("\n[SECTION 11] Gradient grid computation...")

# Extract shelf gradient data from ETOPO
shelf_rows, shelf_cols = np.where(shelf_mask)
shelf_lats = elev_lats[shelf_rows]
shelf_lons = elev_lons[shelf_cols]
shelf_gradients = grad_mag[shelf_rows, shelf_cols]
print(f"  Shelf cells: {len(shelf_lats):,}")
print(f"  Gradient stats: mean={shelf_gradients.mean():.1f}, "
      f"p95={np.percentile(shelf_gradients, 95):.1f}, "
      f"max={shelf_gradients.max():.1f} m/km")

# Build aggregated 0.1-degree grid
print("  Building gradient grid...")
gradient_by_cell = defaultdict(list)
for i in range(len(shelf_lats)):
    lat_bin = round(shelf_lats[i] / GRID_RES) * GRID_RES
    lon_bin = round(shelf_lons[i] / GRID_RES) * GRID_RES
    gradient_by_cell[(lat_bin, lon_bin)].append(float(shelf_gradients[i]))

print(f"  Grid cells with shelf data: {len(gradient_by_cell)}")

# Compute p95 gradient for all UAP + control points
all_coords = np.concatenate([
    np.column_stack([coastal_lats, coastal_lons]),
    np.column_stack([ctrl_lats, ctrl_lons])
])
print(f"  Computing p95 gradient for {len(all_coords)} points...")
p95_gradient_all_pre = get_p95_gradient_gridded(
    all_coords, gradient_by_cell, grid_res=GRID_RES, radius_km=GRADIENT_RADIUS_KM
)

# Apply valid_mask to align with X_raw / y_raw
p95_gradient_all = p95_gradient_all_pre[valid_mask.values]

p95_gradient_uap = p95_gradient_all[y_raw == 1]
p95_gradient_ctrl = p95_gradient_all[y_raw == 0]

# Log1p transform
log_gradient_all = np.log1p(p95_gradient_all)

# Zero-inflation report
pct_zero = np.mean(p95_gradient_all == 0) * 100
print(f"\n  Zero-inflation: {pct_zero:.1f}% of points have p95_gradient = 0")
print(f"  UAP gradient p95: mean={p95_gradient_uap.mean():.1f}, "
      f"median={np.median(p95_gradient_uap):.1f}, "
      f"zeros={np.sum(p95_gradient_uap == 0)} ({np.mean(p95_gradient_uap == 0)*100:.1f}%)")
print(f"  Control gradient p95: mean={p95_gradient_ctrl.mean():.1f}, "
      f"median={np.median(p95_gradient_ctrl):.1f}, "
      f"zeros={np.sum(p95_gradient_ctrl == 0)} ({np.mean(p95_gradient_ctrl == 0)*100:.1f}%)")

print(f"  Section 11 done ({time.time() - t_partB:.1f}s)")


# ============================================================
# SECTION 12: PART B — DOSE-RESPONSE MODEL & SENSITIVITY
# ============================================================
print("\n[SECTION 12] Dose-response models...")

# --- Step B3: Dose-response bins ---
print("\n  Dose-response: UAP fraction by p95 canyon gradient within 25 km")

uap_bins = np.digitize(p95_gradient_uap, DOSE_BINS) - 1
ctrl_bins = np.digitize(p95_gradient_ctrl, DOSE_BINS) - 1
all_bins = np.digitize(p95_gradient_all, DOSE_BINS) - 1

# Reference: first non-zero bin
ref_bin = 1  # '0-5 m/km'
ref_uap_n = (uap_bins == ref_bin).sum()
ref_ctrl_n = (ctrl_bins == ref_bin).sum()
ref_odds = ref_uap_n / ref_ctrl_n if ref_ctrl_n > 0 else np.nan

print(f"{'Gradient bin':>15} {'N_UAP':>8} {'N_ctrl':>8} {'UAP_frac':>10} {'OR vs ref':>12}")
dose_table = []
for b in range(len(DOSE_BIN_LABELS)):
    n_uap_b = (uap_bins == b).sum()
    n_ctrl_b = (ctrl_bins == b).sum()
    frac = n_uap_b / (n_uap_b + n_ctrl_b) if (n_uap_b + n_ctrl_b) > 0 else np.nan
    odds = n_uap_b / n_ctrl_b if n_ctrl_b > 0 else np.nan
    or_vs_ref = odds / ref_odds if (ref_odds and ref_odds > 0 and not np.isnan(odds)) else np.nan
    print(f"{DOSE_BIN_LABELS[b]:>15} {n_uap_b:>8} {n_ctrl_b:>8} {frac:>10.3f} {or_vs_ref:>12.2f}")
    dose_table.append({
        'bin': DOSE_BIN_LABELS[b],
        'n_uap': int(n_uap_b),
        'n_ctrl': int(n_ctrl_b),
        'uap_frac': float(frac) if not np.isnan(frac) else None,
        'or_vs_ref': float(or_vs_ref) if not np.isnan(or_vs_ref) else None,
    })

# Monotonic trend test (Spearman)
bin_midpoints = [0, 2.5, 7.5, 15, 35, 75, 200]
bin_ors = []
for b in range(len(DOSE_BIN_LABELS)):
    n_uap_b = (uap_bins == b).sum()
    n_ctrl_b = (ctrl_bins == b).sum()
    odds = n_uap_b / n_ctrl_b if n_ctrl_b > 0 else np.nan
    bin_ors.append(odds / ref_odds if (ref_odds > 0 and not np.isnan(odds)) else np.nan)

valid_trend = [(m, o) for m, o in zip(bin_midpoints, bin_ors) if not np.isnan(o)]
if len(valid_trend) >= 3:
    rho_trend, p_trend = spearmanr([v[0] for v in valid_trend],
                                    [v[1] for v in valid_trend])
    print(f"\n  Monotonic trend test: Spearman rho = {rho_trend:.3f}, p = {p_trend:.4f}")
else:
    rho_trend, p_trend = np.nan, np.nan
    print("\n  Monotonic trend test: insufficient valid bins")

# --- Step B4: Logistic regression with gradient ---
print("\n  Fitting logistic regression (Model B + gradient)...")

# Build feature matrix with gradient
features_dose = X_raw[feat_B].copy()
features_dose['log_p95_gradient'] = log_gradient_all

feat_dose_names = feat_B + ['log_p95_gradient']

# Z-score all features
from scipy.stats import zscore as scipy_zscore
X_dose_z = features_dose.apply(scipy_zscore)
X_dose_z_const = sm.add_constant(X_dose_z)

model_dose = sm.Logit(y_raw, X_dose_z_const).fit(disp=0, maxiter=1000)
print("\n  --- Full model with gradient ---")
print(model_dose.summary())

# Fit Model B without gradient for LR test
X_modelb_z = X_raw[feat_B].apply(scipy_zscore)
X_modelb_z_const = sm.add_constant(X_modelb_z)
model_b_nograds = sm.Logit(y_raw, X_modelb_z_const).fit(disp=0, maxiter=1000)

lr_stat = 2 * (model_dose.llf - model_b_nograds.llf)
lr_pvalue = 1 - chi2_dist.cdf(lr_stat, df=1)

gradient_beta = model_dose.params['log_p95_gradient']
gradient_ci = model_dose.conf_int().loc['log_p95_gradient'].values
gradient_p = model_dose.pvalues['log_p95_gradient']

print(f"\n  LR test for gradient: chi2 = {lr_stat:.2f}, p = {lr_pvalue:.2e}")
print(f"  Gradient coefficient: {gradient_beta:.4f} "
      f"(95% CI: [{gradient_ci[0]:.4f}, {gradient_ci[1]:.4f}])")
print(f"  Key: gradient adds predictive signal BEYOND simple canyon proximity "
      f"(dist_to_canyon is also in the model)")

# --- Step B5: Port-stratified check ---
print("\n  Port-stratified check...")

# Correlation between gradient and port density (among non-zero gradient)
nonzero_grad_mask = p95_gradient_all > 0
if nonzero_grad_mask.sum() > 10:
    rho_gp, p_gp = spearmanr(p95_gradient_all[nonzero_grad_mask],
                               port_count_all[nonzero_grad_mask])
    print(f"  Spearman(gradient, port_count) among non-zero: rho={rho_gp:.3f}, p={p_gp:.2e}")
else:
    rho_gp, p_gp = np.nan, np.nan

# Stratified OR
gradient_all_full = p95_gradient_all
try:
    port_terciles = pd.qcut(port_count_all, 3, labels=[1, 2, 3], duplicates='drop').astype(int)
except ValueError:
    # If too many ties, use manual quantile boundaries
    q33 = np.percentile(port_count_all, 33.3)
    q66 = np.percentile(port_count_all, 66.6)
    port_terciles = np.where(port_count_all <= q33, 1,
                              np.where(port_count_all <= q66, 2, 3))

stratified_results = {}
print("\n  Gradient effect stratified by port density:")
for t in [1, 2, 3]:
    t_mask = (port_terciles == t)
    t_uap = t_mask & (y_raw == 1)
    t_ctrl = t_mask & (y_raw == 0)

    nonzero_grad = gradient_all_full > 0
    high_thresh = np.percentile(gradient_all_full[nonzero_grad], 75) if nonzero_grad.sum() > 0 else 1
    low_thresh = np.percentile(gradient_all_full[nonzero_grad], 25) if nonzero_grad.sum() > 0 else 0

    high_grad = gradient_all_full > high_thresh
    low_grad = (gradient_all_full > 0) & (gradient_all_full <= low_thresh)

    n_high_uap = (t_uap & high_grad).sum()
    n_high_ctrl = (t_ctrl & high_grad).sum()
    n_low_uap = (t_uap & low_grad).sum()
    n_low_ctrl = (t_ctrl & low_grad).sum()

    if n_high_ctrl > 0 and n_low_ctrl > 0 and n_low_uap > 0 and n_high_uap > 0:
        or_high = n_high_uap / n_high_ctrl
        or_low = n_low_uap / n_low_ctrl
        ratio = or_high / or_low
        stratified_results[f'tercile_{t}'] = float(ratio)
        print(f"    Port tercile {t}: OR(high gradient)/OR(low gradient) = {ratio:.2f}")
    else:
        stratified_results[f'tercile_{t}'] = None
        print(f"    Port tercile {t}: insufficient data in one cell")

# --- Step B6: GAM for gradient ---
gam_results = {}
if HAS_PYGAM:
    print("\n  Fitting GAM with gradient spline...")
    try:
        X_gam_dose = np.column_stack([
            log_gradient_all,                                    # col 0: raw (spline)
            scipy_zscore(X_raw['dist_to_canyon_km'].values),     # col 1
            scipy_zscore(X_raw['dist_to_coast_km'].values),      # col 2
            scipy_zscore(X_raw['dist_to_military_km'].values),   # col 3
            scipy_zscore(X_raw['pop_density_proxy'].values),     # col 4
            scipy_zscore(X_raw['depth_nearest_ocean'].values),   # col 5
            scipy_zscore(X_raw['dist_to_nearest_port'].values),  # col 6
            scipy_zscore(X_raw['log_port_count_25km'].values),   # col 7
        ])

        gam_dose = LogisticGAM(
            s(0, n_splines=15) +
            l_term(1) + l_term(2) + l_term(3) + l_term(4) +
            l_term(5) + l_term(6) + l_term(7)
        )
        lam_grid = [0.001, 0.01, 0.1, 0.6, 1.0, 5.0, 10.0, 50.0, 100.0]
        gam_dose.gridsearch(X_gam_dose, y_raw, lam=np.array(lam_grid))

        XX_grad = gam_dose.generate_X_grid(term=0, n=200)
        pdep_grad, confi_grad = gam_dose.partial_dependence(term=0, X=XX_grad, width=0.95)

        gam_results = {
            'aic': float(gam_dose.statistics_['AIC']),
            'pseudo_r2': float(gam_dose.statistics_.get('pseudo_r2', {}).get('mcfadden', np.nan))
            if isinstance(gam_dose.statistics_.get('pseudo_r2'), dict) else np.nan,
        }
        print(f"    GAM AIC: {gam_results['aic']:.1f}")

        # --- Figure 5: GAM partial dependence ---
        fig, ax = plt.subplots(figsize=(10, 6))
        grad_vals = XX_grad[:, 0]
        ax.plot(grad_vals, pdep_grad, 'b-', linewidth=2, label='Partial dependence')
        ax.fill_between(grad_vals, confi_grad[:, 0], confi_grad[:, 1],
                         alpha=0.2, color='blue', label='95% CI')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)

        # Mark key gradient thresholds on original scale
        for gthresh in [20, 50, 100]:
            log_g = np.log1p(gthresh)
            if grad_vals.min() <= log_g <= grad_vals.max():
                ax.axvline(log_g, color='red', linestyle=':', alpha=0.5)
                ax.text(log_g, ax.get_ylim()[1] * 0.95, f'{gthresh} m/km',
                        fontsize=8, color='red', ha='center')

        ax.set_xlabel('log1p(p95 gradient) [m/km]', fontsize=12)
        ax.set_ylabel('Partial effect (log-odds)', fontsize=12)
        ax.set_title('GAM: Canyon Gradient Dose-Response\n'
                     '(controlling for canyon distance + 6 other covariates)', fontsize=13)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'sprint3_dose_response_gam.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()
        print("    Saved figures/sprint3_dose_response_gam.png")

    except Exception as e:
        print(f"    GAM fitting failed: {e}")
        gam_results = {'error': str(e)}
else:
    print("\n  Skipping GAM (pyGAM not available)")
    gam_results = {'note': 'pyGAM not installed'}

# --- Step B7: Sensitivity — alternative gradient features ---
print("\n  Sensitivity: alternative gradient features...")
alt_features = {}
for mode in ['max', 'mean_top10', 'count_steep_50']:
    print(f"    Computing {mode}...")
    feat_alt_pre = get_gradient_feature(
        all_coords, gradient_by_cell, grid_res=GRID_RES,
        radius_km=GRADIENT_RADIUS_KM, mode=mode
    )
    feat_alt = feat_alt_pre[valid_mask.values]
    log_feat_alt = np.log1p(feat_alt)

    features_alt = X_raw[feat_B].copy()
    features_alt['log_alt_gradient'] = log_feat_alt
    X_alt_z = features_alt.apply(scipy_zscore)
    X_alt_z_const = sm.add_constant(X_alt_z)

    model_alt = sm.Logit(y_raw, X_alt_z_const).fit(disp=0, maxiter=1000)
    lr_alt = 2 * (model_alt.llf - model_b_nograds.llf)
    p_alt = 1 - chi2_dist.cdf(lr_alt, df=1)
    beta_alt = model_alt.params['log_alt_gradient']

    alt_features[mode] = {
        'beta': float(beta_alt),
        'lr_p': float(p_alt),
        'lr_stat': float(lr_alt),
    }
    print(f"      {mode}: beta={beta_alt:.4f}, LR p={p_alt:.2e}")

# --- Figure 4: Dose-response bins ---
fig, ax = plt.subplots(figsize=(10, 6))
valid_ors = [(i, dose_table[i]['or_vs_ref']) for i in range(len(dose_table))
             if dose_table[i]['or_vs_ref'] is not None]
if valid_ors:
    x_pos = [v[0] for v in valid_ors]
    y_vals = [v[1] for v in valid_ors]
    colors_bar = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(valid_ors)))
    bars = ax.bar(x_pos, y_vals, color=colors_bar, edgecolor='black', linewidth=0.5)

    # N labels
    for xi, yi in valid_ors:
        n_total = dose_table[xi]['n_uap'] + dose_table[xi]['n_ctrl']
        ax.text(xi, yi + 0.02, f'n={n_total}', ha='center', fontsize=8)

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Reference OR = 1')
    ax.set_xticks(range(len(DOSE_BIN_LABELS)))
    ax.set_xticklabels(DOSE_BIN_LABELS, rotation=30, ha='right', fontsize=9)
    ax.set_xlabel('p95 gradient within 25 km (m/km)', fontsize=12)
    ax.set_ylabel('Odds Ratio vs reference (0-5 m/km)', fontsize=12)
    ax.set_title('Dose-Response: UAP Excess by Canyon Gradient', fontsize=13)
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'sprint3_dose_response_bins.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("  Saved figures/sprint3_dose_response_bins.png")

# --- Figure 6: Port-stratified ---
fig, ax = plt.subplots(figsize=(8, 6))
tercile_labels = ['Low ports', 'Medium ports', 'High ports']
tercile_vals = [stratified_results.get(f'tercile_{t}', None) for t in [1, 2, 3]]
valid_terciles = [(i, v) for i, v in enumerate(tercile_vals) if v is not None]
if valid_terciles:
    x_t = [v[0] for v in valid_terciles]
    y_t = [v[1] for v in valid_terciles]
    ax.bar(x_t, y_t, color=['lightblue', 'steelblue', 'navy'],
           edgecolor='black', linewidth=0.5)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='No gradient effect')
    ax.set_xticks(range(3))
    ax.set_xticklabels(tercile_labels, fontsize=11)
    ax.set_ylabel('OR(high gradient) / OR(low gradient)', fontsize=12)
    ax.set_title('Gradient Effect Stratified by Port Density', fontsize=13)
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'sprint3_dose_response_stratified.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("  Saved figures/sprint3_dose_response_stratified.png")

# Dose-response verdict
if lr_pvalue < 0.05 and not np.isnan(rho_trend) and rho_trend > 0:
    dose_verdict = (f"DOSE-RESPONSE CONFIRMED — steeper canyons predict more UAP excess, "
                    f"monotonically (rho={rho_trend:.3f}), beyond simple proximity and port "
                    f"infrastructure. LR test p={lr_pvalue:.2e}.")
elif lr_pvalue < 0.05:
    dose_verdict = (f"PARTIAL DOSE-RESPONSE — gradient adds predictive signal (LR p={lr_pvalue:.2e}) "
                    f"but monotonic trend is {'weak' if not np.isnan(rho_trend) else 'untested'} "
                    f"(rho={rho_trend:.3f}, p={p_trend:.4f}).")
else:
    dose_verdict = (f"NO DOSE-RESPONSE — canyon proximity matters but magnitude does not. "
                    f"LR test p={lr_pvalue:.2e}. The association may reflect coastline type, "
                    f"not canyon depth.")

print(f"\n  DOSE-RESPONSE VERDICT: {dose_verdict}")
print(f"\n  Part B done ({time.time() - t_partB:.1f}s)")


# ============================================================
# SECTION 13: RESULTS EXPORT & COMBINED SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SECTION 13: Results Export")
print("=" * 70)

runtime = time.time() - t_start

# Serialize sensitivity results
sens_serialized = {}
for k, v in sensitivity_results.items():
    key_str = f"tw{k[0]}_sr{k[1]}_ct{k[2]}"
    sens_serialized[key_str] = v

results = {
    'metadata': {
        'script': 'sprint3_temporal_doseresponse.py',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_uap_temporal': int(N_temporal),
        'n_uap_coastal': int(n_uap),
        'n_ctrl': int(n_ctrl),
        'n_total': int(N_total),
        'runtime_seconds': float(runtime),
        'pre_registered_specs': {
            'temporal': {
                'spatial_radius_km': SPATIAL_RADIUS_PRIMARY,
                'temporal_window_days': TEMPORAL_WINDOW_PRIMARY,
                'canyon_threshold_km': CANYON_THRESHOLD_PRIMARY,
                'n_permutations': N_PERM_PRIMARY,
            },
            'dose_response': {
                'gradient_feature': 'p95_gradient_within_25km',
                'transform': 'log1p',
                'model': 'Model_B_plus_gradient',
            },
        },
    },
    'part_a_temporal': {
        'primary': {
            'observed_diff': float(observed_diff),
            'p_value': float(p_value_perm),
            'n_permutations': int(N_PERM_PRIMARY),
            'perm_null_mean': float(perm_diffs.mean()),
            'perm_null_std': float(perm_diffs.std()),
            'perm_null_95pct': float(np.percentile(perm_diffs, 95)),
            'near_mean_td': float(observed_td[near_mask_primary].mean()),
            'far_mean_td': float(observed_td[far_mask_primary].mean()),
            'near_n': int(near_mask_primary.sum()),
            'far_n': int(far_mask_primary.sum()),
            'z_score': float(z_score_perm) if not np.isnan(z_score_perm) else None,
        },
        'robustness': robustness_results,
        'sensitivity': sens_serialized,
        'bh_fdr': {
            'n_significant': int(n_significant_fdr),
            'n_same_sign': int(n_same_sign),
            'n_total': int(len(sensitivity_results)),
        },
        'flap_episodes': flap_episodes,
        'n_flap_clusters': int(n_clusters),
        'verdict': temporal_verdict,
    },
    'part_b_dose_response': {
        'gradient_stats': {
            'n_shelf_cells': int(len(shelf_lats)),
            'pct_zero': float(pct_zero),
            'uap_mean': float(p95_gradient_uap.mean()),
            'ctrl_mean': float(p95_gradient_ctrl.mean()),
        },
        'dose_table': dose_table,
        'monotonic_trend': {
            'rho': float(rho_trend) if not np.isnan(rho_trend) else None,
            'p': float(p_trend) if not np.isnan(p_trend) else None,
        },
        'logistic_model': {
            'gradient_beta': float(gradient_beta),
            'gradient_ci': [float(gradient_ci[0]), float(gradient_ci[1])],
            'gradient_p': float(gradient_p),
            'lr_test_stat': float(lr_stat),
            'lr_test_p': float(lr_pvalue),
        },
        'port_stratified': stratified_results,
        'port_gradient_correlation': {
            'rho': float(rho_gp) if not np.isnan(rho_gp) else None,
            'p': float(p_gp) if not np.isnan(p_gp) else None,
        },
        'gam': gam_results,
        'sensitivity_features': alt_features,
        'verdict': dose_verdict,
    },
}

# Combined verdict
temporal_confirmed = p_value_perm < 0.05
dose_confirmed = lr_pvalue < 0.05
if temporal_confirmed and dose_confirmed:
    combined = ("Both confirmed: Two new independent evidence lines. "
                "Qualitatively different paper.")
elif temporal_confirmed or dose_confirmed:
    which = "Temporal" if temporal_confirmed else "Dose-response"
    combined = (f"One confirmed ({which}): Partial support. Report with caveats.")
else:
    combined = ("Neither confirmed: Sprint 1-2 finding = spatial rate difference only. "
                "Still publishable.")

results['combined_verdict'] = combined

# Save JSON
results_file = os.path.join(BASE_DIR, "sprint3_results.json")
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved {results_file}")

# --- Print combined summary ---
print("\n")
print("=" * 70)
print("SPRINT 3 RESULTS")
print("=" * 70)

print(f"""
PRE-REGISTERED PRIMARY SPECIFICATIONS:
  Temporal: {SPATIAL_RADIUS_PRIMARY} km spatial, +/-{TEMPORAL_WINDOW_PRIMARY} days, {CANYON_THRESHOLD_PRIMARY} km canyon threshold, {N_PERM_PRIMARY} time permutations within year
  Dose-response: p95_gradient_within_25km, log1p transform, LR test vs Model B

PART A: TEMPORAL CLUSTERING
----------------------------

1. Observed temporal density:
   Near-canyon: mean = {observed_td[near_mask_primary].mean():.2f} neighbors ({SPATIAL_RADIUS_PRIMARY} km, +/-{TEMPORAL_WINDOW_PRIMARY} days), median = {np.median(observed_td[near_mask_primary]):.2f}
   Far-canyon:  mean = {observed_td[far_mask_primary].mean():.2f}, median = {np.median(observed_td[far_mask_primary]):.2f}

2. Time-permutation test (PRIMARY):
   Observed excess_near - excess_far = {observed_diff:.4f}
   Permutation p = {p_value_perm:.4f} (k={k_exceeded}/{N_PERM_PRIMARY}, corrected (k+1)/(N+1))
   Z-score vs null: {z_score_perm:.1f}
   Permutation null: mean = {perm_diffs.mean():.4f}, std = {perm_diffs.std():.4f}

3. Robustness:
   Trimmed mean (5-95%):    p = {p_trimmed:.4f}, z = {z_trimmed:.1f}
   Heavy tail (frac td>=3): p = {p_tail:.4f}, z = {z_tail:.1f}
   Within-month null:       p = {p_monthly:.4f}, z = {z_monthly:.1f}

4. VERDICT:
   {temporal_verdict}

5. Sensitivity: {n_same_sign}/{len(sensitivity_results)} parameter combinations same sign, {n_significant_fdr}/{len(sensitivity_results)} significant after BH-FDR

6. Illustrative flap episodes (DESCRIPTIVE, not statistical evidence):""")

for ep in flap_episodes[:5]:
    print(f"   Episode #{ep['id']}: {ep['n_reports']} reports, "
          f"location {ep['lat_mean']:.2f}N/{ep['lon_mean']:.2f}W, "
          f"{ep['time_start']} to {ep['time_end']}, "
          f"nearest canyon: {ep['nearest_canyon_km']:.1f} km")

print(f"""
PART B: DOSE-RESPONSE
----------------------

1. Feature: p95_gradient_within_25km
   Zero-inflation: {pct_zero:.1f}% of points have gradient = 0

2. Dose-response table:""")
for row in dose_table:
    or_str = f"{row['or_vs_ref']:.2f}" if row['or_vs_ref'] is not None else "N/A"
    print(f"   Gradient {row['bin']:>15}: OR = {or_str}")

rho_str = f"{rho_trend:.3f}" if not np.isnan(rho_trend) else "N/A"
p_trend_str = f"{p_trend:.4f}" if not np.isnan(p_trend) else "N/A"
print(f"   Monotonic trend: Spearman rho = {rho_str}, p = {p_trend_str}")

print(f"""
3. Continuous model (PRIMARY):
   log_p95_gradient beta = {gradient_beta:.4f} (95% CI: [{gradient_ci[0]:.4f}, {gradient_ci[1]:.4f}])
   LR test vs Model B: chi2 = {lr_stat:.2f}, p = {lr_pvalue:.2e}
   NOTE: dist_to_canyon is also in the model — gradient adds BEYOND proximity

4. Port-stratified:""")
for t in [1, 2, 3]:
    val = stratified_results.get(f'tercile_{t}', None)
    val_str = f"{val:.2f}" if val is not None else "insufficient data"
    print(f"   Port tercile {t}: gradient OR ratio = {val_str}")

print(f"""
5. Sensitivity (alternative features):""")
for mode, res in alt_features.items():
    print(f"   {mode:>25}: beta = {res['beta']:.4f}, LR p = {res['lr_p']:.2e}")

print(f"""
6. VERDICT:
   {dose_verdict}

COMBINED SPRINT 3 VERDICT:
   Temporal:      {'CONFIRMED' if temporal_confirmed else 'NEGATIVE'}
   Dose-response: {'CONFIRMED' if dose_confirmed else 'NEGATIVE'}

   {combined}

Runtime: {runtime:.0f} seconds ({runtime/3600:.1f} hours)
""")

print("=" * 70)
print("Sprint 3 complete.")
print("=" * 70)
