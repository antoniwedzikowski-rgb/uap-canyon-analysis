#!/usr/bin/env python3
"""
Sprint 3 — Dose-Response Robustness: Control Bias Correction
=============================================================

Addresses the structural bias in control generation:
  Controls are drawn with land_weight=3.0 (land) vs 0.05 (ocean),
  creating 60x underrepresentation in offshore/shelf-edge zones
  where steep bathymetric gradients exist.

Three corrections:
  1. Coastal support: restrict to dist_coast <= 50 km (+ sensitivity 25/100 km)
  2. Weighted OR: importance-weight controls by 1/sampling_score
  3. Weighted GLM: logistic regression with sampling weights
     + re-generated controls with alternative land_weight priors

Uses IDENTICAL data loading, canyon detection, and feature computation
as 03_sprint3_temporal_doseresponse.py to ensure consistency.

Output:
  results/sprint3_dose_robustness.json
  figures/sprint3_dose_robustness_weighted_or.png
  figures/sprint3_dose_robustness_coastal_trend.png
"""

import os
import warnings
import time
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from scipy.ndimage import label as ndimage_label
from scipy.stats import spearmanr, chi2 as chi2_dist, zscore as scipy_zscore
from sklearn.neighbors import BallTree
import statsmodels.api as sm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# CONFIGURATION
# ============================================================
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_DIR, "data")
FIG_DIR = os.path.join(REPO_DIR, "figures")
RESULTS_DIR = os.path.join(REPO_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

R_EARTH = 6371.0
CANYON_GRADIENT_THRESHOLD = 20.0
COASTAL_BAND_KM = 200
N_CONTROL = 20000
PORT_CACHE_FILE = os.path.join(DATA_DIR, "port_coords_cache.npz")

GRADIENT_RADIUS_KM = 25
GRID_RES = 0.1
DOSE_BINS = [0, 0.01, 5, 10, 20, 50, 100, 500]
DOSE_BIN_LABELS = ['0 (no shelf)', '0-5', '5-10', '10-20', '20-50', '50-100', '100+']

COASTAL_SUPPORT_CUTS = [25, 50, 100]

print("=" * 70)
print("SPRINT 3 — DOSE-RESPONSE ROBUSTNESS: CONTROL BIAS CORRECTION")
print("=" * 70)
t_start = time.time()
t_global = time.time()


# ============================================================
# SECTION 1: UTILITY FUNCTIONS (copied from sprint3)
# ============================================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def get_p95_gradient_gridded(points, gradient_by_cell, grid_res=0.1, radius_km=25):
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


# ============================================================
# SECTION 2: DATA LOADING (identical to sprint3)
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
if 'y' in ds:
    elev_lats = ds['y'].values
    elev_lons = ds['x'].values
else:
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
# SECTION 3: COASTLINE & CANYON DETECTION (identical to sprint3)
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

# Canyon detection — shelf to -500m, connected component labeling
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

# Shelf gradient data for dose-response grid
shelf_rows, shelf_cols = np.where(shelf_mask)
shelf_lats = elev_lats[shelf_rows]
shelf_lons = elev_lons[shelf_cols]
shelf_gradients = grad_mag[shelf_rows, shelf_cols]

print(f"  Shelf cells: {len(shelf_lats):,}")
print(f"  Section 3 done ({time.time() - t_start:.1f}s)")


# ============================================================
# SECTION 4: COASTAL FILTERING & UAP METRICS (identical to sprint3)
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
N_UAP = len(coastal_lats)
print(f"  Coastal reports: {N_UAP}")

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

# Load port data early (needed by compute_ctrl_features)
data_ports = np.load(PORT_CACHE_FILE, allow_pickle=True)
port_coords = data_ports['port_coords']
port_tree = BallTree(np.radians(port_coords), metric='haversine')
print(f"  Port tree loaded: {len(port_coords)} ports")

# ============================================================
# SECTION 5: CONTROL GENERATION WITH SAMPLING SCORES
# ============================================================
print("\n[SECTION 5] Control points with sampling scores...")

def generate_controls_with_scores(n_ctrl, land_w, ocean_w, seed=42):
    """Generate controls with known sampling weights for importance weighting."""
    rng = np.random.RandomState(seed)

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
    lw = np.where(grid_elev >= 0, land_w, ocean_w)
    weights *= lw

    # Save raw weights before normalization
    raw_weights = weights.copy()
    weights = weights / weights.sum()

    chosen = rng.choice(len(glat_coastal), size=n_ctrl, p=weights, replace=True)
    jitter_val = 0.12
    c_lats = glat_coastal[chosen] + rng.uniform(-jitter_val, jitter_val, n_ctrl)
    c_lons = glon_coastal[chosen] + rng.uniform(-jitter_val, jitter_val, n_ctrl)

    sampling_scores = raw_weights[chosen]

    # Filter to coastal band
    _, cc_idx = coast_tree.query(np.column_stack([c_lats, c_lons]), k=1)
    cc_km = haversine_km(c_lats, c_lons, coast_lats[cc_idx], coast_lons[cc_idx])
    mask = cc_km <= COASTAL_BAND_KM
    c_lats = c_lats[mask]
    c_lons = c_lons[mask]
    cc_km = cc_km[mask]
    sampling_scores = sampling_scores[mask]

    return c_lats, c_lons, cc_km, sampling_scores


def compute_ctrl_features(c_lats, c_lons, c_coast_km):
    """Compute all features for control points (same metrics as sprint3)."""
    n = len(c_lats)
    _, c_ocean_idx = ocean_tree.query(np.column_stack([c_lats, c_lons]), k=1)
    c_depths = ocean_depths[c_ocean_idx]

    _, c_canyon_idx = canyon_tree.query(np.column_stack([c_lats, c_lons]), k=1)
    c_canyon_km = haversine_km(c_lats, c_lons,
                                canyon_lats[c_canyon_idx], canyon_lons[c_canyon_idx])

    c_base_dists, _ = base_tree.query(np.column_stack([c_lats, c_lons]), k=1)
    c_base_km = c_base_dists * 111.0

    c_county_dists, c_county_idx = county_tree.query(
        np.column_stack([c_lats, c_lons]), k=5)
    c_pop = np.zeros(n)
    for k in range(5):
        d_km = c_county_dists[:, k] * 111.0 + 1.0
        c_pop += counties_pop[c_county_idx[:, k]] / (d_km ** 2)

    c_coords_rad = np.radians(np.column_stack([c_lats, c_lons]))
    c_port_dists, _ = port_tree.query(c_coords_rad, k=1)
    c_port_km = c_port_dists.flatten() * R_EARTH
    c_port_count_25 = port_tree.query_radius(c_coords_rad, r=25.0/R_EARTH, count_only=True)
    c_log_port_count_25 = np.log1p(c_port_count_25)

    return {
        'dist_to_canyon_km': c_canyon_km,
        'dist_to_coast_km': c_coast_km,
        'dist_to_military_km': c_base_km,
        'pop_density_proxy': c_pop,
        'depth_nearest_ocean': c_depths,
        'dist_to_nearest_port': c_port_km,
        'log_port_count_25km': c_log_port_count_25,
    }


# Generate original controls (same seed=42 as sprint3)
ctrl_lats, ctrl_lons, ctrl_coast_km, ctrl_sampling_scores = generate_controls_with_scores(
    N_CONTROL, land_w=3.0, ocean_w=0.05, seed=42
)
N_CTRL = len(ctrl_lats)
print(f"  Controls: {N_CTRL}")

ctrl_features = compute_ctrl_features(ctrl_lats, ctrl_lons, ctrl_coast_km)

# UAP port metrics (port_tree already loaded above)
uap_coords_rad = np.radians(np.column_stack([coastal_lats, coastal_lons]))
port_dists_rad, _ = port_tree.query(uap_coords_rad, k=1)
df_coastal['dist_to_nearest_port'] = port_dists_rad.flatten() * R_EARTH
counts_25_uap = port_tree.query_radius(uap_coords_rad, r=25.0 / R_EARTH, count_only=True)
df_coastal['log_port_count_25km'] = np.log1p(counts_25_uap)

# Control port metrics (update ctrl_features)
ctrl_coords_rad = np.radians(np.column_stack([ctrl_lats, ctrl_lons]))
ctrl_port_dists_rad, _ = port_tree.query(ctrl_coords_rad, k=1)
ctrl_features['dist_to_nearest_port'] = ctrl_port_dists_rad.flatten() * R_EARTH
ctrl_port_count_25 = port_tree.query_radius(ctrl_coords_rad, r=25.0/R_EARTH, count_only=True)
ctrl_features['log_port_count_25km'] = np.log1p(ctrl_port_count_25)


# ============================================================
# SECTION 6: FEATURE ASSEMBLY
# ============================================================
print("\n[SECTION 6] Feature assembly...")

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

X_ctrl_raw = pd.DataFrame(ctrl_features)

X_raw_pre = pd.concat([X_uap_raw, X_ctrl_raw], ignore_index=True)
y_raw_pre = np.concatenate([np.ones(len(X_uap_raw)), np.zeros(len(X_ctrl_raw))])

# Sampling weights: UAP=1, CTRL=1/sampling_score
sampling_weights_pre = np.concatenate([
    np.ones(N_UAP),
    1.0 / (ctrl_sampling_scores + 1e-10)
])
# Normalize ctrl weights to sum to N_CTRL
ctrl_w = sampling_weights_pre[N_UAP:]
ctrl_w = ctrl_w / ctrl_w.sum() * N_CTRL
sampling_weights_pre[N_UAP:] = ctrl_w

# Coast distance for all points
all_coast_km_pre = np.concatenate([
    df_coastal['dist_to_coast_km'].values,
    ctrl_coast_km
])

# Valid mask
valid_mask = X_raw_pre.notna().all(axis=1)
n_dropped = (~valid_mask).sum()
print(f"  NaN rows dropped: {n_dropped}")

X_raw = X_raw_pre[valid_mask].reset_index(drop=True)
y_raw = y_raw_pre[valid_mask.values]
sampling_weights = sampling_weights_pre[valid_mask.values]
all_coast_km = all_coast_km_pre[valid_mask.values]

N_UAP_valid = int(y_raw.sum())
N_CTRL_valid = int((1 - y_raw).sum())
print(f"  Dataset: {len(y_raw):,} ({N_UAP_valid:,} UAP, {N_CTRL_valid:,} ctrl)")


# ============================================================
# SECTION 7: GRADIENT COMPUTATION
# ============================================================
print("\n[SECTION 7] Gradient computation...")

gradient_by_cell = defaultdict(list)
for i in range(len(shelf_lats)):
    lat_bin = round(shelf_lats[i] / GRID_RES) * GRID_RES
    lon_bin = round(shelf_lons[i] / GRID_RES) * GRID_RES
    gradient_by_cell[(lat_bin, lon_bin)].append(float(shelf_gradients[i]))
print(f"  Grid cells: {len(gradient_by_cell)}")

all_coords_pre = np.concatenate([
    np.column_stack([coastal_lats, coastal_lons]),
    np.column_stack([ctrl_lats, ctrl_lons])
])
print(f"  Computing p95 gradient for {len(all_coords_pre)} points...")
p95_gradient_all_pre = get_p95_gradient_gridded(
    all_coords_pre, gradient_by_cell, grid_res=GRID_RES, radius_km=GRADIENT_RADIUS_KM
)
p95_gradient_all = p95_gradient_all_pre[valid_mask.values]
log_gradient_all = np.log1p(p95_gradient_all)

p95_uap = p95_gradient_all[y_raw == 1]
p95_ctrl = p95_gradient_all[y_raw == 0]

print(f"  UAP gradient: mean={p95_uap.mean():.1f}, zeros={np.sum(p95_uap == 0)}")
print(f"  Ctrl gradient: mean={p95_ctrl.mean():.1f}, zeros={np.sum(p95_ctrl == 0)}")
print(f"  Section 7 done ({time.time() - t_start:.1f}s)")


# ============================================================
# SECTION 8: STEP 1 — COASTAL SUPPORT
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: COASTAL SUPPORT — Restrict to dist_coast <= X km")
print("=" * 70)

results = {'coastal_support': {}, 'weighted_or': {}, 'weighted_glm': {},
           'alt_controls': {}, 'metadata': {}}

for coast_cut in COASTAL_SUPPORT_CUTS:
    print(f"\n  --- Coastal cut: {coast_cut} km ---")
    coast_mask_cut = all_coast_km <= coast_cut
    y_cut = y_raw[coast_mask_cut]
    grad_cut = p95_gradient_all[coast_mask_cut]

    n_uap_cut = int(y_cut.sum())
    n_ctrl_cut = int((1 - y_cut).sum())
    print(f"  N: {n_uap_cut} UAP, {n_ctrl_cut} ctrl")

    uap_g = grad_cut[y_cut == 1]
    ctrl_g = grad_cut[y_cut == 0]
    uap_bins = np.digitize(uap_g, DOSE_BINS) - 1
    ctrl_bins = np.digitize(ctrl_g, DOSE_BINS) - 1

    ref_bin = 1
    ref_uap_n = (uap_bins == ref_bin).sum()
    ref_ctrl_n = (ctrl_bins == ref_bin).sum()
    ref_odds = ref_uap_n / ref_ctrl_n if ref_ctrl_n > 0 else np.nan

    bin_results = []
    print(f"  {'Bin':>15} {'N_UAP':>8} {'N_ctrl':>8} {'OR':>10}")
    for b in range(len(DOSE_BIN_LABELS)):
        n_u = int((uap_bins == b).sum())
        n_c = int((ctrl_bins == b).sum())
        odds = n_u / n_c if n_c > 0 else np.nan
        or_val = odds / ref_odds if (ref_odds > 0 and not np.isnan(odds) and not np.isnan(ref_odds)) else np.nan
        or_str = f"{or_val:.2f}" if not np.isnan(or_val) else "nan"
        print(f"  {DOSE_BIN_LABELS[b]:>15} {n_u:>8} {n_c:>8} {or_str:>10}")
        bin_results.append({'bin': DOSE_BIN_LABELS[b], 'n_uap': n_u, 'n_ctrl': n_c,
                           'or_vs_ref': float(or_val) if not np.isnan(or_val) else None})

    bin_midpoints = [0, 2.5, 7.5, 15, 35, 75, 200]
    bin_ors = [r['or_vs_ref'] for r in bin_results]
    valid = [(m, o) for m, o in zip(bin_midpoints, bin_ors) if o is not None]
    if len(valid) >= 3:
        rho, p = spearmanr([v[0] for v in valid], [v[1] for v in valid])
    else:
        rho, p = np.nan, np.nan
    rho_str = f"{rho:.3f}" if not np.isnan(rho) else "N/A"
    p_str = f"{p:.4f}" if not np.isnan(p) else "N/A"
    print(f"  Spearman rho = {rho_str}, p = {p_str}")

    results['coastal_support'][f'{coast_cut}km'] = {
        'n_uap': n_uap_cut, 'n_ctrl': n_ctrl_cut,
        'dose_table': bin_results,
        'spearman_rho': float(rho) if not np.isnan(rho) else None,
        'spearman_p': float(p) if not np.isnan(p) else None,
    }


# ============================================================
# SECTION 9: STEP 2 — WEIGHTED OR
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: WEIGHTED OR — Importance-weighted dose-response bins")
print("=" * 70)

ctrl_weights = sampling_weights[y_raw == 0]

uap_g = p95_gradient_all[y_raw == 1]
ctrl_g = p95_gradient_all[y_raw == 0]
uap_bins_all = np.digitize(uap_g, DOSE_BINS) - 1
ctrl_bins_all = np.digitize(ctrl_g, DOSE_BINS) - 1

ref_uap_n = (uap_bins_all == 1).sum()
ref_ctrl_w = ctrl_weights[ctrl_bins_all == 1].sum()
ref_odds_w = ref_uap_n / ref_ctrl_w if ref_ctrl_w > 0 else np.nan

# Also raw reference
ref_ctrl_raw = (ctrl_bins_all == 1).sum()
ref_raw_odds = ref_uap_n / ref_ctrl_raw if ref_ctrl_raw > 0 else np.nan

weighted_bin_results = []
print(f"\n  {'Bin':>15} {'N_UAP':>8} {'N_ctrl':>8} {'W_ctrl':>10} {'Raw OR':>10} {'Wtd OR':>10}")
for b in range(len(DOSE_BIN_LABELS)):
    n_u = int((uap_bins_all == b).sum())
    n_c_raw = int((ctrl_bins_all == b).sum())
    w_c = float(ctrl_weights[ctrl_bins_all == b].sum())

    raw_odds = n_u / n_c_raw if n_c_raw > 0 else np.nan
    raw_or = raw_odds / ref_raw_odds if (ref_raw_odds > 0 and not np.isnan(raw_odds)) else np.nan

    w_odds = n_u / w_c if w_c > 0 else np.nan
    w_or = w_odds / ref_odds_w if (ref_odds_w > 0 and not np.isnan(w_odds)) else np.nan

    raw_str = f"{raw_or:.2f}" if not np.isnan(raw_or) else "nan"
    w_str = f"{w_or:.2f}" if not np.isnan(w_or) else "nan"
    print(f"  {DOSE_BIN_LABELS[b]:>15} {n_u:>8} {n_c_raw:>8} {w_c:>10.1f} {raw_str:>10} {w_str:>10}")
    weighted_bin_results.append({
        'bin': DOSE_BIN_LABELS[b], 'n_uap': n_u, 'n_ctrl_raw': n_c_raw,
        'w_ctrl': float(w_c),
        'raw_or': float(raw_or) if not np.isnan(raw_or) else None,
        'weighted_or': float(w_or) if not np.isnan(w_or) else None,
    })

bin_midpoints = [0, 2.5, 7.5, 15, 35, 75, 200]
w_ors = [r['weighted_or'] for r in weighted_bin_results]
valid = [(m, o) for m, o in zip(bin_midpoints, w_ors) if o is not None]
if len(valid) >= 3:
    rho_w, p_w = spearmanr([v[0] for v in valid], [v[1] for v in valid])
else:
    rho_w, p_w = np.nan, np.nan
rho_str = f"{rho_w:.3f}" if not np.isnan(rho_w) else "N/A"
p_str = f"{p_w:.4f}" if not np.isnan(p_w) else "N/A"
print(f"\n  Weighted OR Spearman: rho = {rho_str}, p = {p_str}")

results['weighted_or'] = {
    'dose_table': weighted_bin_results,
    'spearman_rho': float(rho_w) if not np.isnan(rho_w) else None,
    'spearman_p': float(p_w) if not np.isnan(p_w) else None,
}


# ============================================================
# SECTION 10: STEP 3A — WEIGHTED GLM
# ============================================================
print("\n" + "=" * 70)
print("STEP 3A: WEIGHTED GLM — Logistic with importance weights")
print("=" * 70)

features_dose = X_raw[feat_B].copy()
features_dose['log_p95_gradient'] = log_gradient_all

X_z = features_dose.apply(scipy_zscore)
X_z_const = sm.add_constant(X_z)

# Unweighted
model_unweighted = sm.Logit(y_raw, X_z_const).fit(disp=0, maxiter=1000)
beta_uw = model_unweighted.params['log_p95_gradient']
ci_uw = model_unweighted.conf_int().loc['log_p95_gradient'].values
p_uw = model_unweighted.pvalues['log_p95_gradient']
print(f"\n  Unweighted: beta = {beta_uw:.4f} [{ci_uw[0]:.4f}, {ci_uw[1]:.4f}], p = {p_uw:.2e}")

# Weighted GLM
model_weighted = sm.GLM(y_raw, X_z_const,
                         family=sm.families.Binomial(),
                         freq_weights=sampling_weights).fit(disp=0, maxiter=1000)
beta_w = model_weighted.params['log_p95_gradient']
ci_w = model_weighted.conf_int().loc['log_p95_gradient'].values
p_w_glm = model_weighted.pvalues['log_p95_gradient']
print(f"  Weighted:   beta = {beta_w:.4f} [{ci_w[0]:.4f}, {ci_w[1]:.4f}], p = {p_w_glm:.2e}")

# Deviance test
model_b_weighted = sm.GLM(y_raw, sm.add_constant(X_raw[feat_B].apply(scipy_zscore)),
                           family=sm.families.Binomial(),
                           freq_weights=sampling_weights).fit(disp=0, maxiter=1000)
deviance_diff = model_b_weighted.deviance - model_weighted.deviance
deviance_p = 1 - chi2_dist.cdf(deviance_diff, df=1)
print(f"  Deviance test (gradient added): stat = {deviance_diff:.1f}, p = {deviance_p:.2e}")

results['weighted_glm'] = {
    'unweighted': {'beta': float(beta_uw), 'ci': [float(ci_uw[0]), float(ci_uw[1])], 'p': float(p_uw)},
    'weighted': {'beta': float(beta_w), 'ci': [float(ci_w[0]), float(ci_w[1])], 'p': float(p_w_glm)},
    'deviance_test': {'stat': float(deviance_diff), 'p': float(deviance_p)},
}


# ============================================================
# SECTION 11: STEP 3B — ALTERNATIVE CONTROL PRIORS
# ============================================================
print("\n" + "=" * 70)
print("STEP 3B: ALTERNATIVE CONTROL PRIORS")
print("=" * 70)

LAND_WEIGHT_ALTS = {
    'original': {'land': 3.0, 'ocean': 0.05},
    'balanced': {'land': 1.0, 'ocean': 0.5},
    'ocean_friendly': {'land': 1.0, 'ocean': 1.0},
}

for prior_name, prior_weights in LAND_WEIGHT_ALTS.items():
    print(f"\n  --- Prior: {prior_name} (land={prior_weights['land']}, ocean={prior_weights['ocean']}) ---")

    alt_lats, alt_lons, alt_coast_km, _ = generate_controls_with_scores(
        N_CONTROL, land_w=prior_weights['land'], ocean_w=prior_weights['ocean'], seed=123
    )
    n_alt = len(alt_lats)
    print(f"  Generated {n_alt} controls")

    alt_features = compute_ctrl_features(alt_lats, alt_lons, alt_coast_km)

    # Gradient
    alt_coords = np.column_stack([alt_lats, alt_lons])
    print(f"  Computing gradient for {n_alt} alt controls...")
    alt_gradient = get_p95_gradient_gridded(alt_coords, gradient_by_cell,
                                             grid_res=GRID_RES, radius_km=GRADIENT_RADIUS_KM)

    # Build dataset
    X_alt_ctrl = pd.DataFrame(alt_features)
    X_alt = pd.concat([X_uap_raw, X_alt_ctrl], ignore_index=True)
    y_alt = np.concatenate([np.ones(N_UAP), np.zeros(n_alt)])
    grad_alt_all = np.concatenate([p95_uap, alt_gradient])
    log_grad_alt = np.log1p(grad_alt_all)

    valid_alt = X_alt.notna().all(axis=1)
    X_alt = X_alt[valid_alt].reset_index(drop=True)
    y_alt = y_alt[valid_alt.values]
    log_grad_alt = log_grad_alt[valid_alt.values]
    grad_alt_valid = grad_alt_all[valid_alt.values]

    # Logistic with gradient
    features_alt = X_alt[feat_B].copy()
    features_alt['log_p95_gradient'] = log_grad_alt
    X_alt_z = features_alt.apply(scipy_zscore)
    X_alt_z_const = sm.add_constant(X_alt_z)

    model_alt = sm.Logit(y_alt, X_alt_z_const).fit(disp=0, maxiter=1000)
    beta_alt = model_alt.params['log_p95_gradient']
    ci_alt = model_alt.conf_int().loc['log_p95_gradient'].values
    p_alt = model_alt.pvalues['log_p95_gradient']

    X_alt_nograds = sm.add_constant(X_alt[feat_B].apply(scipy_zscore))
    model_alt_no = sm.Logit(y_alt, X_alt_nograds).fit(disp=0, maxiter=1000)
    lr_alt = 2 * (model_alt.llf - model_alt_no.llf)
    lr_alt_p = 1 - chi2_dist.cdf(lr_alt, df=1)

    print(f"  beta_gradient = {beta_alt:.4f} [{ci_alt[0]:.4f}, {ci_alt[1]:.4f}], p = {p_alt:.2e}")
    print(f"  LR test: stat = {lr_alt:.1f}, p = {lr_alt_p:.2e}")

    # Dose-response bins
    uap_g_alt = grad_alt_valid[y_alt == 1]
    ctrl_g_alt = grad_alt_valid[y_alt == 0]
    uap_bins_alt = np.digitize(uap_g_alt, DOSE_BINS) - 1
    ctrl_bins_alt = np.digitize(ctrl_g_alt, DOSE_BINS) - 1

    ref_u = (uap_bins_alt == 1).sum()
    ref_c = (ctrl_bins_alt == 1).sum()
    ref_o = ref_u / ref_c if ref_c > 0 else np.nan

    alt_dose = []
    for b in range(len(DOSE_BIN_LABELS)):
        n_u = int((uap_bins_alt == b).sum())
        n_c = int((ctrl_bins_alt == b).sum())
        odds = n_u / n_c if n_c > 0 else np.nan
        or_val = odds / ref_o if (ref_o > 0 and not np.isnan(odds)) else np.nan
        alt_dose.append({'bin': DOSE_BIN_LABELS[b], 'n_uap': n_u, 'n_ctrl': n_c,
                        'or_vs_ref': float(or_val) if not np.isnan(or_val) else None})

    alt_ors = [r['or_vs_ref'] for r in alt_dose]
    valid_t = [(m, o) for m, o in zip([0,2.5,7.5,15,35,75,200], alt_ors) if o is not None]
    if len(valid_t) >= 3:
        rho_alt, p_rho_alt = spearmanr([v[0] for v in valid_t], [v[1] for v in valid_t])
    else:
        rho_alt, p_rho_alt = np.nan, np.nan

    results['alt_controls'][prior_name] = {
        'n_ctrl': n_alt,
        'beta_gradient': float(beta_alt),
        'ci': [float(ci_alt[0]), float(ci_alt[1])],
        'p': float(p_alt),
        'lr_stat': float(lr_alt), 'lr_p': float(lr_alt_p),
        'dose_table': alt_dose,
        'spearman_rho': float(rho_alt) if not np.isnan(rho_alt) else None,
        'spearman_p': float(p_rho_alt) if not np.isnan(p_rho_alt) else None,
    }


# ============================================================
# SECTION 12: FIGURES
# ============================================================
print("\n[SECTION 12] Generating figures...")

# Figure 1: Weighted vs Raw OR
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

raw_ors_plot = [r['raw_or'] if r['raw_or'] is not None else 0 for r in weighted_bin_results]
wtd_ors_plot = [r['weighted_or'] if r['weighted_or'] is not None else 0 for r in weighted_bin_results]
x = np.arange(len(DOSE_BIN_LABELS))

axes[0].bar(x - 0.2, raw_ors_plot, 0.35, label='Raw OR', color='steelblue', alpha=0.8)
axes[0].bar(x + 0.2, wtd_ors_plot, 0.35, label='Weighted OR', color='coral', alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(DOSE_BIN_LABELS, rotation=45, ha='right', fontsize=9)
axes[0].set_ylabel('OR vs reference (0-5 m/km)', fontsize=11)
axes[0].set_title('Raw vs Importance-Weighted OR', fontsize=12)
axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
axes[0].legend(fontsize=10)

n_ctrl_raw_plot = [r['n_ctrl_raw'] for r in weighted_bin_results]
w_ctrl_plot = [r['w_ctrl'] for r in weighted_bin_results]
axes[1].bar(x - 0.2, n_ctrl_raw_plot, 0.35, label='Raw N_ctrl', color='steelblue', alpha=0.8)
axes[1].bar(x + 0.2, w_ctrl_plot, 0.35, label='Weighted N_ctrl', color='coral', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(DOSE_BIN_LABELS, rotation=45, ha='right', fontsize=9)
axes[1].set_ylabel('Control count / weight', fontsize=11)
axes[1].set_title('Control Representation by Gradient Bin', fontsize=12)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'sprint3_dose_robustness_weighted_or.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("  Saved sprint3_dose_robustness_weighted_or.png")

# Figure 2: Coastal support trends
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for i, coast_cut in enumerate(COASTAL_SUPPORT_CUTS):
    key = f'{coast_cut}km'
    data_cs = results['coastal_support'][key]
    ors = [r['or_vs_ref'] if r['or_vs_ref'] is not None else 0 for r in data_cs['dose_table']]
    axes[i].bar(range(len(DOSE_BIN_LABELS)), ors, color='teal', alpha=0.7)
    axes[i].set_xticks(range(len(DOSE_BIN_LABELS)))
    axes[i].set_xticklabels(DOSE_BIN_LABELS, rotation=45, ha='right', fontsize=8)
    rho_val = data_cs['spearman_rho']
    p_val = data_cs['spearman_p']
    rho_str = f"{rho_val:.3f}" if rho_val is not None else "N/A"
    p_str = f"{p_val:.4f}" if p_val is not None else "N/A"
    axes[i].set_title(f'Coast <= {coast_cut} km\nrho={rho_str}, p={p_str}', fontsize=11)
    axes[i].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

axes[0].set_ylabel('OR vs reference', fontsize=11)
fig.suptitle('Dose-Response Under Coastal Support Restrictions', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'sprint3_dose_robustness_coastal_trend.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("  Saved sprint3_dose_robustness_coastal_trend.png")


# ============================================================
# SECTION 13: RESULTS SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

results['metadata'] = {
    'script': '04_sprint3_dose_robustness.py',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'n_uap': N_UAP_valid,
    'n_ctrl_original': N_CTRL_valid,
    'runtime_seconds': time.time() - t_global,
    'purpose': 'Control bias correction for dose-response (land_weight 60x asymmetry)',
}

# Print all results
coastal_rhos = {k: v['spearman_rho'] for k, v in results['coastal_support'].items()}
weighted_rho = results['weighted_or']['spearman_rho']
weighted_beta = results['weighted_glm']['weighted']['beta']
alt_betas = {k: v['beta_gradient'] for k, v in results['alt_controls'].items()}

print(f"\n  Coastal support trends (Spearman rho):")
for k, v in coastal_rhos.items():
    v_str = f"{v:.3f}" if v is not None else "N/A"
    print(f"    {k}: rho = {v_str}")

w_rho_str = f"{weighted_rho:.3f}" if weighted_rho is not None else "N/A"
print(f"\n  Weighted OR: rho = {w_rho_str}")
print(f"  Gradient beta (unweighted): {results['weighted_glm']['unweighted']['beta']:.4f}")
print(f"  Gradient beta (weighted):   {weighted_beta:.4f}")

print(f"\n  Alternative control priors:")
for k, v in alt_betas.items():
    rho_v = results['alt_controls'][k].get('spearman_rho')
    rho_str = f"{rho_v:.3f}" if rho_v is not None else "N/A"
    print(f"    {k}: beta = {v:.4f}, rho = {rho_str}")

# Verdict
trend_survives = (weighted_rho is not None and weighted_rho > 0.5 and
                  all(v is not None and v > 0.3 for v in coastal_rhos.values()))
beta_stable = all(abs(v) > 0.1 for v in alt_betas.values())

if trend_survives and beta_stable:
    verdict = ("DOSE-RESPONSE ROBUST — monotonic trend survives importance weighting, "
               "coastal support restriction, and alternative control priors. "
               f"Weighted beta = {weighted_beta:.3f}.")
elif beta_stable:
    verdict = ("DOSE-RESPONSE PARTIALLY ROBUST — gradient beta is stable across "
               f"control priors ({min(abs(v) for v in alt_betas.values()):.3f} to "
               f"{max(abs(v) for v in alt_betas.values()):.3f}), "
               "but binned OR trend requires weighting correction.")
else:
    verdict = ("DOSE-RESPONSE FRAGILE — the signal does not survive "
               "control bias correction.")

results['verdict'] = verdict
print(f"\n  VERDICT: {verdict}")

results_file = os.path.join(RESULTS_DIR, "sprint3_dose_robustness.json")
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved {results_file}")
print(f"\n  Total runtime: {time.time() - t_global:.1f}s")
print("\nDONE.")
