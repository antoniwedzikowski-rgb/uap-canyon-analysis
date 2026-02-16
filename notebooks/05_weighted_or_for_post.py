#!/usr/bin/env python3
"""
Compute binned weighted OR with bootstrap CI for Reddit post.

Uses the same data pipeline as 04_sprint3_dose_robustness.py but:
  - 4 gradient categories instead of 7 (flat / moderate / steep / very steep)
  - Bootstrap CI (2000 resamples) on weighted OR per bin
  - Output: results/weighted_or_binned.json + console table

Gradient categories:
  flat:        p95_gradient = 0 (no shelf data)
  moderate:    0 < gradient <= 10
  steep:       10 < gradient <= 50
  very_steep:  gradient > 50
Reference bin: moderate (0-10)
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree
import statsmodels.api as sm

warnings.filterwarnings('ignore')
t_start = time.time()

# ============================================================
# PATHS (same as script 04)
# ============================================================
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = "/Users/antoniwedzikowski/Desktop/UAP research/data"
RESULTS_DIR = os.path.join(REPO_DIR, "results")

R_EARTH = 6371.0
CANYON_GRADIENT_THRESHOLD = 20.0
COASTAL_BAND_KM = 200
N_CONTROL = 20000
PORT_CACHE_FILE = os.path.join(DATA_DIR, "port_coords_cache.npz")
GRADIENT_RADIUS_KM = 25
GRID_RES = 0.1

N_BOOTSTRAP = 2000
RNG_SEED = 42

# 4-bin categories (per user spec: natural breakpoints, no bin < 30 obs)
BIN_EDGES = [0, 10, 30, 60, 500]
BIN_LABELS = ['0-10 (flat)', '10-30 (moderate)', '30-60 (steep)', '60+ (very steep)']
REF_BIN = 0  # flat as reference

WINSORIZE_PCT = 95  # cap importance weights at this percentile

print("=" * 70)
print("WEIGHTED OR WITH BOOTSTRAP CI — 4 GRADIENT CATEGORIES")
print("=" * 70)

# ============================================================
# SECTIONS 2-7: REPLICATE DATA PIPELINE FROM SPRINT 3
# (identical to script 04 — load data, detect canyons,
#  generate controls with sampling scores, compute gradient)
# ============================================================

# --- Section 2: Load data ---
print("\n[2] Loading data...")
df = pd.read_csv(os.path.join(DATA_DIR, "nuforc_reports.csv"), header=None,
                 names=['datetime_str','city','state','country','shape',
                        'duration_seconds','duration_text','description',
                        'date_posted','latitude','longitude'])
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df = df.dropna(subset=['latitude','longitude'])
# Sprint 3 does NOT filter by country — just CONUS bounding box
df = df[(df['latitude'] != 0) & (df['longitude'] != 0)]
df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]
df = df.drop_duplicates(subset=['latitude', 'longitude', 'datetime_str'])
df_us = df[(df['latitude'] >= 20) & (df['latitude'] <= 55) &
           (df['longitude'] >= -135) & (df['longitude'] <= -55)].copy()
print(f"  US reports: {len(df_us)}")

import netCDF4 as nc
ds = nc.Dataset(os.path.join(DATA_DIR, "etopo_subset.nc"))
if 'y' in ds.variables:
    elev_lats = ds.variables['y'][:]
    elev_lons = ds.variables['x'][:]
else:
    elev_lats = ds.variables['lat'][:]
    elev_lons = ds.variables['lon'][:]
elevation = ds.variables['z'][:]
ds.close()

bases_df = pd.read_csv(os.path.join(DATA_DIR, "military_bases_us.csv"))
base_lats = bases_df['lat'].values
base_lons = bases_df['lon'].values
base_tree = cKDTree(np.column_stack([base_lats, base_lons]))

counties_df = pd.read_csv(os.path.join(DATA_DIR, "county_centroids_pop.csv"))
counties_lat = counties_df['lat'].values
counties_lon = counties_df['lon'].values
counties_pop = counties_df['pop'].values
county_tree = cKDTree(np.column_stack([counties_lat, counties_lon]))

print(f"  Data loaded ({time.time()-t_start:.1f}s)")

# --- Section 3: Coastline & canyon detection ---
print("\n[3] Coastline & canyon detection...")

def haversine_km(lat1, lon1, lat2, lon2):
    r = np.radians
    dlat = r(lat2 - lat1)
    dlon = r(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(r(lat1))*np.cos(r(lat2))*np.sin(dlon/2)**2
    return R_EARTH * 2 * np.arcsin(np.sqrt(a))

coast_mask_arr = np.zeros_like(elevation, dtype=bool)
nrows, ncols = elevation.shape
for i in range(1, nrows-1):
    for j in range(1, ncols-1):
        if elevation[i, j] < 0:
            neighbors = elevation[i-1:i+2, j-1:j+2]
            if np.any(neighbors >= 0):
                coast_mask_arr[i, j] = True
coast_i, coast_j = np.where(coast_mask_arr)
coast_lats = elev_lats[coast_i]
coast_lons = elev_lons[coast_j]
coast_tree = cKDTree(np.column_stack([coast_lats, coast_lons]))

from scipy.ndimage import label as ndimage_label
shelf_mask = (elevation < 0) & (elevation > -500)
lat_res = abs(float(elev_lats[1] - elev_lats[0]))
lon_res = abs(float(elev_lons[1] - elev_lons[0]))
grad_y = np.abs(np.gradient(elevation, axis=0)) / (lat_res * 111.0)
grad_x = np.abs(np.gradient(elevation, axis=1)) / (lon_res * 111.0 * np.cos(np.radians(elev_lats[:, None])))
gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
steep = (gradient_mag > CANYON_GRADIENT_THRESHOLD) & shelf_mask
labeled, n_feat = ndimage_label(steep)
for c_id in range(1, n_feat+1):
    if np.sum(labeled == c_id) < 3:
        steep[labeled == c_id] = False
canyon_i, canyon_j = np.where(steep)
canyon_lats = elev_lats[canyon_i]
canyon_lons = elev_lons[canyon_j]
canyon_tree = cKDTree(np.column_stack([canyon_lats, canyon_lons]))
print(f"  Canyon cells: {len(canyon_lats)}")

shelf_i, shelf_j = np.where(shelf_mask)
shelf_lats = elev_lats[shelf_i]
shelf_lons = elev_lons[shelf_j]
shelf_gradients = gradient_mag[shelf_i, shelf_j]
print(f"  Shelf cells: {len(shelf_lats):,}")

ocean_mask = elevation < 0
ocean_i, ocean_j = np.where(ocean_mask)
ocean_lats = elev_lats[ocean_i]
ocean_lons = elev_lons[ocean_j]
ocean_depths = elevation[ocean_i, ocean_j]
ocean_tree = cKDTree(np.column_stack([ocean_lats, ocean_lons]))

# --- Section 4: Coastal filtering & UAP metrics ---
print("\n[4] Coastal filtering & UAP metrics...")
uap_lats = df_us['latitude'].values
uap_lons = df_us['longitude'].values
_, coast_idx = coast_tree.query(np.column_stack([uap_lats, uap_lons]), k=1)
coast_km = haversine_km(uap_lats, uap_lons, coast_lats[coast_idx], coast_lons[coast_idx])
coastal_mask = coast_km <= COASTAL_BAND_KM
df_coastal = df_us[coastal_mask].copy()
df_coastal['dist_to_coast_km'] = coast_km[coastal_mask]
coastal_lats = df_coastal['latitude'].values
coastal_lons = df_coastal['longitude'].values
print(f"  Coastal reports: {len(df_coastal)}")

_, uap_canyon_idx = canyon_tree.query(np.column_stack([coastal_lats, coastal_lons]), k=1)
df_coastal['dist_to_canyon_km'] = haversine_km(
    coastal_lats, coastal_lons, canyon_lats[uap_canyon_idx], canyon_lons[uap_canyon_idx])

_, uap_ocean_idx = ocean_tree.query(np.column_stack([coastal_lats, coastal_lons]), k=1)
df_coastal['depth_nearest_ocean'] = ocean_depths[uap_ocean_idx]

base_dists_deg, _ = base_tree.query(np.column_stack([coastal_lats, coastal_lons]), k=1)
df_coastal['dist_to_military_km'] = base_dists_deg * 111.0

uap_county_dists, uap_county_idx = county_tree.query(
    np.column_stack([coastal_lats, coastal_lons]), k=5)
uap_pop = np.zeros(len(coastal_lats))
for k in range(5):
    d_km = uap_county_dists[:, k] * 111.0 + 1.0
    uap_pop += counties_pop[uap_county_idx[:, k]] / (d_km**2)
df_coastal['pop_density_proxy'] = uap_pop

# Port tree
data_ports = np.load(PORT_CACHE_FILE, allow_pickle=True)
port_coords = data_ports['port_coords']
port_tree = BallTree(np.radians(port_coords), metric='haversine')

uap_coords_rad = np.radians(np.column_stack([coastal_lats, coastal_lons]))
port_dists_rad, _ = port_tree.query(uap_coords_rad, k=1)
df_coastal['dist_to_nearest_port'] = port_dists_rad.flatten() * R_EARTH
counts_25_uap = port_tree.query_radius(uap_coords_rad, r=25.0/R_EARTH, count_only=True)
df_coastal['log_port_count_25km'] = np.log1p(counts_25_uap)

print(f"  UAP metrics done ({time.time()-t_start:.1f}s)")

# --- Section 5: Control generation ---
print("\n[5] Control generation...")

def generate_controls_with_scores(n_ctrl, land_w, ocean_w, seed=42):
    rng = np.random.RandomState(seed)
    grid_lat = np.linspace(22, 52, 300)
    grid_lon = np.linspace(-130, -60, 600)
    glat, glon = np.meshgrid(grid_lat, grid_lon, indexing='ij')
    glat_flat, glon_flat = glat.flatten(), glon.flatten()
    coast_dists_grid, _ = coast_tree.query(np.column_stack([glat_flat, glon_flat]), k=1)
    coast_km_grid = coast_dists_grid * 111.0
    coastal_grid_mask = coast_km_grid <= COASTAL_BAND_KM
    glat_coastal = glat_flat[coastal_grid_mask]
    glon_coastal = glon_flat[coastal_grid_mask]

    grid_elev_idx = np.argmin(np.abs(elev_lats[:, None] - glat_coastal[None, :]), axis=0)
    grid_elon_idx = np.argmin(np.abs(elev_lons[:, None] - glon_coastal[None, :]), axis=0)
    grid_elev = elevation[grid_elev_idx, grid_elon_idx]
    land_weight = np.where(grid_elev >= 0, land_w, ocean_w)

    county_dists, county_idx = county_tree.query(
        np.column_stack([glat_coastal, glon_coastal]), k=5)
    pop_weight = np.zeros(len(glat_coastal))
    for kk in range(5):
        dd = county_dists[:, kk] * 111.0 + 1.0
        pop_weight += counties_pop[county_idx[:, kk]] / (dd**2)

    raw_weights = pop_weight * land_weight
    probs = raw_weights / raw_weights.sum()
    chosen = rng.choice(len(glat_coastal), size=n_ctrl, replace=True, p=probs)

    jitter_val = 0.12
    c_lats = glat_coastal[chosen] + rng.uniform(-jitter_val, jitter_val, n_ctrl)
    c_lons = glon_coastal[chosen] + rng.uniform(-jitter_val, jitter_val, n_ctrl)
    sampling_scores = raw_weights[chosen]

    _, cc_idx = coast_tree.query(np.column_stack([c_lats, c_lons]), k=1)
    cc_km = haversine_km(c_lats, c_lons, coast_lats[cc_idx], coast_lons[cc_idx])
    mask = cc_km <= COASTAL_BAND_KM
    return c_lats[mask], c_lons[mask], cc_km[mask], sampling_scores[mask]

ctrl_lats, ctrl_lons, ctrl_coast_km, ctrl_sampling_scores = generate_controls_with_scores(
    N_CONTROL, land_w=3.0, ocean_w=0.05, seed=42)
print(f"  Controls: {len(ctrl_lats)}")

# Control features
_, c_ocean_idx = ocean_tree.query(np.column_stack([ctrl_lats, ctrl_lons]), k=1)
_, c_canyon_idx = canyon_tree.query(np.column_stack([ctrl_lats, ctrl_lons]), k=1)
c_base_dists, _ = base_tree.query(np.column_stack([ctrl_lats, ctrl_lons]), k=1)
c_county_dists, c_county_idx = county_tree.query(np.column_stack([ctrl_lats, ctrl_lons]), k=5)
c_pop = np.zeros(len(ctrl_lats))
for k in range(5):
    d_km = c_county_dists[:, k] * 111.0 + 1.0
    c_pop += counties_pop[c_county_idx[:, k]] / (d_km**2)

ctrl_coords_rad = np.radians(np.column_stack([ctrl_lats, ctrl_lons]))
ctrl_port_dists_rad, _ = port_tree.query(ctrl_coords_rad, k=1)
ctrl_port_count_25 = port_tree.query_radius(ctrl_coords_rad, r=25.0/R_EARTH, count_only=True)

# --- Section 6: Feature assembly ---
print("\n[6] Feature assembly...")
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
    'dist_to_canyon_km': haversine_km(ctrl_lats, ctrl_lons,
                                       canyon_lats[c_canyon_idx], canyon_lons[c_canyon_idx]),
    'dist_to_coast_km': ctrl_coast_km,
    'dist_to_military_km': c_base_dists * 111.0,
    'pop_density_proxy': c_pop,
    'depth_nearest_ocean': ocean_depths[c_ocean_idx],
    'dist_to_nearest_port': ctrl_port_dists_rad.flatten() * R_EARTH,
    'log_port_count_25km': np.log1p(ctrl_port_count_25),
})

X_raw_pre = pd.concat([X_uap_raw, X_ctrl_raw], ignore_index=True)
y_raw_pre = np.concatenate([np.ones(len(X_uap_raw)), np.zeros(len(X_ctrl_raw))])
sampling_weights_pre = np.concatenate([np.ones(len(X_uap_raw)), ctrl_sampling_scores])

valid_mask = X_raw_pre.notna().all(axis=1)
X_raw = X_raw_pre[valid_mask].reset_index(drop=True)
y_raw = y_raw_pre[valid_mask.values]
sampling_weights = sampling_weights_pre[valid_mask.values]
print(f"  Dataset: {len(y_raw):,} ({int(y_raw.sum()):,} UAP, {int((1-y_raw).sum()):,} ctrl)")

# --- Section 7: Gradient computation ---
print("\n[7] Gradient computation...")
gradient_by_cell = defaultdict(list)
for i in range(len(shelf_lats)):
    lat_bin = round(shelf_lats[i] / GRID_RES) * GRID_RES
    lon_bin = round(shelf_lons[i] / GRID_RES) * GRID_RES
    gradient_by_cell[(lat_bin, lon_bin)].append(float(shelf_gradients[i]))

def get_p95_gradient_gridded(coords, grad_cells, grid_res=0.1, radius_km=25):
    n = len(coords)
    result = np.zeros(n)
    r_deg = radius_km / 111.0
    for idx in range(n):
        if idx > 0 and idx % 20000 == 0:
            print(f"    Gradient p95 lookup: {idx}/{n}")
        lat, lon = coords[idx]
        lat_lo = round((lat - r_deg) / grid_res) * grid_res
        lat_hi = round((lat + r_deg) / grid_res) * grid_res
        lon_lo = round((lon - r_deg) / grid_res) * grid_res
        lon_hi = round((lon + r_deg) / grid_res) * grid_res
        vals = []
        la = lat_lo
        while la <= lat_hi + 1e-9:
            lo = lon_lo
            while lo <= lon_hi + 1e-9:
                key = (round(la / grid_res) * grid_res, round(lo / grid_res) * grid_res)
                if key in grad_cells:
                    vals.extend(grad_cells[key])
                lo += grid_res
            la += grid_res
        if vals:
            result[idx] = np.percentile(vals, 95)
    return result

all_coords = np.concatenate([
    np.column_stack([coastal_lats, coastal_lons]),
    np.column_stack([ctrl_lats, ctrl_lons])
])
p95_gradient_all_pre = get_p95_gradient_gridded(
    all_coords, gradient_by_cell, grid_res=GRID_RES, radius_km=GRADIENT_RADIUS_KM)
p95_gradient_all = p95_gradient_all_pre[valid_mask.values]

print(f"  Section 7 done ({time.time()-t_start:.1f}s)")


# ============================================================
# SECTION 8: BINNED WEIGHTED OR WITH BOOTSTRAP CI
# ============================================================
print("\n" + "=" * 70)
print("BINNED WEIGHTED OR — 4 GRADIENT CATEGORIES + BOOTSTRAP CI")
print("=" * 70)

# Assign bins
bin_idx = np.digitize(p95_gradient_all, BIN_EDGES) - 1
bin_idx = np.clip(bin_idx, 0, len(BIN_LABELS) - 1)

is_uap = (y_raw == 1)
is_ctrl = (y_raw == 0)
ctrl_w_raw = 1.0 / sampling_weights[is_ctrl]  # importance weight = 1/sampling_score

# Winsorize: cap at 95th percentile
cap = np.percentile(ctrl_w_raw, WINSORIZE_PCT)
ctrl_w_capped = np.minimum(ctrl_w_raw, cap)
print(f"  Importance weights: median={np.median(ctrl_w_raw):.4f}, p95={cap:.4f}, max={ctrl_w_raw.max():.4f}")
print(f"  After winsorize: max={ctrl_w_capped.max():.4f}")

# Check bin sizes
bin_idx_check = np.digitize(p95_gradient_all, BIN_EDGES) - 1
bin_idx_check = np.clip(bin_idx_check, 0, len(BIN_LABELS) - 1)
for b in range(len(BIN_LABELS)):
    n_u = (bin_idx_check[is_uap] == b).sum()
    n_c = (bin_idx_check[is_ctrl] == b).sum()
    print(f"  Bin {BIN_LABELS[b]:>25s}: {n_u} UAP, {n_c} ctrl")


def compute_weighted_or_table(uap_mask, ctrl_mask, ctrl_weights_vec,
                              gradient_vals, ref_bin=REF_BIN):
    """Compute weighted OR per bin. Returns dict of bin -> OR."""
    uap_bins = np.digitize(gradient_vals[uap_mask], BIN_EDGES) - 1
    uap_bins = np.clip(uap_bins, 0, len(BIN_LABELS) - 1)
    ctrl_bins_arr = np.digitize(gradient_vals[ctrl_mask], BIN_EDGES) - 1
    ctrl_bins_arr = np.clip(ctrl_bins_arr, 0, len(BIN_LABELS) - 1)

    ref_n_uap = (uap_bins == ref_bin).sum()
    ref_w_ctrl = ctrl_weights_vec[ctrl_bins_arr == ref_bin].sum()
    if ref_w_ctrl == 0 or ref_n_uap == 0:
        return {b: np.nan for b in range(len(BIN_LABELS))}
    ref_odds = ref_n_uap / ref_w_ctrl

    ors = {}
    for b in range(len(BIN_LABELS)):
        n_u = (uap_bins == b).sum()
        w_c = ctrl_weights_vec[ctrl_bins_arr == b].sum()
        if w_c == 0:
            ors[b] = np.nan
        else:
            ors[b] = (n_u / w_c) / ref_odds
    return ors


def run_bootstrap_or(ctrl_weights_vec, label=""):
    """Run point estimate + bootstrap for given weights."""
    point_ors = compute_weighted_or_table(is_uap, is_ctrl, ctrl_weights_vec, p95_gradient_all)

    print(f"\n  Running {N_BOOTSTRAP} bootstrap resamples ({label})...")
    rng = np.random.RandomState(RNG_SEED)
    n_total = len(y_raw)

    boot_ors = {b: [] for b in range(len(BIN_LABELS))}
    for rep in range(N_BOOTSTRAP):
        if rep > 0 and rep % 500 == 0:
            print(f"    Bootstrap {rep}/{N_BOOTSTRAP}")
        idx = rng.choice(n_total, size=n_total, replace=True)
        y_b = y_raw[idx]
        grad_b = p95_gradient_all[idx]
        sw_b = sampling_weights[idx]

        uap_mask_b = (y_b == 1)
        ctrl_mask_b = (y_b == 0)
        w_b = 1.0 / sw_b[ctrl_mask_b]
        if label == "capped":
            w_b = np.minimum(w_b, cap)

        ors_b = compute_weighted_or_table(uap_mask_b, ctrl_mask_b, w_b, grad_b)
        for b in range(len(BIN_LABELS)):
            boot_ors[b].append(ors_b[b])

    table = []
    print(f"\n  {label.upper():>10s} {'Bin':>25s} {'N_UAP':>7s} {'N_ctrl':>7s} {'Wtd OR':>8s} {'95% CI':>20s}")
    print("  " + "-" * 80)
    for b in range(len(BIN_LABELS)):
        n_u = int((bin_idx[is_uap] == b).sum())
        n_c = int((bin_idx[is_ctrl] == b).sum())
        or_val = point_ors[b]

        boot_vals = [v for v in boot_ors[b] if not np.isnan(v) and np.isfinite(v)]
        if len(boot_vals) >= 100:
            ci_lo = np.percentile(boot_vals, 2.5)
            ci_hi = np.percentile(boot_vals, 97.5)
        else:
            ci_lo, ci_hi = np.nan, np.nan

        or_str = f"{or_val:.2f}" if not np.isnan(or_val) else "N/A"
        ci_str = f"[{ci_lo:.2f}, {ci_hi:.2f}]" if not np.isnan(ci_lo) else "N/A"
        print(f"  {label:>10s} {BIN_LABELS[b]:>25s} {n_u:>7d} {n_c:>7d} {or_str:>8s} {ci_str:>20s}")

        table.append({
            'bin': BIN_LABELS[b],
            'gradient_range': f"{BIN_EDGES[b]}-{BIN_EDGES[b+1]}",
            'n_uap': n_u, 'n_ctrl': n_c,
            'weighted_or': float(or_val) if not np.isnan(or_val) else None,
            'ci_lo': float(ci_lo) if not np.isnan(ci_lo) else None,
            'ci_hi': float(ci_hi) if not np.isnan(ci_hi) else None,
        })
    return table

# Assign bins
bin_idx = np.digitize(p95_gradient_all, BIN_EDGES) - 1
bin_idx = np.clip(bin_idx, 0, len(BIN_LABELS) - 1)

# Run both uncapped and capped
results_uncapped = run_bootstrap_or(ctrl_w_raw, label="uncapped")
results_capped = run_bootstrap_or(ctrl_w_capped, label="capped")

# Raw (unweighted) OR for comparison
print(f"\n{'Bin':>25s} {'Raw OR':>8s}")
print("-" * 40)
uap_bins_raw = bin_idx[is_uap]
ctrl_bins_raw = bin_idx[is_ctrl]
ref_n_u = (uap_bins_raw == REF_BIN).sum()
ref_n_c = (ctrl_bins_raw == REF_BIN).sum()
ref_raw = ref_n_u / ref_n_c if ref_n_c > 0 else np.nan
for b in range(len(BIN_LABELS)):
    n_u = (uap_bins_raw == b).sum()
    n_c = (ctrl_bins_raw == b).sum()
    raw_or = (n_u / n_c) / ref_raw if n_c > 0 else np.nan
    print(f"  {BIN_LABELS[b]:>23s} {raw_or:>8.2f}")
    results_capped[b]['raw_or'] = float(raw_or) if not np.isnan(raw_or) else None
    results_uncapped[b]['raw_or'] = float(raw_or) if not np.isnan(raw_or) else None

# Save
output = {
    'capped': {
        'gradient_categories': results_capped,
        'winsorize_pct': WINSORIZE_PCT,
        'cap_value': float(cap),
    },
    'uncapped': {
        'gradient_categories': results_uncapped,
    },
    'reference_bin': BIN_LABELS[REF_BIN],
    'n_bootstrap': N_BOOTSTRAP,
    'note': 'Weighted OR uses importance weights (1/sampling_score) to correct for 60x land_weight asymmetry. Capped = weights winsorized at p95.',
    'for_reddit_post': {
        'very_steep_capped': {
            'or': results_capped[3]['weighted_or'],
            'ci_lo': results_capped[3]['ci_lo'],
            'ci_hi': results_capped[3]['ci_hi'],
            'label': 'gradient > 60 m/km (winsorized weights)',
        },
        'steep_capped': {
            'or': results_capped[2]['weighted_or'],
            'ci_lo': results_capped[2]['ci_lo'],
            'ci_hi': results_capped[2]['ci_hi'],
            'label': 'gradient 30-60 m/km (winsorized weights)',
        },
    },
    'metadata': {
        'script': '05_weighted_or_for_post.py',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'runtime_seconds': time.time() - t_start,
    }
}

out_path = os.path.join(RESULTS_DIR, "weighted_or_binned.json")
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved {out_path}")
print(f"  Total runtime: {time.time()-t_start:.1f}s")
print("\nDONE.")
