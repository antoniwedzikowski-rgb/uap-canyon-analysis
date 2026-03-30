#!/usr/bin/env python3
"""
CRIT-4 fix: Threshold sensitivity analysis for 60 m/km
======================================================
The 60 m/km threshold was derived from Sprint 3 dose-response bins
(data-derived, not pre-registered). This script tests whether the
S-logR correlation is specific to 60 m/km or robust across thresholds.

Tests thresholds: 20, 30, 40, 50, 60, 70, 80, 100 m/km.
For each, recomputes S from scratch (new canyon detection, new rankings)
and evaluates Spearman(S, logR) on West Coast.
"""

import os, time, json, warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.ndimage import label as ndimage_label
from scipy.stats import spearmanr, rankdata
from collections import defaultdict

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR = os.environ.get("UAP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "results", "phase_ev2")

R_EARTH = 6371.0
GRID_DEG = 0.5
MIN_REPORTS = 20
WEST_COAST_LON_MAX = -115.0
WEST_COAST_LAT_MIN = 30.0
COAST_KM = 200
MIN_COMPONENT_SIZE = 3

def elapsed():
    return f"{time.time()-t0:.1f}s"

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.asarray, [lat1, lon1, lat2, lon2])
    r = np.radians
    dlat = r(lat2 - lat1)
    dlon = r(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(r(lat1))*np.cos(r(lat2))*np.sin(dlon/2)**2
    return R_EARTH * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

print("=" * 70)
print("CRIT-4: THRESHOLD SENSITIVITY ANALYSIS")
print("Testing gradient thresholds: 20, 30, 40, 50, 60, 70, 80, 100 m/km")
print("=" * 70)

# ============================================================
# LOAD GEOMETRY
# ============================================================
print(f"\n[LOAD] ETOPO... ({elapsed()})")
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

# Coast
coast_mask_arr = np.zeros_like(elevation, dtype=bool)
nrows, ncols = elevation.shape
for i in range(1, nrows-1):
    for j in range(1, ncols-1):
        if elevation[i, j] < 0:
            if np.any(elevation[i-1:i+2, j-1:j+2] >= 0):
                coast_mask_arr[i, j] = True
coast_i, coast_j = np.where(coast_mask_arr)
coast_lats = elev_lats[coast_i].astype(np.float64)
coast_lons = elev_lons[coast_j].astype(np.float64)
coast_tree = cKDTree(np.column_stack([coast_lats, coast_lons]))

# Shelf + gradient
shelf_mask = (elevation < 0) & (elevation > -500)
lat_res = abs(float(elev_lats[1] - elev_lats[0]))
lon_res = abs(float(elev_lons[1] - elev_lons[0]))
grad_y = np.abs(np.gradient(elevation, axis=0)) / (lat_res * 111.0)
grad_x = np.abs(np.gradient(elevation, axis=1)) / (lon_res * 111.0 *
         np.cos(np.radians(elev_lats[:, None])))
gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

shelf_i, shelf_j = np.where(shelf_mask)
shelf_lats = elev_lats[shelf_i]
shelf_lons = elev_lons[shelf_j]
shelf_gradients = gradient_mag[shelf_i, shelf_j]

# County population (for E_i)
counties_df = pd.read_csv(os.path.join(DATA_DIR, "county_centroids_pop.csv"))
county_lats = counties_df['lat'].values.astype(np.float64)
county_lons = counties_df['lon'].values.astype(np.float64)
county_tree = cKDTree(np.column_stack([county_lats, county_lons]))
counties_pop = counties_df['pop'].values.astype(np.float64)

print(f"  Geometry loaded ({elapsed()})")

# ============================================================
# LOAD NUFORC
# ============================================================
print(f"\n[LOAD] NUFORC... ({elapsed()})")
df_raw = pd.read_csv(os.path.join(DATA_DIR, "nuforc_reports.csv"),
    names=['datetime_str','city','state','country','shape',
           'duration_seconds','duration_text','description','date_posted','lat','lon'],
    header=None, low_memory=False)
df_raw['lat'] = pd.to_numeric(df_raw['lat'], errors='coerce')
df_raw['lon'] = pd.to_numeric(df_raw['lon'], errors='coerce')
df_raw = df_raw.dropna(subset=['lat','lon'])
df_raw = df_raw[(df_raw['lat'] != 0) & (df_raw['lon'] != 0)]
df_raw['_dt'] = pd.to_datetime(df_raw['datetime_str'], errors='coerce')
df_raw['_year'] = df_raw['_dt'].dt.year
df_raw = df_raw[(df_raw['_year'] >= 1990) & (df_raw['_year'] <= 2014)]
df_raw = df_raw[(df_raw['lat'] >= 20) & (df_raw['lat'] <= 55) &
                (df_raw['lon'] >= -135) & (df_raw['lon'] <= -55)]

# Filter to West Coast coastal band
_, c_idx = coast_tree.query(np.column_stack([df_raw['lat'].values, df_raw['lon'].values]), k=1)
d_coast = haversine_km(df_raw['lat'].values, df_raw['lon'].values,
                        coast_lats[c_idx], coast_lons[c_idx])
df_band = df_raw[d_coast <= COAST_KM].copy()
df_west = df_band[(df_band['lon'] <= WEST_COAST_LON_MAX) &
                   (df_band['lat'] >= WEST_COAST_LAT_MIN)].copy()
print(f"  West Coast reports in {COAST_KM}km band: {len(df_west)}")

# E_i computation (reuse from phase_e_red_v2)
def compute_expected(cell_lat, cell_lon, half_deg):
    grid_lat = np.linspace(cell_lat - half_deg, cell_lat + half_deg, 30)
    grid_lon = np.linspace(cell_lon - half_deg, cell_lon + half_deg, 30)
    glat, glon = np.meshgrid(grid_lat, grid_lon, indexing='ij')
    gf, gln = glat.flatten(), glon.flatten()
    _, ci = coast_tree.query(np.column_stack([gf, gln]), k=1)
    cd = haversine_km(gf, gln, coast_lats[ci], coast_lons[ci])
    mask = cd <= COAST_KM
    gc_lat, gc_lon = gf[mask], gln[mask]
    if len(gc_lat) < 5:
        return 0.0
    n_k = min(10, len(counties_pop))
    _, ci_c = county_tree.query(np.column_stack([gc_lat, gc_lon]), k=n_k)
    w = np.zeros(len(gc_lat))
    for k in range(n_k):
        d = haversine_km(gc_lat, gc_lon, county_lats[ci_c[:, k]], county_lons[ci_c[:, k]]) + 1.0
        w += counties_pop[ci_c[:, k]] / (d**2)
    lat_i = np.clip(np.searchsorted(elev_lats, gc_lat), 0, len(elev_lats)-1)
    lon_i = np.clip(np.searchsorted(elev_lons, gc_lon), 0, len(elev_lons)-1)
    lw = np.where(elevation[lat_i, lon_i] >= 0, 3.0, 0.05)
    w *= lw
    return float(w.sum())


# ============================================================
# SWEEP THRESHOLDS
# ============================================================
thresholds = [20, 30, 40, 50, 60, 70, 80, 100]
half = GRID_DEG / 2

# Pre-build gradient cell lookup
GRID_RES_FINE = 0.1
gradient_by_cell = defaultdict(list)
for idx in range(len(shelf_lats)):
    lat_bin = round(shelf_lats[idx] / GRID_RES_FINE) * GRID_RES_FINE
    lon_bin = round(shelf_lons[idx] / GRID_RES_FINE) * GRID_RES_FINE
    gradient_by_cell[(round(lat_bin, 1), round(lon_bin, 1))].append(float(shelf_gradients[idx]))

def get_p95_gradient(lat, lon):
    lat_bin = round(lat / GRID_RES_FINE) * GRID_RES_FINE
    lon_bin = round(lon / GRID_RES_FINE) * GRID_RES_FINE
    grads = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            key = (round((lat_bin + di*GRID_RES_FINE) * 10) / 10,
                   round((lon_bin + dj*GRID_RES_FINE) * 10) / 10)
            grads.extend(gradient_by_cell.get(key, []))
    return float(np.percentile(grads, 95)) if grads else 0.0

results = []

print(f"\n{'='*70}")
print(f"  {'Threshold':>10} {'n_steep':>8} {'n_hot':>6} {'n_cold':>7} {'rho':>8} {'p':>10} {'sig':>4}")
print(f"  {'-'*60}")

for thresh in thresholds:
    # Canyon detection at this threshold
    steep = (gradient_mag > thresh) & shelf_mask
    labeled, n_feat = ndimage_label(steep)
    for c_id in range(1, n_feat + 1):
        if np.sum(labeled == c_id) < MIN_COMPONENT_SIZE:
            steep[labeled == c_id] = False
    labeled, n_feat = ndimage_label(steep)
    canyon_i, canyon_j = np.where(steep)
    c_lats = elev_lats[canyon_i]
    c_lons = elev_lons[canyon_j]

    if len(c_lats) < 3:
        print(f"  {thresh:>10} m/km: too few steep cells ({len(c_lats)})")
        results.append({'threshold': thresh, 'n_steep': len(c_lats), 'rho': None, 'p': None})
        continue

    # Compute G, P, C for steep cells
    G_vals = np.array([get_p95_gradient(lat, lon) for lat, lon in zip(c_lats, c_lons)])
    _, ci = coast_tree.query(np.column_stack([c_lats, c_lons]), k=1)
    d_shore = haversine_km(c_lats, c_lons, coast_lats[ci], coast_lons[ci])
    P_vals = np.exp(-d_shore / 50.0)
    approx_deg_25 = 25.0 / 111.0
    steep_tree_t = cKDTree(np.column_stack([c_lats, c_lons]))
    C_vals = np.zeros(len(c_lats))
    for idx in range(len(c_lats)):
        C_vals[idx] = len(coast_tree.query_ball_point([c_lats[idx], c_lons[idx]], approx_deg_25))

    # Rank and score
    def grank(v):
        return (rankdata(v) - 1) / max(len(v) - 1, 1)
    cell_score = grank(G_vals) + grank(P_vals) + grank(C_vals)

    # Aggregate to 0.5° grid (West Coast only)
    grid_lat_edges = np.arange(WEST_COAST_LAT_MIN, 50.01, GRID_DEG)
    grid_lon_edges = np.arange(-130.0, WEST_COAST_LON_MAX + 0.01, GRID_DEG)

    grid_cells = []
    approx_50 = 50.0 / 111.0

    for lat_c in grid_lat_edges + half:
        for lon_c in grid_lon_edges + half:
            _, ci2 = coast_tree.query([lat_c, lon_c], k=1)
            dc = haversine_km(lat_c, lon_c, float(coast_lats[ci2]), float(coast_lons[ci2]))
            if dc > COAST_KM:
                continue

            nearby = steep_tree_t.query_ball_point([lat_c, lon_c], approx_50)
            S = float(np.mean(cell_score[nearby])) if nearby else 0.0

            # Count UAP
            in_cell = ((df_west['lat'].values >= lat_c - half) &
                       (df_west['lat'].values < lat_c + half) &
                       (df_west['lon'].values >= lon_c - half) &
                       (df_west['lon'].values < lon_c + half))
            O_i = int(in_cell.sum())

            grid_cells.append({'lat': lat_c, 'lon': lon_c, 'S': S, 'O_i': O_i})

    # Compute E_i
    for gc in grid_cells:
        gc['E_i_raw'] = compute_expected(gc['lat'], gc['lon'], half)

    total_O = sum(gc['O_i'] for gc in grid_cells)
    total_E_raw = sum(gc['E_i_raw'] for gc in grid_cells)
    if total_E_raw == 0:
        results.append({'threshold': thresh, 'n_steep': len(c_lats), 'rho': None, 'p': None})
        continue

    scale = total_O / total_E_raw
    for gc in grid_cells:
        gc['E_i'] = gc['E_i_raw'] * scale

    testable = [gc for gc in grid_cells if gc['O_i'] >= MIN_REPORTS and gc['E_i'] > 0]
    for gc in testable:
        gc['R_i'] = gc['O_i'] / gc['E_i']
        gc['logR'] = np.log(gc['R_i']) if gc['R_i'] > 0 else -4.6

    n_hot = sum(1 for gc in testable if gc['S'] > 0)
    n_cold = sum(1 for gc in testable if gc['S'] == 0)

    if len(testable) < 10:
        results.append({'threshold': thresh, 'n_steep': len(c_lats),
                        'n_testable': len(testable), 'rho': None, 'p': None})
        continue

    S_arr = np.array([gc['S'] for gc in testable])
    logR_arr = np.array([gc['logR'] for gc in testable])
    rho, p = spearmanr(S_arr, logR_arr)

    sig = "*" if p < 0.05 else ""
    print(f"  {thresh:>10} m/km {len(c_lats):>8} {n_hot:>6} {n_cold:>7} {rho:>+8.3f} {p:>10.6f} {sig:>4}")

    results.append({
        'threshold': thresh,
        'n_steep': int(len(c_lats)),
        'n_testable': len(testable),
        'n_hot': n_hot,
        'n_cold': n_cold,
        'rho': round(float(rho), 4),
        'p': round(float(p), 6),
    })

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")

sig_results = [r for r in results if r.get('p') is not None and r['p'] < 0.05]
all_results = [r for r in results if r.get('rho') is not None]
if all_results:
    rhos = [r['rho'] for r in all_results]
    print(f"  Significant at p<0.05: {len(sig_results)}/{len(all_results)} thresholds")
    print(f"  Rho range: [{min(rhos):.3f}, {max(rhos):.3f}]")
    best = max(all_results, key=lambda r: r['rho'])
    print(f"  Best threshold: {best['threshold']} m/km (rho={best['rho']:.3f})")
    if best['threshold'] == 60:
        print(f"  NOTE: 60 m/km is the best threshold — consistent with double-dipping concern")
    else:
        print(f"  NOTE: Best threshold differs from 60 m/km — result is not threshold-specific")

# Save
out = {
    'test': 'CRIT-4: Threshold sensitivity analysis',
    'thresholds_tested': thresholds,
    'results': results,
}
out_file = os.path.join(OUT_DIR, "phase_e_threshold_sensitivity.json")
with open(out_file, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_file}")
print(f"DONE ({elapsed()})")
