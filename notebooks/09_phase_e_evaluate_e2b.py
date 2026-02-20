#!/usr/bin/env python3
"""
Phase E2b + E2c: Evaluate Frozen Predictions With CONUS Footprint Mask
========================================================================
Design fix documented in E2b_note.md.

E2a found that all 20 hot predictions fell outside the NUFORC data footprint
(Bahamas, Cuba, BC, Isla Guadalupe). This script applies a CONUS bounding box
mask BEFORE selecting top-20 / bottom-20, then evaluates.

E2b: CONUS bbox (24.5-49.0N, -125.0 to -66.0W) + coastal band, no N filter
E2c: Same as E2b + cells with <10 NUFORC reports excluded from candidate pools

The SCORING FUNCTION, RANKS, AND EVALUATION THRESHOLDS ARE UNCHANGED
from the Phase E freeze tag (phase-e-frozen, commit 09de8d8).
Only the geographic evaluation mask changes.
"""

import os, time, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR = os.environ.get("UAP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "phase_e")
os.makedirs(OUT_DIR, exist_ok=True)

R_EARTH = 6371.0
COASTAL_BAND_KM = 200
GRID_DEG = 0.5
WINSORIZE_PCT = 95
N_BOOTSTRAP = 1000
RNG_SEED = 42
CANYON_GRADIENT_THRESHOLD = 20.0
GRID_RES_FINE = 0.1
MIN_REPORTS_FOR_OR = 20
DEDUP_RADIUS_KM = 200
N_HOT = 20
N_COLD = 20

# E2b CONUS footprint mask
CONUS_LAT_MIN = 24.5
CONUS_LAT_MAX = 49.0
CONUS_LON_MIN = -125.0
CONUS_LON_MAX = -66.0

# E2c measurability threshold
MEASURABILITY_MIN_REPORTS = 10

def elapsed():
    return f"{time.time()-t0:.1f}s"

def haversine_km_vec(lat1, lon1, lat2, lon2):
    r = np.radians
    dlat = r(np.asarray(lat2) - np.asarray(lat1))
    dlon = r(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dlat/2)**2 + np.cos(r(lat1))*np.cos(r(lat2))*np.sin(dlon/2)**2
    return R_EARTH * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ============================================================
# LOAD FROZEN GRID SCORES (from phase_e_scoring.py, unchanged)
# ============================================================
print("=" * 70)
print("PHASE E2b/E2c: CONUS FOOTPRINT MASK EVALUATION")
print("Scoring function UNCHANGED from phase-e-frozen (09de8d8)")
print("=" * 70)

grid_file = os.path.join(OUT_DIR, "phase_e_grid.json")
with open(grid_file) as f:
    grid_data = json.load(f)

print(f"  Full grid: {len(grid_data)} cells")


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

print(f"[LOAD] Coast... ({elapsed()})")
coast_mask_arr = np.zeros_like(elevation, dtype=bool)
nrows, ncols = elevation.shape
for i in range(1, nrows-1):
    for j in range(1, ncols-1):
        if elevation[i, j] < 0:
            if np.any(elevation[i-1:i+2, j-1:j+2] >= 0):
                coast_mask_arr[i, j] = True
coast_i, coast_j = np.where(coast_mask_arr)
coast_lats = elev_lats[coast_i]
coast_lons = elev_lons[coast_j]
coast_tree = cKDTree(np.column_stack([coast_lats, coast_lons]))

print(f"[LOAD] Gradient... ({elapsed()})")
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

gradient_by_cell = defaultdict(list)
for idx in range(len(shelf_lats)):
    lat_bin = round(shelf_lats[idx] / GRID_RES_FINE) * GRID_RES_FINE
    lon_bin = round(shelf_lons[idx] / GRID_RES_FINE) * GRID_RES_FINE
    gradient_by_cell[(round(lat_bin, 1), round(lon_bin, 1))].append(
        float(shelf_gradients[idx]))

counties_df = pd.read_csv(os.path.join(DATA_DIR, "county_centroids_pop.csv"))
county_tree = cKDTree(np.column_stack([counties_df['lat'].values, counties_df['lon'].values]))
counties_pop = counties_df['pop'].values

print(f"[LOAD] Geometry complete ({elapsed()})")


# ============================================================
# LOAD NUFORC
# ============================================================
print(f"\n[LOAD] *** OPENING NUFORC DATA *** ({elapsed()})")
nuforc_cols = ['datetime_str','city','state','country','shape',
               'duration_seconds','duration_text','description',
               'date_posted','latitude','longitude']
df_raw = pd.read_csv(os.path.join(DATA_DIR, "nuforc_reports.csv"),
                      names=nuforc_cols, header=None, low_memory=False)
df_raw['latitude'] = pd.to_numeric(df_raw['latitude'], errors='coerce')
df_raw['longitude'] = pd.to_numeric(df_raw['longitude'], errors='coerce')
df_raw = df_raw.dropna(subset=['latitude', 'longitude'])
df_raw = df_raw[(df_raw['latitude'] != 0) & (df_raw['longitude'] != 0)]
df_raw['_dt'] = pd.to_datetime(df_raw['datetime_str'], errors='coerce')
df_raw['_year'] = df_raw['_dt'].dt.year
df_raw = df_raw[(df_raw['_year'] >= 1990) & (df_raw['_year'] <= 2014)]
df_raw = df_raw[(df_raw['latitude'] >= 20) & (df_raw['latitude'] <= 55) &
                (df_raw['longitude'] >= -135) & (df_raw['longitude'] <= -55)]
df_raw = df_raw.reset_index(drop=True)

# Coastal filter
_, c_idx = coast_tree.query(np.column_stack([df_raw['latitude'].values,
                                              df_raw['longitude'].values]), k=1)
d_coast = haversine_km_vec(df_raw['latitude'].values, df_raw['longitude'].values,
                            coast_lats[c_idx], coast_lons[c_idx])
df_coastal = df_raw[d_coast <= COASTAL_BAND_KM].copy()
df_coastal = df_coastal.reset_index(drop=True)
print(f"  NUFORC coastal reports: {len(df_coastal):,}")


# ============================================================
# PRE-COMPUTE: UAP report counts per grid cell
# ============================================================
print(f"\n[PREP] Counting UAP reports per grid cell... ({elapsed()})")
cell_uap_counts = {}
for cell in grid_data:
    lat_c, lon_c = cell['lat'], cell['lon']
    in_cell = ((df_coastal['latitude'].values >= lat_c - GRID_DEG/2) &
               (df_coastal['latitude'].values < lat_c + GRID_DEG/2) &
               (df_coastal['longitude'].values >= lon_c - GRID_DEG/2) &
               (df_coastal['longitude'].values < lon_c + GRID_DEG/2))
    cell_uap_counts[(lat_c, lon_c)] = int(in_cell.sum())

print(f"  Cells with >0 reports: {sum(1 for v in cell_uap_counts.values() if v > 0)}")
print(f"  Cells with >=10 reports: {sum(1 for v in cell_uap_counts.values() if v >= 10)}")
print(f"  Cells with >=20 reports: {sum(1 for v in cell_uap_counts.values() if v >= 20)}")


# ============================================================
# E2b: CONUS FOOTPRINT MASK — RE-SELECT TOP/BOTTOM FROM GRID
# ============================================================
print(f"\n{'='*70}")
print("E2b: CONUS FOOTPRINT MASK (no N filter)")
print(f"  Bbox: {CONUS_LAT_MIN}-{CONUS_LAT_MAX}N, {CONUS_LON_MIN} to {CONUS_LON_MAX}W")
print(f"  Coastal band: {COASTAL_BAND_KM}km")
print(f"{'='*70}")

# Filter grid to CONUS bbox
grid_conus = [c for c in grid_data
              if (CONUS_LAT_MIN <= c['lat'] <= CONUS_LAT_MAX and
                  CONUS_LON_MIN <= c['lon'] <= CONUS_LON_MAX)]

print(f"  Grid cells in CONUS bbox: {len(grid_conus)} (of {len(grid_data)} total)")
n_with_steep = sum(1 for c in grid_conus if c['S'] > 0)
n_without = sum(1 for c in grid_conus if c['S'] == 0)
print(f"  With steep cells (S > 0): {n_with_steep}")
print(f"  Without steep cells (S = 0): {n_without}")


def greedy_dedup(candidates, n_select, descending=True):
    """Greedy selection: pick best, exclude within DEDUP_RADIUS_KM, repeat."""
    sorted_cands = sorted(candidates, key=lambda x: x['S'], reverse=descending)
    selected = []
    for cand in sorted_cands:
        if len(selected) >= n_select:
            break
        too_close = False
        for sel in selected:
            d = haversine_km_vec(cand['lat'], cand['lon'], sel['lat'], sel['lon'])
            if d < DEDUP_RADIUS_KM:
                too_close = True
                break
        if not too_close:
            selected.append(cand)
    return selected


# E2b hot: highest S among CONUS cells with S > 0
hot_candidates_e2b = [c for c in grid_conus if c['S'] > 0]
hot_e2b = greedy_dedup(hot_candidates_e2b, N_HOT, descending=True)

# E2b cold: S == 0 among CONUS cells, prefer closer to coast
cold_candidates_e2b = [c for c in grid_conus if c['S'] == 0]
for c in cold_candidates_e2b:
    _, ci = coast_tree.query([c['lat'], c['lon']], k=1)
    c['_d_coast'] = float(haversine_km_vec(c['lat'], c['lon'],
                                            float(coast_lats[ci]), float(coast_lons[ci])))
cold_candidates_e2b.sort(key=lambda x: x['_d_coast'])
cold_e2b = greedy_dedup(cold_candidates_e2b, N_COLD, descending=False)

print(f"\n  E2b HOT selected: {len(hot_e2b)}")
for i, h in enumerate(hot_e2b):
    n_uap = cell_uap_counts.get((h['lat'], h['lon']), 0)
    print(f"    #{i+1:2d}: ({h['lat']:.2f}, {h['lon']:.2f}) S={h['S']:.3f} "
          f"[{h['n_steep_cells']} steep cells] NUFORC_n={n_uap}")

print(f"\n  E2b COLD selected: {len(cold_e2b)}")
for i, c in enumerate(cold_e2b):
    n_uap = cell_uap_counts.get((c['lat'], c['lon']), 0)
    print(f"    #{i+1:2d}: ({c['lat']:.2f}, {c['lon']:.2f}) S={c['S']:.3f} "
          f"[coast_dist={c['_d_coast']:.1f}km] NUFORC_n={n_uap}")


# ============================================================
# E2c: SAME + MEASURABILITY MASK (N >= 10)
# ============================================================
print(f"\n{'='*70}")
print(f"E2c: CONUS + MEASURABILITY MASK (N >= {MEASURABILITY_MIN_REPORTS})")
print(f"{'='*70}")

grid_conus_measurable = [c for c in grid_conus
                         if cell_uap_counts.get((c['lat'], c['lon']), 0)
                            >= MEASURABILITY_MIN_REPORTS]
print(f"  Measurable cells (N >= {MEASURABILITY_MIN_REPORTS}): {len(grid_conus_measurable)}")

hot_candidates_e2c = [c for c in grid_conus_measurable if c['S'] > 0]
hot_e2c = greedy_dedup(hot_candidates_e2c, N_HOT, descending=True)

cold_candidates_e2c = [c for c in grid_conus_measurable if c['S'] == 0]
for c in cold_candidates_e2c:
    if '_d_coast' not in c:
        _, ci = coast_tree.query([c['lat'], c['lon']], k=1)
        c['_d_coast'] = float(haversine_km_vec(c['lat'], c['lon'],
                                                float(coast_lats[ci]), float(coast_lons[ci])))
cold_candidates_e2c.sort(key=lambda x: x['_d_coast'])
cold_e2c = greedy_dedup(cold_candidates_e2c, N_COLD, descending=False)

print(f"\n  E2c HOT selected: {len(hot_e2c)}")
for i, h in enumerate(hot_e2c):
    n_uap = cell_uap_counts.get((h['lat'], h['lon']), 0)
    print(f"    #{i+1:2d}: ({h['lat']:.2f}, {h['lon']:.2f}) S={h['S']:.3f} "
          f"[{h['n_steep_cells']} steep cells] NUFORC_n={n_uap}")

print(f"\n  E2c COLD selected: {len(cold_e2c)}")
for i, c in enumerate(cold_e2c):
    n_uap = cell_uap_counts.get((c['lat'], c['lon']), 0)
    print(f"    #{i+1:2d}: ({c['lat']:.2f}, {c['lon']:.2f}) S={c['S']:.3f} "
          f"[coast_dist={c.get('_d_coast', 0):.1f}km] NUFORC_n={n_uap}")


# ============================================================
# HELPER: Compute OR for a geographic cell
# ============================================================
BIN_EDGES = [0, 10, 30, 60, 500]
BIN_LABELS = ['0-10 (flat)', '10-30 (moderate)', '30-60 (steep)', '60+ (very steep)']

def get_p95_gradient(lat, lon):
    lat_bin = round(lat / GRID_RES_FINE) * GRID_RES_FINE
    lon_bin = round(lon / GRID_RES_FINE) * GRID_RES_FINE
    grads = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            key = (round((lat_bin + di*GRID_RES_FINE) * 10) / 10,
                   round((lon_bin + dj*GRID_RES_FINE) * 10) / 10)
            grads.extend(gradient_by_cell.get(key, []))
    if not grads:
        return 0.0
    return float(np.percentile(grads, 95))

def compute_cell_or(cell_lat, cell_lon, half_deg=GRID_DEG/2):
    """Compute weighted OR for a single 0.5deg grid cell."""
    in_cell = ((df_coastal['latitude'].values >= cell_lat - half_deg) &
               (df_coastal['latitude'].values < cell_lat + half_deg) &
               (df_coastal['longitude'].values >= cell_lon - half_deg) &
               (df_coastal['longitude'].values < cell_lon + half_deg))
    n_uap_total = in_cell.sum()

    if n_uap_total < MIN_REPORTS_FOR_OR:
        return None

    uap_lats = df_coastal.loc[in_cell, 'latitude'].values
    uap_lons = df_coastal.loc[in_cell, 'longitude'].values

    uap_p95 = np.array([get_p95_gradient(lat, lon) for lat, lon in zip(uap_lats, uap_lons)])
    uap_bins = np.clip(np.digitize(uap_p95, BIN_EDGES) - 1, 0, 3)

    n_flat_uap = (uap_bins == 0).sum()
    n_steep_uap = (uap_bins == 3).sum()

    rng = np.random.RandomState(RNG_SEED)
    n_ctrl = max(500, n_uap_total * 3)
    grid_lat = np.linspace(cell_lat - half_deg, cell_lat + half_deg, 50)
    grid_lon = np.linspace(cell_lon - half_deg, cell_lon + half_deg, 50)
    glat, glon = np.meshgrid(grid_lat, grid_lon, indexing='ij')
    gf, gln = glat.flatten(), glon.flatten()

    cd_grid, _ = coast_tree.query(np.column_stack([gf, gln]), k=1)
    coastal_mask = (cd_grid * 111.0) <= COASTAL_BAND_KM
    gc_lat = gf[coastal_mask]
    gc_lon = gln[coastal_mask]

    if len(gc_lat) < 10:
        return None

    n_k = min(10, len(counties_pop))
    cd_county, ci_county = county_tree.query(np.column_stack([gc_lat, gc_lon]), k=n_k)
    weights = np.zeros(len(gc_lat))
    for k in range(n_k):
        d_km = cd_county[:, k] * 111.0 + 1.0
        weights += counties_pop[ci_county[:, k]] / (d_km**2)

    lat_idx = np.clip(np.searchsorted(elev_lats, gc_lat), 0, len(elev_lats)-1)
    lon_idx = np.clip(np.searchsorted(elev_lons, gc_lon), 0, len(elev_lons)-1)
    ge = elevation[lat_idx, lon_idx]
    lw = np.where(ge >= 0, 3.0, 0.05)
    weights *= lw
    if weights.sum() == 0:
        return None
    weights = weights / weights.sum()

    chosen = rng.choice(len(gc_lat), size=n_ctrl, replace=True, p=weights)
    jitter = 0.05
    c_lats = gc_lat[chosen] + rng.uniform(-jitter, jitter, n_ctrl)
    c_lons = gc_lon[chosen] + rng.uniform(-jitter, jitter, n_ctrl)
    ctrl_w = weights[chosen] * len(gc_lat)

    ctrl_p95 = np.array([get_p95_gradient(lat, lon) for lat, lon in zip(c_lats, c_lons)])
    ctrl_bins = np.clip(np.digitize(ctrl_p95, BIN_EDGES) - 1, 0, 3)

    n_flat_ctrl = (ctrl_bins == 0).sum()
    n_steep_ctrl = (ctrl_bins == 3).sum()

    if n_flat_uap < 5 or n_flat_ctrl < 5:
        if n_steep_ctrl == 0:
            return None
        ratio = (n_steep_uap / n_uap_total) / (n_steep_ctrl / n_ctrl)
        return {
            'or_60plus': round(ratio, 3),
            'ci_lo': None,
            'ci_hi': None,
            'n_uap': n_uap_total,
            'n_ctrl': n_ctrl,
            'n_uap_steep': int(n_steep_uap),
            'n_ctrl_steep': int(n_steep_ctrl),
            'method': 'simple_ratio',
        }

    cd_u, ci_u = county_tree.query(np.column_stack([uap_lats, uap_lons]), k=5)
    uap_pop = np.zeros(len(uap_lats))
    for k in range(5):
        d_km = cd_u[:, k] * 111.0 + 1.0
        uap_pop += counties_pop[ci_u[:, k]] / (d_km**2)

    flat_u = uap_bins == 0
    flat_c = ctrl_bins == 0
    steep_u = uap_bins == 3
    steep_c = ctrl_bins == 3

    if n_steep_uap < 3 or n_steep_ctrl < 3:
        return {
            'or_60plus': 0.0, 'ci_lo': 0.0, 'ci_hi': 0.0,
            'n_uap': n_uap_total, 'n_ctrl': n_ctrl,
            'n_uap_steep': int(n_steep_uap), 'n_ctrl_steep': int(n_steep_ctrl),
            'method': 'too_few_steep',
        }

    iw_flat_u = 1.0 / (uap_pop[flat_u] + 1e-10)
    cap = np.percentile(iw_flat_u, WINSORIZE_PCT)
    iw_flat_u = np.minimum(iw_flat_u, cap)

    iw_flat_c = 1.0 / (ctrl_w[flat_c] + 1e-10)
    cap_c = np.percentile(iw_flat_c, WINSORIZE_PCT)
    iw_flat_c = np.minimum(iw_flat_c, cap_c)

    iw_steep_u = 1.0 / (uap_pop[steep_u] + 1e-10)
    cap_su = np.percentile(iw_steep_u, WINSORIZE_PCT)
    iw_steep_u = np.minimum(iw_steep_u, cap_su)

    iw_steep_c = 1.0 / (ctrl_w[steep_c] + 1e-10)
    cap_sc = np.percentile(iw_steep_c, WINSORIZE_PCT)
    iw_steep_c = np.minimum(iw_steep_c, cap_sc)

    flat_u_w = iw_flat_u.sum()
    flat_c_w = iw_flat_c.sum()
    steep_u_w = iw_steep_u.sum()
    steep_c_w = iw_steep_c.sum()

    if flat_u_w == 0 or flat_c_w == 0 or steep_c_w == 0:
        return None

    or_val = (steep_u_w / flat_u_w) / (steep_c_w / flat_c_w)

    rng_bs = np.random.RandomState(RNG_SEED)
    or_boots = []
    for _ in range(N_BOOTSTRAP):
        bfu = rng_bs.choice(len(iw_flat_u), len(iw_flat_u), replace=True)
        bfc = rng_bs.choice(len(iw_flat_c), len(iw_flat_c), replace=True)
        bsu = rng_bs.choice(len(iw_steep_u), len(iw_steep_u), replace=True)
        bsc = rng_bs.choice(len(iw_steep_c), len(iw_steep_c), replace=True)
        wfu = iw_flat_u[bfu].sum(); wfc = iw_flat_c[bfc].sum()
        wsu = iw_steep_u[bsu].sum(); wsc = iw_steep_c[bsc].sum()
        if wfu > 0 and wfc > 0 and wsc > 0:
            or_boots.append((wsu / wfu) / (wsc / wfc))

    ci_lo = np.percentile(or_boots, 2.5) if len(or_boots) > 100 else None
    ci_hi = np.percentile(or_boots, 97.5) if len(or_boots) > 100 else None

    return {
        'or_60plus': round(or_val, 3),
        'ci_lo': round(ci_lo, 3) if ci_lo is not None else None,
        'ci_hi': round(ci_hi, 3) if ci_hi is not None else None,
        'n_uap': n_uap_total,
        'n_ctrl': n_ctrl,
        'n_uap_steep': int(n_steep_uap),
        'n_ctrl_steep': int(n_steep_ctrl),
        'method': 'weighted_or',
    }


# ============================================================
# EVALUATE: shared function for E2b and E2c
# ============================================================
def evaluate_predictions(hot_preds, cold_preds, all_grid_cells, label):
    """
    Evaluate a set of hot/cold predictions against NUFORC.
    Returns full results dict.
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING {label}")
    print(f"{'='*70}")

    # Collect all cells to evaluate (hot + cold + rest with S>0 for Spearman)
    cells_to_eval = set()
    for p in hot_preds:
        cells_to_eval.add((p['lat'], p['lon']))
    for p in cold_preds:
        cells_to_eval.add((p['lat'], p['lon']))
    # Also include other cells with S>0 for Spearman correlation
    for c in all_grid_cells:
        if c['S'] > 0:
            cells_to_eval.add((c['lat'], c['lon']))

    print(f"  Cells to evaluate: {len(cells_to_eval)}")

    # Compute OR for all target cells
    all_cells_with_or = {}
    computed = 0
    for lat_c, lon_c in cells_to_eval:
        n_uap = cell_uap_counts.get((lat_c, lon_c), 0)
        if n_uap >= MIN_REPORTS_FOR_OR:
            result = compute_cell_or(lat_c, lon_c)
            computed += 1
            if computed % 50 == 0:
                print(f"    Computed {computed} cells... ({elapsed()})")
            if result is not None:
                all_cells_with_or[(lat_c, lon_c)] = result
        elif (lat_c, lon_c) in set((p['lat'], p['lon']) for p in hot_preds + cold_preds):
            all_cells_with_or[(lat_c, lon_c)] = {
                'or_60plus': None, 'ci_lo': None, 'ci_hi': None,
                'n_uap': n_uap, 'n_ctrl': 0,
                'n_uap_steep': 0, 'n_ctrl_steep': 0,
                'method': 'too_few_reports',
            }

    print(f"  Cells with OR computed: {len(all_cells_with_or)}")

    # --- HOT evaluation ---
    print(f"\n  HOT predictions (lower CI > 1.0 = HIT):")
    hot_results = []
    n_hit = n_miss = n_nodata = 0
    for i, pred in enumerate(hot_preds):
        lat, lon, S = pred['lat'], pred['lon'], pred['S']
        m = all_cells_with_or.get((lat, lon))
        if m and m['or_60plus'] is not None and m['ci_lo'] is not None:
            if m['ci_lo'] > 1.0:
                hit = "HIT"; n_hit += 1
            else:
                hit = "MISS"; n_miss += 1
            hot_results.append({
                'rank': i+1, 'lat': lat, 'lon': lon, 'S': S,
                'or': m['or_60plus'], 'ci_lo': m['ci_lo'], 'ci_hi': m['ci_hi'],
                'n_uap': m['n_uap'], 'verdict': hit,
            })
        else:
            n_nodata += 1
            n_uap = m['n_uap'] if m else 0
            hot_results.append({
                'rank': i+1, 'lat': lat, 'lon': lon, 'S': S,
                'or': None, 'ci_lo': None, 'ci_hi': None,
                'n_uap': n_uap, 'verdict': 'INSUFFICIENT_DATA',
            })

    for r in hot_results:
        or_str = f"{r['or']:.2f}" if r['or'] is not None else "N/A"
        ci_str = f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]" if r['ci_lo'] is not None else "[N/A]"
        print(f"    #{r['rank']:2d} ({r['lat']:.1f}, {r['lon']:.1f}) S={r['S']:.3f} | "
              f"OR={or_str:>7s} {ci_str:>15s} n={r['n_uap']:>5} | {r['verdict']}")

    hot_evaluable = n_hit + n_miss
    precision_hot = n_hit / hot_evaluable if hot_evaluable > 0 else 0
    print(f"\n  HOT: {n_hit} hits / {hot_evaluable} evaluable = Precision@{hot_evaluable} = {precision_hot:.2%}")
    print(f"  ({n_nodata} had insufficient data)")

    # --- COLD evaluation ---
    print(f"\n  COLD predictions (OR < 1.0 AND upper CI < 1.5 = HIT):")
    cold_results = []
    n_hit_c = n_miss_c = n_nodata_c = 0
    for i, pred in enumerate(cold_preds):
        lat, lon, S = pred['lat'], pred['lon'], pred['S']
        m = all_cells_with_or.get((lat, lon))
        if m and m['or_60plus'] is not None and m['ci_hi'] is not None:
            if m['or_60plus'] < 1.0 and m['ci_hi'] < 1.5:
                hit = "HIT"; n_hit_c += 1
            else:
                hit = "MISS"; n_miss_c += 1
            cold_results.append({
                'rank': i+1, 'lat': lat, 'lon': lon, 'S': S,
                'or': m['or_60plus'], 'ci_lo': m['ci_lo'], 'ci_hi': m['ci_hi'],
                'n_uap': m['n_uap'], 'verdict': hit,
            })
        else:
            n_nodata_c += 1
            n_uap = m['n_uap'] if m else 0
            cold_results.append({
                'rank': i+1, 'lat': lat, 'lon': lon, 'S': S,
                'or': None, 'ci_lo': None, 'ci_hi': None,
                'n_uap': n_uap, 'verdict': 'INSUFFICIENT_DATA',
            })

    for r in cold_results:
        or_str = f"{r['or']:.2f}" if r['or'] is not None else "N/A"
        ci_str = f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]" if r['ci_lo'] is not None else "[N/A]"
        print(f"    #{r['rank']:2d} ({r['lat']:.1f}, {r['lon']:.1f}) S={r['S']:.3f} | "
              f"OR={or_str:>7s} {ci_str:>15s} n={r['n_uap']:>5} | {r['verdict']}")

    cold_evaluable = n_hit_c + n_miss_c
    precision_cold = n_hit_c / cold_evaluable if cold_evaluable > 0 else 0
    print(f"\n  COLD: {n_hit_c} hits / {cold_evaluable} evaluable = Precision@{cold_evaluable} = {precision_cold:.2%}")
    print(f"  ({n_nodata_c} had insufficient data)")

    # --- Spearman ---
    valid_cells = [(k, v) for k, v in all_cells_with_or.items()
                   if v['or_60plus'] is not None
                   and v.get('method') not in ('too_few_reports', 'too_few_steep')]
    # Get S scores for these cells
    s_lookup = {(c['lat'], c['lon']): c['S'] for c in all_grid_cells}
    spearman_pairs = [(s_lookup.get(k, 0), v['or_60plus']) for k, v in valid_cells
                      if k in s_lookup]

    rho = p_val = None
    if len(spearman_pairs) > 10:
        s_arr = np.array([p[0] for p in spearman_pairs])
        or_arr = np.array([p[1] for p in spearman_pairs])
        rho, p_val = spearmanr(s_arr, or_arr)
        print(f"\n  Spearman(S, OR): rho = {rho:.3f}, p = {p_val:.4f} (n = {len(spearman_pairs)})")
    else:
        print(f"\n  Spearman: insufficient valid cells ({len(spearman_pairs)})")

    # --- AUC ---
    auc = None
    hot_set = set((p['lat'], p['lon']) for p in hot_preds)
    if len(spearman_pairs) > 20:
        labels = np.array([1 if k in hot_set else 0 for k, v in valid_cells if k in s_lookup])
        scores = np.array([v['or_60plus'] for k, v in valid_cells if k in s_lookup])
        if labels.sum() > 0 and labels.sum() < len(labels):
            auc = roc_auc_score(labels, scores)
            print(f"  AUC(hot vs rest): {auc:.3f}")
        else:
            print(f"  AUC: cannot compute (all same label)")
    else:
        print(f"  AUC: insufficient data")

    # --- Verdict ---
    verdict = "INCONCLUSIVE"
    if hot_evaluable >= 5 and cold_evaluable >= 5:
        if precision_hot >= 0.5 and rho is not None and rho > 0.3:
            verdict = "GEOMETRY_PREDICTS"
        elif precision_hot < 0.3 and (rho is None or rho < 0.1):
            verdict = "GEOMETRY_FAILS"
        else:
            verdict = "MIXED"

    print(f"\n  VERDICT: {verdict}")

    return {
        "label": label,
        "hot_evaluation": {
            "n_predictions": len(hot_preds),
            "n_evaluable": hot_evaluable,
            "n_hits": n_hit,
            "n_misses": n_miss,
            "n_insufficient_data": n_nodata,
            "precision": round(precision_hot, 4),
            "details": hot_results,
        },
        "cold_evaluation": {
            "n_predictions": len(cold_preds),
            "n_evaluable": cold_evaluable,
            "n_hits": n_hit_c,
            "n_misses": n_miss_c,
            "n_insufficient_data": n_nodata_c,
            "precision": round(precision_cold, 4),
            "details": cold_results,
        },
        "global_metrics": {
            "spearman_rho": round(rho, 4) if rho is not None else None,
            "spearman_p": round(p_val, 4) if p_val is not None else None,
            "spearman_n": len(spearman_pairs),
            "auc_hot_vs_rest": round(auc, 4) if auc is not None else None,
        },
        "verdict": verdict,
    }


# ============================================================
# RUN E2b
# ============================================================
results_e2b = evaluate_predictions(hot_e2b, cold_e2b, grid_conus, "E2b (CONUS bbox, no N filter)")


# ============================================================
# RUN E2c
# ============================================================
results_e2c = evaluate_predictions(hot_e2c, cold_e2c, grid_conus_measurable,
                                    f"E2c (CONUS + N>={MEASURABILITY_MIN_REPORTS})")


# ============================================================
# COMBINED SUMMARY
# ============================================================
print(f"\n{'='*70}")
print("COMBINED SUMMARY: E2b vs E2c")
print(f"{'='*70}")

for label, res in [("E2b", results_e2b), ("E2c", results_e2c)]:
    h = res['hot_evaluation']
    c = res['cold_evaluation']
    g = res['global_metrics']
    rho_s = f"{g['spearman_rho']:.3f}" if g['spearman_rho'] is not None else "N/A"
    p_s = f"{g['spearman_p']:.4f}" if g['spearman_p'] is not None else "N/A"
    auc_s = f"{g['auc_hot_vs_rest']:.3f}" if g['auc_hot_vs_rest'] is not None else "N/A"
    print(f"\n  {label}: {res['label']}")
    print(f"    HOT:  {h['n_hits']}/{h['n_evaluable']} = {h['precision']:.1%} "
          f"({h['n_insufficient_data']} no data)")
    print(f"    COLD: {c['n_hits']}/{c['n_evaluable']} = {c['precision']:.1%} "
          f"({c['n_insufficient_data']} no data)")
    print(f"    Spearman(S, OR): rho={rho_s}, p={p_s} (n={g['spearman_n']})")
    print(f"    AUC: {auc_s}")
    print(f"    Verdict: {res['verdict']}")


# ============================================================
# SAVE
# ============================================================
combined_results = {
    "metadata": {
        "script": "phase_e_evaluate_e2b.py",
        "timestamp": datetime.now().isoformat(),
        "frozen_commit_tag": "phase-e-frozen",
        "design_fix": "E2b_note.md — CONUS footprint mask",
        "conus_bbox": {
            "lat_min": CONUS_LAT_MIN, "lat_max": CONUS_LAT_MAX,
            "lon_min": CONUS_LON_MIN, "lon_max": CONUS_LON_MAX,
        },
        "coastal_band_km": COASTAL_BAND_KM,
        "measurability_min_reports": MEASURABILITY_MIN_REPORTS,
        "min_reports_for_or": MIN_REPORTS_FOR_OR,
        "scoring_unchanged": "phase-e-frozen commit 09de8d8",
    },
    "e2b": results_e2b,
    "e2c": results_e2c,
    "e2b_hot_locations": [
        {"rank": i+1, "lat": h['lat'], "lon": h['lon'], "S": round(h['S'], 4)}
        for i, h in enumerate(hot_e2b)
    ],
    "e2b_cold_locations": [
        {"rank": i+1, "lat": c['lat'], "lon": c['lon'], "S": round(c['S'], 4)}
        for i, c in enumerate(cold_e2b)
    ],
    "e2c_hot_locations": [
        {"rank": i+1, "lat": h['lat'], "lon": h['lon'], "S": round(h['S'], 4)}
        for i, h in enumerate(hot_e2c)
    ],
    "e2c_cold_locations": [
        {"rank": i+1, "lat": c['lat'], "lon": c['lon'], "S": round(c['S'], 4)}
        for i, c in enumerate(cold_e2c)
    ],
}

eval_file = os.path.join(OUT_DIR, "phase_e_e2b_e2c_evaluation.json")
with open(eval_file, 'w') as f:
    json.dump(combined_results, f, indent=2, default=str)
print(f"\n  Saved: {eval_file}")

# Also copy to results dir
import shutil
repo_out = os.path.join(BASE_DIR, "uap-canyon-analysis", "results", "phase_e")
os.makedirs(repo_out, exist_ok=True)
shutil.copy2(eval_file, os.path.join(repo_out, "phase_e_e2b_e2c_evaluation.json"))
print(f"  Copied to: {repo_out}")

print(f"\n{'='*70}")
print(f"DONE ({elapsed()})")
print(f"{'='*70}")
