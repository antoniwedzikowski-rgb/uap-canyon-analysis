#!/usr/bin/env python3
"""
Phase E v2: Geometric Scoring — Re-Specification
==================================================
FROZEN BEFORE EVALUATION. Commit + tag required before running eval.

Re-specification rationale (NOT post-hoc tuning):
  Phase C/D established that the UAP-gradient association is concentrated
  in the 60+ m/km bin (OR ≈ 5.09 CONUS-wide). Bins 0-10, 10-30, 30-60
  showed weak or no effect. Phase E v1 used a 20 m/km threshold inherited
  from canyon *detection* (any steep bathymetry), not from the *estimand*
  (where the UAP effect lives). This mismatch meant v1 scored many cells
  with gradients 20-50 m/km as "hot" — locations where Phase C/D found
  no UAP effect.

  Phase E v2 aligns the scoring threshold with the Phase C/D estimand:
  only cells with gradient > 60 m/km are "steep". Everything else
  (formula, ranking, grid, dedup) is identical to v1.

  Label: "refined hypothesis test" — the 60 m/km threshold was defined
  by prior analysis on the same NUFORC dataset, so this is NOT an
  independent out-of-sample preregistration. It IS a hypothesis test
  with the threshold specified before seeing per-location results.

Score: S = mean(rank_G + rank_P + rank_C) for 60+ m/km cells within 50km.
Rankings are GLOBAL across all CONUS shelf cells meeting the 60+ threshold.

NO NUFORC DATA IS LOADED OR USED IN THIS SCRIPT.
"""

import os, time, json, warnings
import numpy as np
from datetime import datetime
from collections import defaultdict
from scipy.spatial import cKDTree
from scipy.ndimage import label as ndimage_label
import netCDF4 as nc

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR = os.environ.get("UAP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "phase_ev2")
os.makedirs(OUT_DIR, exist_ok=True)

R_EARTH = 6371.0

# ============================================================
# FROZEN PARAMETERS
# ============================================================
# >>> KEY CHANGE vs v1: threshold aligned with Phase C/D estimand <<<
CANYON_GRADIENT_THRESHOLD = 60.0   # m/km — Phase C/D 60+ bin (was 20 in v1)
SHELF_DEPTH_MIN = -500             # meters
SHELF_DEPTH_MAX = 0                # meters
MIN_COMPONENT_SIZE = 3             # cells
COASTAL_BAND_KM = 200              # km
GRADIENT_RADIUS_KM = 25            # km
PROXIMITY_EFOLD_KM = 50.0          # km — e-folding scale for shore proximity
COAST_COUNT_RADIUS_KM = 25         # km — radius for coastal complexity
AGGREGATION_RADIUS_KM = 50         # km — radius for aggregating steep cells to grid
GRID_DEG = 0.5                     # degrees — output grid resolution
DEDUP_RADIUS_KM = 200              # km — minimum spacing between selected locations
N_HOT = 20                         # top predictions
N_COLD = 20                        # bottom predictions

# CONUS footprint mask (lesson from E2a: exclude non-NUFORC areas)
CONUS_LAT_MIN = 24.5
CONUS_LAT_MAX = 49.0
CONUS_LON_MIN = -125.0
CONUS_LON_MAX = -66.0

def elapsed():
    return f"{time.time()-t0:.1f}s"

def haversine_km_vec(lat1, lon1, lat2, lon2):
    r = np.radians
    dlat = r(np.asarray(lat2) - np.asarray(lat1))
    dlon = r(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dlat/2)**2 + np.cos(r(lat1))*np.cos(r(lat2))*np.sin(dlon/2)**2
    return R_EARTH * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ============================================================
# LOAD GEOMETRY (ETOPO only — no UAP data)
# ============================================================
print("=" * 70)
print("PHASE E v2: GEOMETRIC SCORING (RE-SPECIFICATION)")
print("Threshold: 60 m/km (aligned with Phase C/D estimand)")
print("NO UAP DATA USED IN THIS SCRIPT")
print("=" * 70)

print(f"\n[LOAD] ETOPO... ({elapsed()})")
ds = nc.Dataset(os.path.join(DATA_DIR, "etopo_subset.nc"))
if 'y' in ds.variables:
    elev_lats = ds.variables['y'][:]
    elev_lons = ds.variables['x'][:]
else:
    elev_lats = ds.variables['lat'][:]
    elev_lons = ds.variables['lon'][:]
elevation = ds.variables['z'][:]
ds.close()
print(f"  Grid: {elevation.shape[0]} x {elevation.shape[1]}")

# --- Coast detection ---
print(f"[LOAD] Coast detection... ({elapsed()})")
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
print(f"  Coast cells: {len(coast_lats):,}")

# --- Shelf + gradient ---
print(f"[LOAD] Gradient computation... ({elapsed()})")
shelf_mask = (elevation < 0) & (elevation > SHELF_DEPTH_MIN)
lat_res = abs(float(elev_lats[1] - elev_lats[0]))
lon_res = abs(float(elev_lons[1] - elev_lons[0]))
grad_y = np.abs(np.gradient(elevation, axis=0)) / (lat_res * 111.0)
grad_x = np.abs(np.gradient(elevation, axis=1)) / (lon_res * 111.0 *
         np.cos(np.radians(elev_lats[:, None])))
gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

# --- Canyon (steep) cells — NOW AT 60 m/km ---
print(f"[LOAD] Canyon detection (threshold={CANYON_GRADIENT_THRESHOLD} m/km)... ({elapsed()})")
steep = (gradient_mag > CANYON_GRADIENT_THRESHOLD) & shelf_mask
labeled, n_feat = ndimage_label(steep)
for c_id in range(1, n_feat + 1):
    if np.sum(labeled == c_id) < MIN_COMPONENT_SIZE:
        steep[labeled == c_id] = False
labeled, n_feat = ndimage_label(steep)
canyon_i, canyon_j = np.where(steep)
canyon_lats = elev_lats[canyon_i]
canyon_lons = elev_lons[canyon_j]
print(f"  Canyon cells (>{CANYON_GRADIENT_THRESHOLD} m/km): {len(canyon_lats):,}, components: {n_feat}")

# --- Shelf cells for gradient aggregation ---
shelf_i, shelf_j = np.where(shelf_mask)
shelf_lats = elev_lats[shelf_i]
shelf_lons = elev_lons[shelf_j]
shelf_gradients = gradient_mag[shelf_i, shelf_j]
print(f"  Shelf cells: {len(shelf_lats):,}")

print(f"\n[GEOMETRY LOADED] ({elapsed()})")


# ============================================================
# STEP 1: Per-cell feature computation for all steep cells
# ============================================================
print("\n" + "=" * 70)
print(f"STEP 1: PER-CELL FEATURES (cells with gradient > {CANYON_GRADIENT_THRESHOLD} m/km)")
print("=" * 70)

n_steep = len(canyon_lats)
if n_steep == 0:
    print("  ERROR: Zero steep cells found. Cannot proceed.")
    exit(1)

print(f"  Computing features for {n_steep:,} steep cells...")

# --- G: p95 gradient within 25km ---
GRID_RES_FINE = 0.1
gradient_by_cell = defaultdict(list)
for idx in range(len(shelf_lats)):
    lat_bin = round(shelf_lats[idx] / GRID_RES_FINE) * GRID_RES_FINE
    lon_bin = round(shelf_lons[idx] / GRID_RES_FINE) * GRID_RES_FINE
    gradient_by_cell[(round(lat_bin, 1), round(lon_bin, 1))].append(
        float(shelf_gradients[idx]))

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

print(f"  Computing G (p95 gradient)... ({elapsed()})")
G_values = np.array([get_p95_gradient(lat, lon)
                      for lat, lon in zip(canyon_lats, canyon_lons)])

# --- P: proximity to shore ---
print(f"  Computing P (shore proximity)... ({elapsed()})")
_, coast_idx = coast_tree.query(np.column_stack([canyon_lats, canyon_lons]), k=1)
d_shore_km = haversine_km_vec(canyon_lats, canyon_lons,
                               coast_lats[coast_idx], coast_lons[coast_idx])
P_values = np.exp(-d_shore_km / PROXIMITY_EFOLD_KM)

# --- C: coastal complexity (coast cells within 25km) ---
print(f"  Computing C (coastal complexity)... ({elapsed()})")
approx_deg_25km = COAST_COUNT_RADIUS_KM / 111.0
C_values = np.zeros(n_steep)
for idx in range(n_steep):
    nearby = coast_tree.query_ball_point(
        [canyon_lats[idx], canyon_lons[idx]], approx_deg_25km)
    C_values[idx] = len(nearby)

print(f"  G range: [{G_values.min():.1f}, {G_values.max():.1f}] m/km")
print(f"  P range: [{P_values.min():.4f}, {P_values.max():.4f}]")
print(f"  C range: [{C_values.min():.0f}, {C_values.max():.0f}]")


# ============================================================
# STEP 2: Global ranking
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: GLOBAL RANKING (across all 60+ m/km steep cells)")
print("=" * 70)

def global_rank(values):
    """Return fractional rank in [0, 1]."""
    from scipy.stats import rankdata
    return (rankdata(values) - 1) / (len(values) - 1)

rank_G = global_rank(G_values)
rank_P = global_rank(P_values)
rank_C = global_rank(C_values)

cell_score = rank_G + rank_P + rank_C  # range [0, 3]

print(f"  Cell scores: min={cell_score.min():.3f}, max={cell_score.max():.3f}, "
      f"mean={cell_score.mean():.3f}")


# ============================================================
# STEP 3: Aggregate to 0.5° grid (CONUS only)
# ============================================================
print("\n" + "=" * 70)
print(f"STEP 3: AGGREGATE TO {GRID_DEG}° GRID (CONUS bbox)")
print("=" * 70)

# CONUS bbox grid
grid_lat_edges = np.arange(CONUS_LAT_MIN, CONUS_LAT_MAX + 0.01, GRID_DEG)
grid_lon_edges = np.arange(CONUS_LON_MIN, CONUS_LON_MAX + 0.01, GRID_DEG)

grid_results = []
approx_deg_50km = AGGREGATION_RADIUS_KM / 111.0

steep_tree = cKDTree(np.column_stack([canyon_lats, canyon_lons]))

print(f"  Scanning {len(grid_lat_edges) * len(grid_lon_edges)} CONUS grid cells...")
for lat_c in grid_lat_edges + GRID_DEG / 2:
    for lon_c in grid_lon_edges + GRID_DEG / 2:
        # Skip if not coastal
        _, ci = coast_tree.query([lat_c, lon_c], k=1)
        d_coast = haversine_km_vec(lat_c, lon_c,
                                    float(coast_lats[ci]), float(coast_lons[ci]))
        if d_coast > COASTAL_BAND_KM:
            continue

        # Find steep (60+) cells within aggregation radius
        nearby_idx = steep_tree.query_ball_point([lat_c, lon_c], approx_deg_50km)
        if len(nearby_idx) == 0:
            grid_results.append({
                'lat': float(lat_c),
                'lon': float(lon_c),
                'S': 0.0,
                'n_steep_cells': 0,
                'mean_G': 0.0,
                'mean_P': 0.0,
                'mean_C': 0.0,
            })
            continue

        nearby_scores = cell_score[nearby_idx]
        nearby_G = G_values[nearby_idx]
        nearby_P = P_values[nearby_idx]
        nearby_C = C_values[nearby_idx]

        grid_results.append({
            'lat': float(lat_c),
            'lon': float(lon_c),
            'S': float(np.mean(nearby_scores)),
            'n_steep_cells': len(nearby_idx),
            'mean_G': float(np.mean(nearby_G)),
            'mean_P': float(np.mean(nearby_P)),
            'mean_C': float(np.mean(nearby_C)),
        })

print(f"  CONUS coastal grid cells: {len(grid_results)}")
nonzero = sum(1 for r in grid_results if r['S'] > 0)
print(f"  With 60+ steep cells (S > 0): {nonzero}")
print(f"  Without 60+ steep cells (S = 0): {len(grid_results) - nonzero}")


# ============================================================
# STEP 4: Spatial dedup + select top/bottom
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: SPATIAL DEDUPLICATION + SELECTION")
print("=" * 70)

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

# Hot: highest S, deduped
hot_candidates = [r for r in grid_results if r['S'] > 0]
hot_selected = greedy_dedup(hot_candidates, N_HOT, descending=True)

# Cold: S == 0, prefer closer to coast
cold_candidates = [r for r in grid_results if r['S'] == 0]
for c in cold_candidates:
    _, ci = coast_tree.query([c['lat'], c['lon']], k=1)
    c['_d_coast'] = float(haversine_km_vec(c['lat'], c['lon'],
                                            float(coast_lats[ci]), float(coast_lons[ci])))
cold_candidates.sort(key=lambda x: x['_d_coast'])
cold_selected = greedy_dedup(cold_candidates, N_COLD, descending=False)

print(f"\n  HOT locations selected: {len(hot_selected)}")
for i, h in enumerate(hot_selected):
    print(f"    #{i+1:2d}: ({h['lat']:.2f}, {h['lon']:.2f}) S={h['S']:.3f} "
          f"[{h['n_steep_cells']} steep cells, G={h['mean_G']:.1f}, "
          f"P={h['mean_P']:.3f}, C={h['mean_C']:.0f}]")

print(f"\n  COLD locations selected: {len(cold_selected)}")
for i, c in enumerate(cold_selected):
    print(f"    #{i+1:2d}: ({c['lat']:.2f}, {c['lon']:.2f}) S={c['S']:.3f} "
          f"[coast_dist={c['_d_coast']:.1f}km]")


# ============================================================
# SAVE PREDICTIONS
# ============================================================
print("\n" + "=" * 70)
print("SAVING PREDICTIONS")
print("=" * 70)

predictions = {
    "metadata": {
        "script": "phase_ev2_scoring.py",
        "timestamp": datetime.now().isoformat(),
        "description": "Phase E v2 (re-specification): geometric predictions with 60 m/km threshold aligned to Phase C/D estimand. NO UAP data used.",
        "label": "refined hypothesis test — threshold from prior Phase C/D analysis on same NUFORC dataset",
        "vs_v1": "v1 used 20 m/km (canyon detection threshold); v2 uses 60 m/km (Phase C/D effect threshold). All other parameters identical.",
        "parameters": {
            "canyon_gradient_threshold_mkm": CANYON_GRADIENT_THRESHOLD,
            "shelf_depth_range_m": [SHELF_DEPTH_MIN, SHELF_DEPTH_MAX],
            "min_component_size": MIN_COMPONENT_SIZE,
            "coastal_band_km": COASTAL_BAND_KM,
            "gradient_radius_km": GRADIENT_RADIUS_KM,
            "proximity_efold_km": PROXIMITY_EFOLD_KM,
            "coast_count_radius_km": COAST_COUNT_RADIUS_KM,
            "aggregation_radius_km": AGGREGATION_RADIUS_KM,
            "grid_deg": GRID_DEG,
            "dedup_radius_km": DEDUP_RADIUS_KM,
            "conus_bbox": {
                "lat_min": CONUS_LAT_MIN, "lat_max": CONUS_LAT_MAX,
                "lon_min": CONUS_LON_MIN, "lon_max": CONUS_LON_MAX,
            },
            "ranking": "global across all CONUS shelf cells with gradient > 60 m/km",
        },
        "n_steep_cells_total": int(n_steep),
        "n_canyon_components": int(n_feat),
        "n_coastal_grid_cells": len(grid_results),
        "n_with_steep": nonzero,
    },
    "hot": [
        {"rank": i+1, "lat": h['lat'], "lon": h['lon'], "S": round(h['S'], 4)}
        for i, h in enumerate(hot_selected)
    ],
    "cold": [
        {"rank": i+1, "lat": h['lat'], "lon": h['lon'], "S": round(h['S'], 4)}
        for i, h in enumerate(cold_selected)
    ],
}

pred_file = os.path.join(OUT_DIR, "phase_ev2_predictions.json")
with open(pred_file, 'w') as f:
    json.dump(predictions, f, indent=2)
print(f"  Saved: {pred_file}")

grid_file = os.path.join(OUT_DIR, "phase_ev2_grid.json")
grid_clean = []
for r in grid_results:
    grid_clean.append({
        'lat': r['lat'],
        'lon': r['lon'],
        'S': round(r['S'], 4),
        'n_steep_cells': r['n_steep_cells'],
    })
with open(grid_file, 'w') as f:
    json.dump(grid_clean, f, indent=2)
print(f"  Saved: {grid_file}")

print(f"\n{'='*70}")
print(f"DONE ({elapsed()})")
print(f"NO UAP DATA WAS USED.")
print(f"{'='*70}")
