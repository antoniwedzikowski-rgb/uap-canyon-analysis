#!/usr/bin/env python3
"""
Phase E v2 Evaluation: Test Frozen Geometric Predictions Against NUFORC
=========================================================================
Opens UAP data AFTER Phase E v2 scoring was committed and tagged
(phase-ev2-frozen, commit c2366d2).

Runs two variants:
  Ev2-b: CONUS bbox, no N filter
  Ev2-c: CONUS bbox + N >= 10 measurability mask

Fixes vs E1 evaluation:
  - too_few_steep returns None (NOT OR=0.0) → routes to INSUFFICIENT_DATA
  - Cold evaluation is flagged if tautological

Scoring function, ranks, and bin edges are UNCHANGED from freeze tag.
"""

import os, time, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR = "/Users/antoniwedzikowski/Desktop/UAP research"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "phase_ev2")
os.makedirs(OUT_DIR, exist_ok=True)

R_EARTH = 6371.0
COASTAL_BAND_KM = 200
GRID_DEG = 0.5
WINSORIZE_PCT = 95
N_BOOTSTRAP = 1000
RNG_SEED = 42
GRID_RES_FINE = 0.1
MIN_REPORTS_FOR_OR = 20
DEDUP_RADIUS_KM = 200
N_HOT = 20
N_COLD = 20
MEASURABILITY_MIN = 10

# OR bin edges — same as Phase C/D
BIN_EDGES = [0, 10, 30, 60, 500]

def elapsed():
    return f"{time.time()-t0:.1f}s"

def haversine_km_vec(lat1, lon1, lat2, lon2):
    r = np.radians
    dlat = r(np.asarray(lat2) - np.asarray(lat1))
    dlon = r(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dlat/2)**2 + np.cos(r(lat1))*np.cos(r(lat2))*np.sin(dlon/2)**2
    return R_EARTH * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ============================================================
# LOAD FROZEN v2 PREDICTIONS
# ============================================================
print("=" * 70)
print("PHASE E v2 EVALUATION")
print("Scoring frozen at phase-ev2-frozen (commit c2366d2)")
print("=" * 70)

grid_file = os.path.join(OUT_DIR, "phase_ev2_grid.json")
pred_file = os.path.join(OUT_DIR, "phase_ev2_predictions.json")

with open(grid_file) as f:
    grid_data = json.load(f)
with open(pred_file) as f:
    predictions = json.load(f)

hot_frozen = predictions['hot']
cold_frozen = predictions['cold']
print(f"  Grid: {len(grid_data)} cells")
print(f"  Hot predictions: {len(hot_frozen)}")
print(f"  Cold predictions: {len(cold_frozen)}")


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

_, c_idx = coast_tree.query(np.column_stack([df_raw['latitude'].values,
                                              df_raw['longitude'].values]), k=1)
d_coast = haversine_km_vec(df_raw['latitude'].values, df_raw['longitude'].values,
                            coast_lats[c_idx], coast_lons[c_idx])
df_coastal = df_raw[d_coast <= COASTAL_BAND_KM].copy().reset_index(drop=True)
print(f"  NUFORC coastal reports: {len(df_coastal):,}")


# ============================================================
# PRE-COMPUTE: UAP counts per grid cell
# ============================================================
print(f"\n[PREP] Counting UAP per grid cell... ({elapsed()})")
cell_uap_counts = {}
for cell in grid_data:
    lat_c, lon_c = cell['lat'], cell['lon']
    in_cell = ((df_coastal['latitude'].values >= lat_c - GRID_DEG/2) &
               (df_coastal['latitude'].values < lat_c + GRID_DEG/2) &
               (df_coastal['longitude'].values >= lon_c - GRID_DEG/2) &
               (df_coastal['longitude'].values < lon_c + GRID_DEG/2))
    cell_uap_counts[(lat_c, lon_c)] = int(in_cell.sum())

n_with_data = sum(1 for v in cell_uap_counts.values() if v > 0)
n_measurable = sum(1 for v in cell_uap_counts.values() if v >= MEASURABILITY_MIN)
n_or_ready = sum(1 for v in cell_uap_counts.values() if v >= MIN_REPORTS_FOR_OR)
print(f"  Cells with >0 reports: {n_with_data}")
print(f"  Cells with >={MEASURABILITY_MIN}: {n_measurable}")
print(f"  Cells with >={MIN_REPORTS_FOR_OR}: {n_or_ready}")


# ============================================================
# HELPER: per-cell gradient lookup and OR computation
# ============================================================
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
    """
    Compute weighted OR (60+ bin) for a 0.5° cell.
    Returns dict with results, or None if insufficient data.
    FIX: too_few_steep returns None (not OR=0.0).
    """
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

    # Generate controls
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

    # FIX: if too few in either bin, return None (not OR=0)
    if n_flat_uap < 5 or n_flat_ctrl < 5:
        return None
    if n_steep_uap < 3 or n_steep_ctrl < 3:
        return None

    # Importance weighting
    cd_u, ci_u = county_tree.query(np.column_stack([uap_lats, uap_lons]), k=5)
    uap_pop = np.zeros(len(uap_lats))
    for k in range(5):
        d_km = cd_u[:, k] * 111.0 + 1.0
        uap_pop += counties_pop[ci_u[:, k]] / (d_km**2)

    flat_u = uap_bins == 0; flat_c = ctrl_bins == 0
    steep_u = uap_bins == 3; steep_c = ctrl_bins == 3

    iw_flat_u = 1.0 / (uap_pop[flat_u] + 1e-10)
    iw_flat_u = np.minimum(iw_flat_u, np.percentile(iw_flat_u, WINSORIZE_PCT))

    iw_flat_c = 1.0 / (ctrl_w[flat_c] + 1e-10)
    iw_flat_c = np.minimum(iw_flat_c, np.percentile(iw_flat_c, WINSORIZE_PCT))

    iw_steep_u = 1.0 / (uap_pop[steep_u] + 1e-10)
    iw_steep_u = np.minimum(iw_steep_u, np.percentile(iw_steep_u, WINSORIZE_PCT))

    iw_steep_c = 1.0 / (ctrl_w[steep_c] + 1e-10)
    iw_steep_c = np.minimum(iw_steep_c, np.percentile(iw_steep_c, WINSORIZE_PCT))

    flat_u_w = iw_flat_u.sum()
    flat_c_w = iw_flat_c.sum()
    steep_u_w = iw_steep_u.sum()
    steep_c_w = iw_steep_c.sum()

    if flat_u_w == 0 or flat_c_w == 0 or steep_c_w == 0:
        return None

    or_val = (steep_u_w / flat_u_w) / (steep_c_w / flat_c_w)

    # Bootstrap CI
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
        'n_uap_flat': int(n_flat_uap),
        'n_uap_steep': int(n_steep_uap),
        'n_ctrl_flat': int(n_flat_ctrl),
        'n_ctrl_steep': int(n_steep_ctrl),
    }


# ============================================================
# EVALUATE: shared function
# ============================================================
def evaluate_variant(hot_preds, cold_preds, all_grid, label):
    print(f"\n{'='*70}")
    print(f"EVALUATING: {label}")
    print(f"{'='*70}")

    # Collect all cells needing OR
    cells_to_eval = set()
    for p in hot_preds + cold_preds:
        cells_to_eval.add((p['lat'], p['lon']))
    for c in all_grid:
        if c['S'] > 0:
            cells_to_eval.add((c['lat'], c['lon']))

    print(f"  Cells to evaluate: {len(cells_to_eval)}")

    # Compute ORs
    or_results = {}
    computed = 0
    for lat_c, lon_c in cells_to_eval:
        n_uap = cell_uap_counts.get((lat_c, lon_c), 0)
        if n_uap >= MIN_REPORTS_FOR_OR:
            result = compute_cell_or(lat_c, lon_c)
            computed += 1
            if computed % 50 == 0:
                print(f"    Computed {computed}... ({elapsed()})")
            if result is not None:
                or_results[(lat_c, lon_c)] = result

    print(f"  Cells with valid OR: {len(or_results)}")

    # --- HOT ---
    print(f"\n  HOT predictions (CI_lo > 1.0 = HIT):")
    hot_out = []
    nh, nm, nd = 0, 0, 0
    for pred in hot_preds:
        lat, lon, S = pred['lat'], pred['lon'], pred['S']
        n_uap = cell_uap_counts.get((lat, lon), 0)
        m = or_results.get((lat, lon))
        if m:
            if m['ci_lo'] is not None and m['ci_lo'] > 1.0:
                v = "HIT"; nh += 1
            else:
                v = "MISS"; nm += 1
            hot_out.append({'rank': pred['rank'], 'lat': lat, 'lon': lon, 'S': S,
                           'or': m['or_60plus'], 'ci_lo': m['ci_lo'], 'ci_hi': m['ci_hi'],
                           'n_uap': m['n_uap'], 'n_uap_steep': m['n_uap_steep'],
                           'n_uap_flat': m['n_uap_flat'], 'verdict': v})
        else:
            nd += 1
            hot_out.append({'rank': pred['rank'], 'lat': lat, 'lon': lon, 'S': S,
                           'or': None, 'ci_lo': None, 'ci_hi': None,
                           'n_uap': n_uap, 'n_uap_steep': 0, 'n_uap_flat': 0,
                           'verdict': 'INSUFFICIENT_DATA'})

    for r in hot_out:
        or_s = f"{r['or']:.2f}" if r['or'] is not None else "N/A"
        ci_s = f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]" if r['ci_lo'] is not None else "[N/A]"
        st_s = f"steep={r['n_uap_steep']}" if r['n_uap_steep'] else ""
        print(f"    #{r['rank']:2d} ({r['lat']:.1f}, {r['lon']:.1f}) S={r['S']:.3f} | "
              f"OR={or_s:>7s} {ci_s:>15s} n={r['n_uap']:>5} {st_s:>10s} | {r['verdict']}")

    hot_eval = nh + nm
    prec_hot = nh / hot_eval if hot_eval > 0 else 0
    print(f"\n  HOT: {nh}/{hot_eval} = {prec_hot:.1%} ({nd} insufficient)")

    # --- COLD ---
    print(f"\n  COLD predictions (OR < 1.0 AND CI_hi < 1.5 = HIT):")
    cold_out = []
    nhc, nmc, ndc = 0, 0, 0
    cold_tautological = 0
    for pred in cold_preds:
        lat, lon, S = pred['lat'], pred['lon'], pred['S']
        n_uap = cell_uap_counts.get((lat, lon), 0)
        m = or_results.get((lat, lon))
        if m:
            if m['or_60plus'] < 1.0 and m['ci_hi'] is not None and m['ci_hi'] < 1.5:
                v = "HIT"; nhc += 1
                # Check if tautological (zero steep in both UAP and ctrl)
                if m['n_uap_steep'] == 0:
                    cold_tautological += 1
            else:
                v = "MISS"; nmc += 1
            cold_out.append({'rank': pred['rank'], 'lat': lat, 'lon': lon, 'S': S,
                            'or': m['or_60plus'], 'ci_lo': m['ci_lo'], 'ci_hi': m['ci_hi'],
                            'n_uap': m['n_uap'], 'verdict': v})
        else:
            ndc += 1
            cold_out.append({'rank': pred['rank'], 'lat': lat, 'lon': lon, 'S': S,
                            'or': None, 'ci_lo': None, 'ci_hi': None,
                            'n_uap': n_uap, 'verdict': 'INSUFFICIENT_DATA'})

    for r in cold_out:
        or_s = f"{r['or']:.2f}" if r['or'] is not None else "N/A"
        ci_s = f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]" if r['ci_lo'] is not None else "[N/A]"
        print(f"    #{r['rank']:2d} ({r['lat']:.1f}, {r['lon']:.1f}) S={r['S']:.3f} | "
              f"OR={or_s:>7s} {ci_s:>15s} n={r['n_uap']:>5} | {r['verdict']}")

    cold_eval = nhc + nmc
    prec_cold = nhc / cold_eval if cold_eval > 0 else 0
    print(f"\n  COLD: {nhc}/{cold_eval} = {prec_cold:.1%} ({ndc} insufficient)")
    if cold_tautological > 0:
        print(f"  WARNING: {cold_tautological}/{nhc} cold HITs are tautological (zero 60+ pixels in cell)")

    # --- Spearman ---
    s_lookup = {(c['lat'], c['lon']): c['S'] for c in all_grid}
    pairs = [(s_lookup.get(k, 0), v['or_60plus'])
             for k, v in or_results.items() if k in s_lookup]

    rho = p_val = None
    if len(pairs) > 10:
        s_arr = np.array([p[0] for p in pairs])
        or_arr = np.array([p[1] for p in pairs])
        rho, p_val = spearmanr(s_arr, or_arr)
        print(f"\n  Spearman(S, OR): rho={rho:.3f}, p={p_val:.4f} (n={len(pairs)})")
    else:
        print(f"\n  Spearman: n={len(pairs)} too small")

    # --- AUC ---
    auc = None
    hot_set = set((p['lat'], p['lon']) for p in hot_preds)
    if len(pairs) > 20:
        labels = np.array([1 if k in hot_set else 0 for k, v in or_results.items() if k in s_lookup])
        scores = np.array([v['or_60plus'] for k, v in or_results.items() if k in s_lookup])
        if 0 < labels.sum() < len(labels):
            auc = roc_auc_score(labels, scores)
            print(f"  AUC(hot vs rest): {auc:.3f}")

    # --- Verdict ---
    verdict = "INCONCLUSIVE"
    if hot_eval >= 5 and cold_eval >= 5:
        if prec_hot >= 0.5 and rho is not None and rho > 0.3:
            verdict = "GEOMETRY_PREDICTS"
        elif prec_hot < 0.3 and (rho is None or rho < 0.1):
            verdict = "GEOMETRY_FAILS"
        else:
            verdict = "MIXED"
    print(f"\n  VERDICT: {verdict}")

    return {
        "label": label,
        "hot": {"n_pred": len(hot_preds), "n_eval": hot_eval,
                "n_hit": nh, "n_miss": nm, "n_insuff": nd,
                "precision": round(prec_hot, 4), "details": hot_out},
        "cold": {"n_pred": len(cold_preds), "n_eval": cold_eval,
                 "n_hit": nhc, "n_miss": nmc, "n_insuff": ndc,
                 "precision": round(prec_cold, 4),
                 "n_tautological": cold_tautological, "details": cold_out},
        "spearman": {"rho": round(rho, 4) if rho else None,
                     "p": round(p_val, 4) if p_val else None,
                     "n": len(pairs)},
        "auc": round(auc, 4) if auc else None,
        "verdict": verdict,
    }


# ============================================================
# Ev2-b: frozen predictions, no N filter
# ============================================================
res_b = evaluate_variant(hot_frozen, cold_frozen, grid_data, "Ev2-b (CONUS, no N filter)")


# ============================================================
# Ev2-c: measurability mask (N >= 10), re-select from grid
# ============================================================
print(f"\n{'='*70}")
print(f"RE-SELECTING FOR Ev2-c (N >= {MEASURABILITY_MIN})")
print(f"{'='*70}")

grid_measurable = [c for c in grid_data
                   if cell_uap_counts.get((c['lat'], c['lon']), 0) >= MEASURABILITY_MIN]
print(f"  Measurable cells: {len(grid_measurable)}")

def greedy_dedup(candidates, n_select, descending=True):
    sorted_cands = sorted(candidates, key=lambda x: x['S'], reverse=descending)
    selected = []
    for cand in sorted_cands:
        if len(selected) >= n_select:
            break
        too_close = any(haversine_km_vec(cand['lat'], cand['lon'], s['lat'], s['lon']) < DEDUP_RADIUS_KM
                        for s in selected)
        if not too_close:
            selected.append(cand)
    return selected

hot_c = [c for c in grid_measurable if c['S'] > 0]
hot_c_sel = greedy_dedup(hot_c, N_HOT, descending=True)
hot_c_preds = [{'rank': i+1, 'lat': h['lat'], 'lon': h['lon'], 'S': h['S']}
               for i, h in enumerate(hot_c_sel)]

cold_c = [c for c in grid_measurable if c['S'] == 0]
for c in cold_c:
    _, ci = coast_tree.query([c['lat'], c['lon']], k=1)
    c['_d_coast'] = float(haversine_km_vec(c['lat'], c['lon'],
                                            float(coast_lats[ci]), float(coast_lons[ci])))
cold_c.sort(key=lambda x: x['_d_coast'])
cold_c_sel = greedy_dedup(cold_c, N_COLD, descending=False)
cold_c_preds = [{'rank': i+1, 'lat': c['lat'], 'lon': c['lon'], 'S': c['S']}
                for i, c in enumerate(cold_c_sel)]

print(f"  Ev2-c hot candidates: {len(hot_c)} → selected {len(hot_c_preds)}")
print(f"  Ev2-c cold candidates: {len(cold_c)} → selected {len(cold_c_preds)}")

for i, h in enumerate(hot_c_preds):
    n = cell_uap_counts.get((h['lat'], h['lon']), 0)
    print(f"    HOT #{i+1:2d}: ({h['lat']:.2f}, {h['lon']:.2f}) S={h['S']:.3f} N={n}")

res_c = evaluate_variant(hot_c_preds, cold_c_preds, grid_measurable,
                          f"Ev2-c (CONUS + N>={MEASURABILITY_MIN})")


# ============================================================
# COMBINED SUMMARY
# ============================================================
print(f"\n{'='*70}")
print("COMBINED SUMMARY: Ev2-b vs Ev2-c")
print(f"{'='*70}")

for tag, res in [("Ev2-b", res_b), ("Ev2-c", res_c)]:
    h, c, sp = res['hot'], res['cold'], res['spearman']
    rho_s = f"{sp['rho']:.3f}" if sp['rho'] is not None else "N/A"
    print(f"\n  {tag}: {res['label']}")
    print(f"    HOT:  {h['n_hit']}/{h['n_eval']} = {h['precision']:.1%} ({h['n_insuff']} insuff)")
    print(f"    COLD: {c['n_hit']}/{c['n_eval']} = {c['precision']:.1%} ({c['n_insuff']} insuff, "
          f"{c['n_tautological']} tautological)")
    print(f"    Spearman: rho={rho_s} (n={sp['n']})")
    print(f"    AUC: {res['auc']}")
    print(f"    Verdict: {res['verdict']}")


# ============================================================
# SAVE
# ============================================================
combined = {
    "metadata": {
        "script": "phase_ev2_evaluate.py",
        "timestamp": datetime.now().isoformat(),
        "frozen_tag": "phase-ev2-frozen",
        "frozen_commit": "c2366d2",
        "threshold_mkm": 60,
        "label": "refined hypothesis test (threshold from Phase C/D)",
    },
    "ev2_b": res_b,
    "ev2_c": res_c,
}

out_file = os.path.join(OUT_DIR, "phase_ev2_evaluation.json")
with open(out_file, 'w') as f:
    json.dump(combined, f, indent=2, default=str)
print(f"\n  Saved: {out_file}")

import shutil
repo_out = os.path.join(BASE_DIR, "uap-canyon-analysis", "results", "phase_ev2")
os.makedirs(repo_out, exist_ok=True)
shutil.copy2(out_file, repo_out)
print(f"  Copied to repo: {repo_out}")

print(f"\n{'='*70}")
print(f"DONE ({elapsed()})")
print(f"{'='*70}")
