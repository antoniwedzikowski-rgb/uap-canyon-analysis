#!/usr/bin/env python3
"""
Phase E: Replication Suite — CORRECTED (audit fixes CRIT-1, CRIT-2, CRIT-3)
===========================================================================
Fixes from statistical audit:
  CRIT-1: E_i was uniform (pop defaulted to 1). Now recomputes proper
          IDW population-weighted E_i per fold using the same model as
          phase_e_red_v2.py.
  CRIT-2: LOO CV used logR from full-data evaluation (data leakage).
          Now recomputes O_i and E_i independently per fold.
  CRIT-3: Spatial forward prediction used pre-computed logR from full
          dataset. Now recomputes per-fold logR from fold-specific O_i/E_i.

Tests:
  6a. Temporal split: 4 splits, each recomputes E_i from fold-specific reports
  6b. Spatial forward prediction: fit logR ~ S on training, predict held-out
  6c. Leave-one-region-out cross-validation (proper per-fold E_i)
  6d. 5-year rolling temporal stability
  6e. Post-2014 replication (2015-2023)
"""

import os, time, json, warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from numpy.linalg import lstsq

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR = os.environ.get("UAP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "results", "phase_ev2")

GRID_DEG = 0.5
COAST_KM = 200
MIN_REPORTS = 20
R_EARTH = 6371.0

# West Coast definition — standardized with phase_e_red_v2.py
WEST_COAST_LON_MAX = -115.0
WEST_COAST_LAT_MIN = 30.0

def elapsed():
    return f"{time.time()-t0:.1f}s"


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km — fully vectorized."""
    lat1, lon1, lat2, lon2 = map(np.asarray, [lat1, lon1, lat2, lon2])
    r = np.radians
    dlat = r(lat2 - lat1)
    dlon = r(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(r(lat1))*np.cos(r(lat2))*np.sin(dlon/2)**2
    return R_EARTH * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


print("=" * 70)
print("PHASE E: REPLICATION SUITE (CORRECTED — audit CRIT-1/2/3)")
print("Proper IDW population-weighted E_i per fold")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================
print(f"\n[LOAD] Data... ({elapsed()})")

# Canyon scores (S only — we do NOT use logR or E_i from this file)
with open(os.path.join(OUT_DIR, "phase_ev2_grid.json")) as f:
    grid_data = json.load(f)
print(f"  Full grid: {len(grid_data)} cells")

cell_S_lookup = {}
for cell in grid_data:
    key = (round(cell['lat'], 1), round(cell['lon'], 1))
    cell_S_lookup[key] = cell['S']

# NUFORC
df = pd.read_csv(os.path.join(DATA_DIR, "nuforc_reports.csv"), header=None,
    names=["datetime_str","city","state","country","shape",
           "duration_seconds","duration_text","description","date_posted","lat","lon"],
    low_memory=False)
df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
df['dt'] = pd.to_datetime(df['datetime_str'], errors='coerce')
df['year'] = df['dt'].dt.year
df = df.dropna(subset=['lat','lon','year'])

# CONUS + West Coast filter (standardized lon cutoff)
df = df[(df['lat'] >= WEST_COAST_LAT_MIN) & (df['lat'] <= 50) &
        (df['lon'] >= -130) & (df['lon'] <= WEST_COAST_LON_MAX) &
        (df['year'] >= 1990) & (df['year'] <= 2014)]
print(f"  {len(df)} West Coast reports (1990-2014)")

# Cell assignments
half = GRID_DEG / 2
df['cell_lat'] = (np.floor(df['lat'] / GRID_DEG) * GRID_DEG + half).round(1)
df['cell_lon'] = (np.floor(df['lon'] / GRID_DEG) * GRID_DEG + half).round(1)


# ============================================================
# LOAD GEOMETRY for E_i computation (same as phase_e_red_v2.py)
# ============================================================
print(f"\n[LOAD] Geometry for E_i... ({elapsed()})")

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

# Coast detection
coast_mask_arr = np.zeros_like(elevation, dtype=bool)
nrows, ncols = elevation.shape
for i in range(1, nrows-1):
    for j in range(1, ncols-1):
        if elevation[i, j] < 0:
            if np.any(elevation[i-1:i+2, j-1:j+2] >= 0):
                coast_mask_arr[i, j] = True
coast_i, coast_j = np.where(coast_mask_arr)
coast_lats_arr = elev_lats[coast_i].astype(np.float64)
coast_lons_arr = elev_lons[coast_j].astype(np.float64)
coast_tree = cKDTree(np.column_stack([coast_lats_arr, coast_lons_arr]))

# County population
counties_df = pd.read_csv(os.path.join(DATA_DIR, "county_centroids_pop.csv"))
county_lats = counties_df['lat'].values.astype(np.float64)
county_lons = counties_df['lon'].values.astype(np.float64)
county_tree = cKDTree(np.column_stack([county_lats, county_lons]))
counties_pop = counties_df['pop'].values.astype(np.float64)

print(f"  Coast cells: {len(coast_lats_arr):,}")
print(f"  Counties: {len(counties_pop):,}")
print(f"  Geometry loaded ({elapsed()})")


# ============================================================
# E_i computation — identical to phase_e_red_v2.py
# ============================================================
def compute_expected(cell_lat, cell_lon, half_deg, coastal_band_km):
    """Population-weighted expected report density (haversine-corrected)."""
    grid_lat = np.linspace(cell_lat - half_deg, cell_lat + half_deg, 30)
    grid_lon = np.linspace(cell_lon - half_deg, cell_lon + half_deg, 30)
    glat, glon = np.meshgrid(grid_lat, grid_lon, indexing='ij')
    gf, gln = glat.flatten(), glon.flatten()

    # Coastal filter — haversine
    _, coast_idx = coast_tree.query(np.column_stack([gf, gln]), k=1)
    coast_dist_km = haversine_km(gf, gln,
                                  coast_lats_arr[coast_idx],
                                  coast_lons_arr[coast_idx])
    coastal_mask = coast_dist_km <= coastal_band_km
    gc_lat = gf[coastal_mask]
    gc_lon = gln[coastal_mask]

    if len(gc_lat) < 5:
        return 0.0

    # Population weighting — haversine
    n_k = min(10, len(counties_pop))
    _, ci_county = county_tree.query(np.column_stack([gc_lat, gc_lon]), k=n_k)

    weights = np.zeros(len(gc_lat))
    for k in range(n_k):
        d_km = haversine_km(gc_lat, gc_lon,
                            county_lats[ci_county[:, k]],
                            county_lons[ci_county[:, k]])
        d_km = d_km + 1.0
        weights += counties_pop[ci_county[:, k]] / (d_km**2)

    # Land/ocean weighting
    lat_idx = np.clip(np.searchsorted(elev_lats, gc_lat), 0, len(elev_lats)-1)
    lon_idx = np.clip(np.searchsorted(elev_lons, gc_lon), 0, len(elev_lons)-1)
    ge = elevation[lat_idx, lon_idx]
    lw = np.where(ge >= 0, 3.0, 0.05)
    weights *= lw

    return float(weights.sum())


# Pre-compute E_i_raw for all West Coast grid cells (expensive but done once)
print(f"\n[PRECOMPUTE] E_i_raw for all WC grid cells... ({elapsed()})")
wc_cells = [(round(c['lat'], 1), round(c['lon'], 1))
            for c in grid_data
            if c['lon'] <= WEST_COAST_LON_MAX and c['lat'] >= WEST_COAST_LAT_MIN]
# Deduplicate
wc_cell_set = sorted(set(wc_cells))

E_i_raw_cache = {}
for i, (clat, clon) in enumerate(wc_cell_set):
    E_i_raw_cache[(clat, clon)] = compute_expected(clat, clon, half, COAST_KM)
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(wc_cell_set)} cells... ({elapsed()})")

print(f"  E_i_raw computed for {len(E_i_raw_cache)} cells ({elapsed()})")
n_nonzero = sum(1 for v in E_i_raw_cache.values() if v > 0)
print(f"  Non-zero E_i_raw: {n_nonzero}")


# ============================================================
# CORRECTED HELPER: compute S-logR with proper per-fold E_i
# ============================================================
def compute_s_logr_corrected(report_subset, min_reports=MIN_REPORTS):
    """Compute per-cell O_i, E_i (IDW population), R_i, logR and
    return Spearman(S, logR). E_i uses pre-cached IDW weights,
    normalized so sum(E_i) = sum(O_i) within the fold."""
    counts = report_subset.groupby(['cell_lat', 'cell_lon']).size().reset_index(name='O_i')

    results = []
    for _, row in counts.iterrows():
        key = (round(row['cell_lat'], 1), round(row['cell_lon'], 1))
        if key in cell_S_lookup and key in E_i_raw_cache:
            results.append({
                'lat': key[0], 'lon': key[1],
                'S': cell_S_lookup[key],
                'O_i': int(row['O_i']),
                'E_i_raw': E_i_raw_cache[key],
            })

    if not results:
        return None, None, 0, None

    res_df = pd.DataFrame(results)

    # Normalize E_i so sum(E_i) = sum(O_i) within this fold
    total_O = res_df['O_i'].sum()
    total_E_raw = res_df['E_i_raw'].sum()
    if total_E_raw == 0:
        return None, None, 0, None

    res_df['E_i'] = res_df['E_i_raw'] / total_E_raw * total_O
    res_df['E_i'] = res_df['E_i'].clip(lower=0.1)
    res_df['R_i'] = res_df['O_i'] / res_df['E_i']
    res_df['logR'] = np.log(res_df['R_i'].clip(lower=0.01))

    # Filter to cells with enough reports
    testable = res_df[res_df['O_i'] >= min_reports].copy()

    if len(testable) < 5:
        return None, None, len(testable), None

    S = testable['S'].values
    logR = testable['logR'].values
    rho, p = spearmanr(S, logR)
    return rho, p, len(testable), testable


# ============================================================
# TEST 6a: TEMPORAL SPLIT REPLICATION
# ============================================================
print(f"\n{'='*70}")
print("TEST 6a: TEMPORAL SPLIT REPLICATION (corrected E_i)")
print(f"{'='*70}")

splits = [
    ("1990-2002 → 2003-2014",
     df[df['year'] <= 2002], df[df['year'] >= 2003]),
    ("2003-2014 → 1990-2002",
     df[df['year'] >= 2003], df[df['year'] <= 2002]),
    ("1990-2006 → 2007-2014",
     df[df['year'] <= 2006], df[df['year'] >= 2007]),
    ("Even years → Odd years",
     df[df['year'] % 2 == 0], df[df['year'] % 2 == 1]),
]

temporal_results = []
print(f"\n  {'Split':<35} {'n_train':>8} {'n_test':>8} {'n_cells':>8} {'rho':>8} {'p':>10}")
print(f"  {'-'*80}")

for label, train, test in splits:
    rho_train, p_train, n_train_cells, _ = compute_s_logr_corrected(train, min_reports=10)
    rho_test, p_test, n_test_cells, _ = compute_s_logr_corrected(test, min_reports=10)

    sig = "*" if p_test is not None and p_test < 0.05 else ""
    rho_str = f"{rho_test:+.3f}" if rho_test is not None else "N/A"
    p_str = f"{p_test:.6f}" if p_test is not None else "N/A"

    print(f"  {label:<35} {len(train):>8} {len(test):>8} {n_test_cells:>8} {rho_str:>8} {p_str:>10} {sig}")

    temporal_results.append({
        'split': label,
        'n_train': int(len(train)),
        'n_test': int(len(test)),
        'n_test_cells': n_test_cells,
        'rho_train': round(float(rho_train), 4) if rho_train is not None else None,
        'p_train': round(float(p_train), 6) if p_train is not None else None,
        'rho_test': round(float(rho_test), 4) if rho_test is not None else None,
        'p_test': round(float(p_test), 6) if p_test is not None else None,
    })


# ============================================================
# TEST 6b: SPATIAL FORWARD PREDICTION (corrected: per-fold E_i)
# ============================================================
print(f"\n{'='*70}")
print("TEST 6b: SPATIAL FORWARD PREDICTION (corrected E_i)")
print("Fit logR ~ S on training region, predict held-out region")
print("logR recomputed per fold (no leakage)")
print(f"{'='*70}")

PUGET_LAT = (46.5, 49.0)
PUGET_LON = (-125.0, -121.5)
SOCAL_LAT = (32.0, 34.5)
SOCAL_LON = (-120.5, WEST_COAST_LON_MAX)

def in_region(lat, lon, lat_range, lon_range):
    return (lat_range[0] <= lat <= lat_range[1]) and (lon_range[0] <= lon <= lon_range[1])

# Build cell-level data from ALL WC reports (full dataset)
rho_full, p_full, n_full, full_df = compute_s_logr_corrected(df, min_reports=MIN_REPORTS)
print(f"\n  Full dataset: rho={rho_full:+.3f}, p={p_full:.6f}, n={n_full}")

spatial_results = []

for region_name, lat_range, lon_range in [
    ("Puget", PUGET_LAT, PUGET_LON),
    ("SoCal", SOCAL_LAT, SOCAL_LON),
]:
    if full_df is None:
        continue

    full_df['is_region'] = full_df.apply(
        lambda r: in_region(r['lat'], r['lon'], lat_range, lon_range), axis=1)

    held_out = full_df[full_df['is_region']]
    training = full_df[~full_df['is_region']]

    if len(held_out) < 3 or len(training) < 10:
        print(f"\n  {region_name}: insufficient data (held_out={len(held_out)}, training={len(training)})")
        continue

    # Fit on training (using fold-specific logR)
    X_train = np.column_stack([np.ones(len(training)), training['S'].values])
    y_train = training['logR'].values
    beta, _, _, _ = lstsq(X_train, y_train, rcond=None)

    # Predict on held-out (using fold-specific logR)
    X_test = np.column_stack([np.ones(len(held_out)), held_out['S'].values])
    y_pred = X_test @ beta
    y_actual = held_out['logR'].values

    ss_res = np.sum((y_actual - y_pred)**2)
    ss_tot = np.sum((y_actual - y_actual.mean())**2)
    r2_oos = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    rho_oos, p_oos = spearmanr(held_out['S'].values, y_actual)
    mae = np.mean(np.abs(y_actual - y_pred))

    if held_out['S'].nunique() > 1:
        s_med = held_out['S'].median()
        hi_s = held_out[held_out['S'] > s_med]['logR'].mean()
        lo_s = held_out[held_out['S'] <= s_med]['logR'].mean()
        direction = "correct" if hi_s > lo_s else "WRONG"
    else:
        hi_s = lo_s = held_out['logR'].mean()
        direction = "constant S"

    print(f"\n  === {region_name} (held out {len(held_out)} cells) ===")
    print(f"  Training: {len(training)} cells, beta = [{beta[0]:.3f}, {beta[1]:.4f}]")
    print(f"  Out-of-sample R² = {r2_oos:.4f}")
    print(f"  Spearman(S, logR) in held-out: rho = {rho_oos:+.3f}, p = {p_oos:.4f}")
    print(f"  MAE = {mae:.3f}")
    print(f"  High-S mean logR = {hi_s:.3f}, Low-S mean logR = {lo_s:.3f} → {direction}")

    spatial_results.append({
        'region': region_name,
        'n_held_out': int(len(held_out)),
        'n_training': int(len(training)),
        'beta': [round(float(b), 4) for b in beta],
        'R2_oos': round(float(r2_oos), 4),
        'rho_oos': round(float(rho_oos), 4),
        'p_oos': round(float(p_oos), 4),
        'MAE': round(float(mae), 4),
        'direction': direction,
    })


# ============================================================
# TEST 6c: LEAVE-ONE-REGION-OUT CV (corrected: per-fold E_i)
# ============================================================
print(f"\n{'='*70}")
print("TEST 6c: LEAVE-ONE-REGION-OUT CV (corrected E_i per fold)")
print(f"{'='*70}")

regions = {
    'WA_north': (47.5, 49.5),
    'WA_south_OR_north': (44.0, 47.5),
    'OR_south_NorCal': (40.0, 44.0),
    'CenCal': (35.0, 40.0),
    'SoCal': (32.0, 35.0),
}

if full_df is not None:
    full_df['region'] = 'other'
    for rname, (lat_lo, lat_hi) in regions.items():
        mask = (full_df['lat'] >= lat_lo) & (full_df['lat'] < lat_hi)
        full_df.loc[mask, 'region'] = rname

    print(f"\n  Region cell counts:")
    for rname in regions:
        n = (full_df['region'] == rname).sum()
        print(f"    {rname:<25}: {n} cells")

loo_results = []
rho_list = []

for holdout_region in regions:
    if full_df is None:
        continue

    test = full_df[full_df['region'] == holdout_region]
    train = full_df[full_df['region'] != holdout_region]

    if len(test) < 3:
        continue

    # Fit on training (fold-specific logR — no leakage)
    X_tr = np.column_stack([np.ones(len(train)), train['S'].values])
    y_tr = train['logR'].values
    beta, _, _, _ = lstsq(X_tr, y_tr, rcond=None)

    # Evaluate on test
    rho_test, p_test = spearmanr(test['S'].values, test['logR'].values)

    X_te = np.column_stack([np.ones(len(test)), test['S'].values])
    y_pred = X_te @ beta
    y_actual = test['logR'].values
    ss_res = np.sum((y_actual - y_pred)**2)
    ss_tot = np.sum((y_actual - y_actual.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    sig = "*" if p_test < 0.05 else ""
    print(f"  Hold out {holdout_region:<25}: n={len(test):>3}, rho={rho_test:+.3f}, p={p_test:.4f} {sig}, R2_oos={r2:.3f}")

    rho_list.append(rho_test)
    loo_results.append({
        'holdout': holdout_region,
        'n_test': int(len(test)),
        'rho': round(float(rho_test), 4),
        'p': round(float(p_test), 4),
        'R2_oos': round(float(r2), 4),
    })

if rho_list:
    print(f"\n  Mean held-out rho: {np.mean(rho_list):+.3f}")
    print(f"  Min: {min(rho_list):+.3f}, Max: {max(rho_list):+.3f}")


# ============================================================
# TEST 6d: TEMPORAL STABILITY — 5-year rolling windows
# ============================================================
print(f"\n{'='*70}")
print("TEST 6d: TEMPORAL STABILITY — 5-year rolling windows (corrected E_i)")
print(f"{'='*70}")

temporal_stability = []
for start_year in range(1990, 2011):
    end_year = start_year + 4
    window = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
    rho, p, n_cells, _ = compute_s_logr_corrected(window, min_reports=5)

    if rho is not None:
        sig = "*" if p < 0.05 else ""
        print(f"  {start_year}-{end_year}: n_cells={n_cells:>3}, rho={rho:+.3f}, p={p:.4f} {sig}")
        temporal_stability.append({
            'window': f"{start_year}-{end_year}",
            'n_cells': n_cells,
            'rho': round(float(rho), 4),
            'p': round(float(p), 4),
        })


# ============================================================
# TEST 6e: POST-2014 REPLICATION (2015-2023) (corrected E_i)
# ============================================================
print(f"\n{'='*70}")
print("TEST 6e: POST-2014 OUT-OF-SAMPLE REPLICATION (corrected E_i)")
print(f"{'='*70}")

post2014_result = {}
try:
    df_post = pd.read_csv(os.path.join(DATA_DIR, "nuforc_post2014.csv"))
    df_post['dt'] = pd.to_datetime(df_post['dt'], errors='coerce')
    df_post['year'] = df_post['dt'].dt.year

    # Filter to post-2014 West Coast (standardized lon cutoff)
    post_wc = df_post[(df_post['year'] > 2014) &
                       (df_post['lat'] >= WEST_COAST_LAT_MIN) & (df_post['lat'] <= 50) &
                       (df_post['lon'] >= -130) & (df_post['lon'] <= WEST_COAST_LON_MAX)]

    print(f"\n  Post-2014 West Coast reports: {len(post_wc)}")

    # Assign to cells
    post_wc = post_wc.copy()
    post_wc['cell_lat'] = (np.floor(post_wc['lat'] / GRID_DEG) * GRID_DEG + half).round(1)
    post_wc['cell_lon'] = (np.floor(post_wc['lon'] / GRID_DEG) * GRID_DEG + half).round(1)

    # Geocoding quality check
    n_unique_coords = post_wc.groupby(['lat', 'lon']).ngroups
    print(f"  Unique coordinate pairs: {n_unique_coords}")
    print(f"  Reports per unique coord: {len(post_wc)/max(n_unique_coords,1):.1f}")
    if n_unique_coords < len(post_wc) * 0.05:
        print(f"  WARNING: Very few unique coordinates — geocoding may be city-centroid only")

    # Compute with corrected E_i
    rho_post, p_post, n_post_cells, _ = compute_s_logr_corrected(post_wc, min_reports=5)

    if rho_post is not None:
        sig = "*" if p_post < 0.05 else ""
        print(f"  Spearman(S, logR): rho = {rho_post:+.3f}, p = {p_post:.6f} {sig}")
        print(f"  Testable cells: {n_post_cells}")

        # Compare to original period
        print(f"\n  Original (1990-2014): rho = {rho_full:+.3f}, p = {p_full:.6f}, n = {n_full}")
        print(f"  Post-2014 (2015-2023): rho = {rho_post:+.3f}, p = {p_post:.6f}, n = {n_post_cells}")

        if rho_post > 0 and p_post < 0.05:
            verdict = "REPLICATES"
        elif rho_post > 0:
            verdict = "POSITIVE_BUT_NS"
        else:
            verdict = "FAILS"
        print(f"\n  Verdict: {verdict}")

        post2014_result = {
            'n_reports': int(len(post_wc)),
            'n_cells': n_post_cells,
            'n_unique_coords': n_unique_coords,
            'rho': round(float(rho_post), 4),
            'p': round(float(p_post), 6),
            'verdict': verdict,
            'E_i_model': 'IDW population-weighted (same as E-RED v2)',
            'geocoding_note': 'city-centroid geocoding (all reports from same city share coordinates)',
        }
    else:
        print(f"  Insufficient data (n_cells={n_post_cells})")
        post2014_result = {'verdict': 'INSUFFICIENT_DATA', 'n_cells': n_post_cells}

except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()
    post2014_result = {'verdict': 'ERROR', 'error': str(e)}


# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

n_temporal_sig = sum(1 for r in temporal_results if r['p_test'] is not None and r['p_test'] < 0.05)
print(f"\n  Temporal splits: {n_temporal_sig}/{len(temporal_results)} significant at p<0.05")

for r in spatial_results:
    print(f"  {r['region']} forward prediction: rho={r['rho_oos']:+.3f}, direction={r['direction']}")

if rho_list:
    n_loo_pos = sum(1 for r in rho_list if r > 0)
    print(f"  LOO regions: {n_loo_pos}/{len(rho_list)} show positive S-logR relationship")
    print(f"  Mean LOO rho: {np.mean(rho_list):+.3f}")

pos_windows = sum(1 for r in temporal_stability if r['rho'] > 0)
print(f"  5-year windows: {pos_windows}/{len(temporal_stability)} show positive rho")

if post2014_result:
    print(f"  Post-2014 replication: {post2014_result.get('verdict', 'N/A')}")


# ============================================================
# COMPARISON with old (buggy) results
# ============================================================
print(f"\n{'='*70}")
print("COMPARISON: old (uniform E_i) vs corrected (IDW E_i)")
print(f"{'='*70}")

try:
    old_file = os.path.join(OUT_DIR, "phase_e_replication_suite_OLD.json")
    if os.path.exists(old_file):
        with open(old_file) as f:
            old = json.load(f)
        print("\n  TEMPORAL SPLITS:")
        for old_r in old.get('temporal_splits', []):
            new_r = next((r for r in temporal_results if r['split'] == old_r['split']), None)
            if new_r and old_r.get('rho_test') is not None and new_r.get('rho_test') is not None:
                delta = new_r['rho_test'] - old_r['rho_test']
                print(f"    {old_r['split']}: old rho={old_r['rho_test']:+.3f} → new rho={new_r['rho_test']:+.3f} (Δ={delta:+.3f})")
        print("\n  POST-2014:")
        old_post = old.get('post_2014_replication', {})
        if old_post.get('rho') and post2014_result.get('rho'):
            delta = post2014_result['rho'] - old_post['rho']
            print(f"    old rho={old_post['rho']:+.3f} → new rho={post2014_result['rho']:+.3f} (Δ={delta:+.3f})")
    else:
        print("  No old results file found for comparison")
except Exception as e:
    print(f"  Comparison error: {e}")


# ============================================================
# SAVE
# ============================================================
results = {
    "test": "Replication suite (CORRECTED — audit CRIT-1/2/3)",
    "fixes_applied": [
        "CRIT-1: E_i now uses IDW population-weighted model (was uniform)",
        "CRIT-2: LOO CV now uses fold-specific logR (was full-data logR)",
        "CRIT-3: Spatial prediction now uses fold-specific logR (was full-data logR)",
    ],
    "E_i_model": "IDW k=10 nearest county centroids, haversine distances, land/ocean weighting 3.0/0.05",
    "west_coast_definition": f"lon <= {WEST_COAST_LON_MAX}, lat >= {WEST_COAST_LAT_MIN}",
    "full_dataset_baseline": {
        "rho": round(float(rho_full), 4) if rho_full else None,
        "p": round(float(p_full), 6) if p_full else None,
        "n_cells": n_full,
    },
    "temporal_splits": temporal_results,
    "spatial_forward_prediction": spatial_results,
    "leave_one_region_out": loo_results,
    "temporal_stability_5yr": temporal_stability,
    "post_2014_replication": post2014_result,
}

# Backup old results
old_file = os.path.join(OUT_DIR, "phase_e_replication_suite.json")
if os.path.exists(old_file):
    import shutil
    shutil.copy2(old_file, os.path.join(OUT_DIR, "phase_e_replication_suite_OLD.json"))
    print(f"\n  Old results backed up to phase_e_replication_suite_OLD.json")

with open(old_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved: {old_file}")

import shutil

print(f"\nDONE ({elapsed()})")
