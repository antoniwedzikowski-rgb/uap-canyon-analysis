#!/usr/bin/env python3
"""
Phase E: Shoreline Type Proxy Test (Option 3 confound)
=======================================================
Question: Is the UAP excess in canyon cells driven by ABOVE-water coastal
topography (rocky cliffs, viewpoints) rather than BELOW-water canyons?

Proxy: compute land-side elevation gradient within each 0.5° cell,
near the coastline (within 5km of shore, on land). Cells with steep
coastal terrain (cliffs) get high "cliff score". Then test:

  logR ~ S + cliff_score + S×cliff_score

If cliff_score explains the variance and S becomes NS → Option 3 (cultural confound).
If S survives after controlling for cliff_score → not explained by viewpoints.

Also: simple correlation between S (underwater) and cliff_score (above water).
If r ≈ 1, they're the same thing and we can't distinguish them.
If r is moderate, we can partial out one from the other.

Uses ETOPO elevation — land gradient near coast as proxy for rocky/cliffy shore.
"""

import os, time, json, warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, pearsonr
from numpy.linalg import lstsq

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR = "/Users/antoniwedzikowski/Desktop/UAP research"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "phase_ev2")

R_EARTH = 6371.0
GRID_DEG = 0.5

def elapsed():
    return f"{time.time()-t0:.1f}s"

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.asarray, [lat1, lon1, lat2, lon2])
    r = np.radians
    dlat = r(lat2 - lat1)
    dlon = r(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(r(lat1))*np.cos(r(lat2))*np.sin(dlon/2)**2
    return R_EARTH * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ============================================================
# LOAD
# ============================================================
print("=" * 70)
print("PHASE E: SHORELINE TYPE PROXY TEST")
print("Above-water coastal gradient vs below-water canyon score")
print("=" * 70)

# Load ETOPO
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

elev_res_lat = abs(elev_lats[1] - elev_lats[0])
elev_res_lon = abs(elev_lons[1] - elev_lons[0])
print(f"  ETOPO resolution: {elev_res_lat:.4f}° × {elev_res_lon:.4f}°")
print(f"  Shape: {elevation.shape}")

# Load E-RED v2 cell details
with open(os.path.join(OUT_DIR, "phase_e_red_v2_evaluation.json")) as f:
    data = json.load(f)
cells = data["primary_200km"]["cell_details"]
print(f"  Loaded {len(cells)} testable West Coast cells")


# ============================================================
# COMPUTE CLIFF SCORE PER CELL
# ============================================================
print(f"\n[COMPUTE] Coastal land gradient per cell... ({elapsed()})")

def compute_cliff_score(cell_lat, cell_lon, half_deg):
    """
    Compute mean land-side elevation gradient within cell, near coast.

    Method:
    1. Find all ETOPO pixels in cell that are on LAND (elevation >= 0)
    2. Among those, find pixels within ~5km of an ocean pixel
    3. Compute gradient (max neighbor difference) for each coastal land pixel
    4. Return mean gradient (m per ETOPO pixel spacing)

    High cliff_score = steep terrain near water = rocky/cliffy coast
    Low cliff_score = flat terrain near water = sandy/flat coast
    """
    # Find ETOPO indices for this cell
    lat_idx_lo = np.searchsorted(elev_lats, cell_lat - half_deg)
    lat_idx_hi = np.searchsorted(elev_lats, cell_lat + half_deg)
    lon_idx_lo = np.searchsorted(elev_lons, cell_lon - half_deg)
    lon_idx_hi = np.searchsorted(elev_lons, cell_lon + half_deg)

    # Expand by 1 pixel for gradient computation
    lat_lo = max(0, lat_idx_lo - 1)
    lat_hi = min(len(elev_lats) - 1, lat_idx_hi + 1)
    lon_lo = max(0, lon_idx_lo - 1)
    lon_hi = min(len(elev_lons) - 1, lon_idx_hi + 1)

    sub_elev = elevation[lat_lo:lat_hi+1, lon_lo:lon_hi+1]

    if sub_elev.size == 0:
        return 0.0, 0

    # Land mask and ocean mask
    land = sub_elev >= 0
    ocean = sub_elev < 0

    if not land.any() or not ocean.any():
        return 0.0, 0

    # Find coastal land pixels (land pixels adjacent to ocean)
    coastal_land = np.zeros_like(land, dtype=bool)
    nr, nc_cols = sub_elev.shape
    for i in range(1, nr-1):
        for j in range(1, nc_cols-1):
            if land[i, j]:
                # Check if any neighbor is ocean
                if ocean[i-1:i+2, j-1:j+2].any():
                    coastal_land[i, j] = True

    # Also check pixels 1 step inland from coastal (within ~5km = ~3 pixels at ~1.5km/pixel)
    for step in range(2):
        expanded = np.zeros_like(coastal_land, dtype=bool)
        for i in range(1, nr-1):
            for j in range(1, nc_cols-1):
                if land[i, j] and not coastal_land[i, j]:
                    if coastal_land[i-1:i+2, j-1:j+2].any():
                        expanded[i, j] = True
        coastal_land |= expanded

    if not coastal_land.any():
        return 0.0, 0

    # Compute gradient for coastal land pixels
    gradients = []
    ci, cj = np.where(coastal_land)
    for i, j in zip(ci, cj):
        if i < 1 or i >= nr-1 or j < 1 or j >= nc_cols-1:
            continue
        center = sub_elev[i, j]
        neighbors = sub_elev[i-1:i+2, j-1:j+2].flatten()
        # Only compare with land neighbors
        land_neighbors = neighbors[sub_elev[i-1:i+2, j-1:j+2].flatten() >= 0]
        if len(land_neighbors) > 1:
            max_diff = np.max(np.abs(land_neighbors - center))
            gradients.append(max_diff)

    if not gradients:
        return 0.0, 0

    return float(np.mean(gradients)), len(gradients)


half = GRID_DEG / 2
for cell in cells:
    cliff, n_pix = compute_cliff_score(cell['lat'], cell['lon'], half)
    cell['cliff_score'] = cliff
    cell['n_coastal_land_pix'] = n_pix

# ============================================================
# ANALYSIS
# ============================================================
S_arr = np.array([c['S'] for c in cells])
logR_arr = np.array([c['logR'] for c in cells])
R_arr = np.array([c['R_i'] for c in cells])
cliff_arr = np.array([c['cliff_score'] for c in cells])
lat_arr = np.array([c['lat'] for c in cells])
n = len(cells)

print(f"\n  Cliff score range: [{cliff_arr.min():.1f}, {cliff_arr.max():.1f}] m/pixel")
print(f"  Cliff score mean: {cliff_arr.mean():.1f}, median: {np.median(cliff_arr):.1f}")


# ============================================================
# TEST 1: Correlation between S and cliff_score
# ============================================================
print(f"\n{'='*70}")
print("TEST 1: S (underwater) vs cliff_score (above water)")
print(f"{'='*70}")

rho_sc, p_sc = spearmanr(S_arr, cliff_arr)
r_sc, p_sc_p = pearsonr(S_arr, cliff_arr)
print(f"  Spearman(S, cliff): rho = {rho_sc:.3f}, p = {p_sc:.4f}")
print(f"  Pearson(S, cliff):  r = {r_sc:.3f}, p = {p_sc_p:.4f}")

if abs(rho_sc) > 0.8:
    print(f"  ⚠️  HIGH CORRELATION — cannot separate underwater from above-water effect")
elif abs(rho_sc) > 0.5:
    print(f"  ⚠️  MODERATE CORRELATION — partial separation possible but collinearity concern")
else:
    print(f"  ✓  LOW-MODERATE CORRELATION — can meaningfully test both predictors")


# ============================================================
# TEST 2: Does cliff_score predict logR?
# ============================================================
print(f"\n{'='*70}")
print("TEST 2: cliff_score → logR")
print(f"{'='*70}")

rho_clR, p_clR = spearmanr(cliff_arr, logR_arr)
print(f"  Spearman(cliff, logR): rho = {rho_clR:.3f}, p = {p_clR:.4f}")


# ============================================================
# TEST 3: Competing models
# ============================================================
print(f"\n{'='*70}")
print("TEST 3: COMPETING MODELS")
print(f"{'='*70}")

# Model A: logR ~ S only
X_a = np.column_stack([np.ones(n), S_arr])
b_a, _, _, _ = lstsq(X_a, logR_arr, rcond=None)
ss_a = np.sum((logR_arr - X_a @ b_a)**2)
r2_a = 1 - ss_a / np.sum((logR_arr - logR_arr.mean())**2)

# Model B: logR ~ cliff only
X_b = np.column_stack([np.ones(n), cliff_arr])
b_b, _, _, _ = lstsq(X_b, logR_arr, rcond=None)
ss_b = np.sum((logR_arr - X_b @ b_b)**2)
r2_b = 1 - ss_b / np.sum((logR_arr - logR_arr.mean())**2)

# Model C: logR ~ S + cliff
X_c = np.column_stack([np.ones(n), S_arr, cliff_arr])
b_c, _, _, _ = lstsq(X_c, logR_arr, rcond=None)
ss_c = np.sum((logR_arr - X_c @ b_c)**2)
r2_c = 1 - ss_c / np.sum((logR_arr - logR_arr.mean())**2)

print(f"\n  Model A (logR ~ S):           β_S = {b_a[1]:+.4f}, R² = {r2_a:.4f}")
print(f"  Model B (logR ~ cliff):       β_cliff = {b_b[1]:+.6f}, R² = {r2_b:.4f}")
print(f"  Model C (logR ~ S + cliff):   β_S = {b_c[1]:+.4f}, β_cliff = {b_c[2]:+.6f}, R² = {r2_c:.4f}")

# F-test: does adding S to cliff improve fit?
from scipy.stats import f as f_dist
df_full = n - 3
F_s_given_cliff = ((ss_b - ss_c) / 1) / (ss_c / df_full)
p_s_given_cliff = 1 - f_dist.cdf(F_s_given_cliff, 1, df_full)

# F-test: does adding cliff to S improve fit?
F_cliff_given_s = ((ss_a - ss_c) / 1) / (ss_c / df_full)
p_cliff_given_s = 1 - f_dist.cdf(F_cliff_given_s, 1, df_full)

print(f"\n  F-test: S improves over cliff-only?     F = {F_s_given_cliff:.3f}, p = {p_s_given_cliff:.4f}")
print(f"  F-test: cliff improves over S-only?     F = {F_cliff_given_s:.3f}, p = {p_cliff_given_s:.4f}")

# Bootstrap CIs for Model C
N_BOOT = 5000
rng = np.random.RandomState(42)
boot_c = []
for _ in range(N_BOOT):
    idx = rng.choice(n, n, replace=True)
    try:
        bb, _, _, _ = lstsq(X_c[idx], logR_arr[idx], rcond=None)
        boot_c.append(bb)
    except:
        pass
boot_c = np.array(boot_c)

print(f"\n  Model C bootstrap CIs:")
for i, name in enumerate(['α', 'β_S', 'β_cliff']):
    lo = np.percentile(boot_c[:, i], 2.5)
    hi = np.percentile(boot_c[:, i], 97.5)
    print(f"    {name:10s}: {b_c[i]:+.6f}  [{lo:+.6f}, {hi:+.6f}]")


# ============================================================
# TEST 4: Within Puget — cliff vs S
# ============================================================
print(f"\n{'='*70}")
print("TEST 4: WITHIN PUGET — cliff vs S")
print(f"{'='*70}")

puget = (lat_arr > 46) & (lat_arr < 50) & (S_arr > 0)
n_p = puget.sum()

if n_p >= 5:
    rho_ps, _ = spearmanr(S_arr[puget], cliff_arr[puget])
    rho_pR, _ = spearmanr(cliff_arr[puget], logR_arr[puget])
    print(f"  n = {n_p}")
    print(f"  Spearman(S, cliff) within Puget: {rho_ps:.3f}")
    print(f"  Spearman(cliff, logR) within Puget: {rho_pR:.3f}")
    print(f"  Spearman(S, logR) within Puget: {spearmanr(S_arr[puget], logR_arr[puget])[0]:.3f}")

    # Show values
    print(f"\n  {'lat':>6} {'lon':>7} {'S':>6} {'cliff':>7} {'logR':>7} {'R':>7}")
    for c in sorted([cells[i] for i in range(n) if puget[i]], key=lambda x: -x['S']):
        print(f"  {c['lat']:6.1f} {c['lon']:7.1f} {c['S']:6.3f} {c['cliff_score']:7.1f} "
              f"{c['logR']:7.3f} {c['R_i']:7.2f}")


# ============================================================
# TEST 5: Puget S=0 — do cliff scores differ?
# ============================================================
print(f"\n{'='*70}")
print("TEST 5: Puget S=0 vs S>0 — cliff score comparison")
print(f"{'='*70}")

puget_all = (lat_arr >= 46) & (lat_arr <= 50)
p_s0 = puget_all & (S_arr == 0)
p_s1 = puget_all & (S_arr > 0)

if p_s0.sum() > 0 and p_s1.sum() > 0:
    print(f"  Puget S=0 (n={p_s0.sum()}): mean cliff = {cliff_arr[p_s0].mean():.1f}")
    print(f"  Puget S>0 (n={p_s1.sum()}): mean cliff = {cliff_arr[p_s1].mean():.1f}")

    from scipy.stats import mannwhitneyu
    U, p_mw = mannwhitneyu(cliff_arr[p_s1], cliff_arr[p_s0], alternative='greater')
    print(f"  Mann-Whitney (S>0 cliff > S=0 cliff): U={U:.0f}, p={p_mw:.4f}")

    if p_mw < 0.05:
        print(f"  ⚠️  Canyon cells have significantly steeper LAND topography")
        print(f"     → confound is plausible: cliffs and canyons co-occur")
    else:
        print(f"  ✓  No significant difference in land topography")
        print(f"     → cliff confound less likely")


# ============================================================
# SAVE
# ============================================================
results = {
    "test": "Shoreline type proxy (cliff score from ETOPO land gradient)",
    "S_cliff_correlation": {
        "spearman_rho": round(float(rho_sc), 4),
        "spearman_p": round(float(p_sc), 4),
        "pearson_r": round(float(r_sc), 4),
    },
    "cliff_logR_spearman": round(float(rho_clR), 4),
    "models": {
        "A_S_only": {"beta_S": round(float(b_a[1]), 4), "R2": round(r2_a, 4)},
        "B_cliff_only": {"beta_cliff": round(float(b_b[1]), 6), "R2": round(r2_b, 4)},
        "C_both": {
            "beta_S": round(float(b_c[1]), 4),
            "beta_cliff": round(float(b_c[2]), 6),
            "R2": round(r2_c, 4),
        },
    },
    "F_tests": {
        "S_given_cliff": {"F": round(F_s_given_cliff, 3), "p": round(p_s_given_cliff, 4)},
        "cliff_given_S": {"F": round(F_cliff_given_s, 3), "p": round(p_cliff_given_s, 4)},
    },
}

out_file = os.path.join(OUT_DIR, "phase_e_shoretype_proxy.json")
with open(out_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_file}")

import shutil
repo_out = os.path.join(BASE_DIR, "uap-canyon-analysis", "results", "phase_ev2")
shutil.copy2(out_file, repo_out)
print(f"Copied to repo")

print(f"\nDONE ({elapsed()})")
