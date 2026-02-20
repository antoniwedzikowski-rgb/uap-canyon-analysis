#!/usr/bin/env python3
"""
Phase E: Oceanographic Confound Test (Option 2)
================================================
Tests whether the UAP-canyon correlation is better explained by
oceanographic variables (depth, temperature proxy) than by canyon
gradient steepness (S).

Hypothesis: If canyons → upwelling → fog/optical phenomena → UAP reports,
then:
  1. Depth alone should predict logR as well as S
  2. Coastal SST anomaly (cold water = upwelling) should predict logR
  3. S should lose significance when controlling for depth/SST

We use available data:
  - Mean ocean depth per cell (from our ETOPO subset)
  - Depth variance per cell (proxy for topographic complexity)
  - Min depth per cell (deepest nearby water = upwelling potential)
  - Distance to deepest water per cell

Compare: logR ~ S  vs  logR ~ depth_metrics  vs  logR ~ S + depth_metrics
"""

import os, time, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import f as f_dist
from numpy.linalg import lstsq
import netCDF4 as nc

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR = os.environ.get("UAP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "phase_ev2")

GRID_DEG = 0.5
R_EARTH = 6371.0

def elapsed():
    return f"{time.time()-t0:.1f}s"

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.asarray, [lat1, lon1, lat2, lon2])
    r = np.radians
    dlat = r(lat2 - lat1); dlon = r(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(r(lat1))*np.cos(r(lat2))*np.sin(dlon/2)**2
    return R_EARTH * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


print("=" * 70)
print("PHASE E: OCEANOGRAPHIC CONFOUND TEST")
print("Depth/upwelling proxy vs canyon steepness (S)")
print("=" * 70)


# ============================================================
# LOAD
# ============================================================
print(f"\n[LOAD] Data... ({elapsed()})")

# ETOPO
ds = nc.Dataset(os.path.join(DATA_DIR, "etopo_subset.nc"))
if 'y' in ds.variables:
    elev_lats = ds.variables['y'][:]
    elev_lons = ds.variables['x'][:]
else:
    elev_lats = ds.variables['lat'][:]
    elev_lons = ds.variables['lon'][:]
elevation = ds.variables['z'][:]
ds.close()

# E-RED v2 cells
with open(os.path.join(OUT_DIR, "phase_e_red_v2_evaluation.json")) as f:
    data = json.load(f)
cells = data["primary_200km"]["cell_details"]
print(f"  {len(cells)} testable West Coast cells")


# ============================================================
# COMPUTE OCEAN DEPTH METRICS PER CELL
# ============================================================
print(f"\n[COMPUTE] Ocean depth metrics per cell... ({elapsed()})")

half = GRID_DEG / 2

for cell in cells:
    clat, clon = cell['lat'], cell['lon']

    lat_lo = np.searchsorted(elev_lats, clat - half)
    lat_hi = np.searchsorted(elev_lats, clat + half)
    lon_lo = np.searchsorted(elev_lons, clon - half)
    lon_hi = np.searchsorted(elev_lons, clon + half)

    sub = elevation[lat_lo:lat_hi, lon_lo:lon_hi]

    # Ocean pixels only
    ocean = sub[sub < 0]

    if len(ocean) > 0:
        cell['mean_depth'] = float(np.mean(ocean))
        cell['min_depth'] = float(np.min(ocean))  # deepest point
        cell['depth_std'] = float(np.std(ocean))
        cell['depth_range'] = float(np.max(ocean) - np.min(ocean))
        cell['n_ocean_pix'] = int(len(ocean))

        # Shelf fraction (0 to -200m)
        shelf = (ocean >= -200) & (ocean < 0)
        cell['shelf_frac'] = float(shelf.sum() / len(ocean))

        # Deep fraction (< -500m)
        deep = ocean < -500
        cell['deep_frac'] = float(deep.sum() / len(ocean))
    else:
        cell['mean_depth'] = 0
        cell['min_depth'] = 0
        cell['depth_std'] = 0
        cell['depth_range'] = 0
        cell['n_ocean_pix'] = 0
        cell['shelf_frac'] = 0
        cell['deep_frac'] = 0


# ============================================================
# ANALYSIS
# ============================================================
n = len(cells)
S_arr = np.array([c['S'] for c in cells])
logR_arr = np.array([c['logR'] for c in cells])
R_arr = np.array([c['R_i'] for c in cells])
mean_depth = np.array([c['mean_depth'] for c in cells])
min_depth = np.array([c['min_depth'] for c in cells])
depth_std = np.array([c['depth_std'] for c in cells])
depth_range = np.array([c['depth_range'] for c in cells])
shelf_frac = np.array([c['shelf_frac'] for c in cells])
deep_frac = np.array([c['deep_frac'] for c in cells])
lat_arr = np.array([c['lat'] for c in cells])


# ============================================================
# TEST 1: Correlations
# ============================================================
print(f"\n{'='*70}")
print("TEST 1: CORRELATIONS — oceanographic metrics vs logR and S")
print(f"{'='*70}")

depth_metrics = {
    'mean_depth': mean_depth,
    'min_depth': min_depth,
    'depth_std': depth_std,
    'depth_range': depth_range,
    'shelf_frac': shelf_frac,
    'deep_frac': deep_frac,
}

print(f"\n  {'Metric':<15} {'vs logR':>12} {'p':>8} {'vs S':>12} {'p':>8}")
print(f"  {'-'*55}")

correlations = {}
for name, arr in depth_metrics.items():
    rho_R, p_R = spearmanr(arr, logR_arr)
    rho_S, p_S = spearmanr(arr, S_arr)
    sig_R = "*" if p_R < 0.05 else ""
    sig_S = "*" if p_S < 0.05 else ""
    print(f"  {name:<15} {rho_R:>+8.3f} {sig_R:>3} {p_R:>8.4f}"
          f"  {rho_S:>+8.3f} {sig_S:>3} {p_S:>8.4f}")
    correlations[name] = {'rho_logR': round(float(rho_R), 4),
                           'p_logR': round(float(p_R), 4),
                           'rho_S': round(float(rho_S), 4),
                           'p_S': round(float(p_S), 4)}

# Reference: S vs logR
rho_sR, p_sR = spearmanr(S_arr, logR_arr)
print(f"  {'S (ref)':<15} {rho_sR:>+8.3f} {'*':>3} {p_sR:>8.4f}")


# ============================================================
# TEST 2: Find best depth predictor
# ============================================================
print(f"\n{'='*70}")
print("TEST 2: BEST DEPTH PREDICTOR — R² comparison")
print(f"{'='*70}")

ss_tot = np.sum((logR_arr - logR_arr.mean())**2)

# Model A: logR ~ S (reference)
X_a = np.column_stack([np.ones(n), S_arr])
b_a, _, _, _ = lstsq(X_a, logR_arr, rcond=None)
ss_a = np.sum((logR_arr - X_a @ b_a)**2)
r2_a = 1 - ss_a / ss_tot

print(f"\n  Model A (logR ~ S):           R² = {r2_a:.4f}")

best_depth_name = None
best_depth_r2 = -1
best_depth_arr = None

for name, arr in depth_metrics.items():
    X = np.column_stack([np.ones(n), arr])
    b, _, _, _ = lstsq(X, logR_arr, rcond=None)
    ss = np.sum((logR_arr - X @ b)**2)
    r2 = 1 - ss / ss_tot
    sig = "*" if r2 > r2_a else ""
    print(f"  Model D_{name:<12} (logR ~ {name:<12}): R² = {r2:.4f} {sig}")

    if r2 > best_depth_r2:
        best_depth_r2 = r2
        best_depth_name = name
        best_depth_arr = arr

print(f"\n  Best depth predictor: {best_depth_name} (R² = {best_depth_r2:.4f})")
print(f"  S R² = {r2_a:.4f}")

if best_depth_r2 > r2_a:
    print(f"  → Depth metric beats S!")
else:
    print(f"  → S beats all depth metrics")


# ============================================================
# TEST 3: Combined model
# ============================================================
print(f"\n{'='*70}")
print(f"TEST 3: COMBINED MODEL — logR ~ S + {best_depth_name}")
print(f"{'='*70}")

# Model B: logR ~ best_depth only
X_b = np.column_stack([np.ones(n), best_depth_arr])
b_b, _, _, _ = lstsq(X_b, logR_arr, rcond=None)
ss_b = np.sum((logR_arr - X_b @ b_b)**2)
r2_b = 1 - ss_b / ss_tot

# Model C: logR ~ S + best_depth
X_c = np.column_stack([np.ones(n), S_arr, best_depth_arr])
b_c, _, _, _ = lstsq(X_c, logR_arr, rcond=None)
ss_c = np.sum((logR_arr - X_c @ b_c)**2)
r2_c = 1 - ss_c / ss_tot

print(f"\n  Model A (S only):            R² = {r2_a:.4f}")
print(f"  Model B ({best_depth_name} only): R² = {r2_b:.4f}")
print(f"  Model C (S + {best_depth_name}):  R² = {r2_c:.4f}")
print(f"    β_S = {b_c[1]:+.4f}, β_{best_depth_name} = {b_c[2]:+.6f}")

# F-tests
df_full = n - 3
F_s_given_depth = ((ss_b - ss_c) / 1) / (ss_c / df_full)
p_s_given_depth = 1 - f_dist.cdf(F_s_given_depth, 1, df_full)

F_depth_given_s = ((ss_a - ss_c) / 1) / (ss_c / df_full)
p_depth_given_s = 1 - f_dist.cdf(F_depth_given_s, 1, df_full)

print(f"\n  F-test: S adds to {best_depth_name}?    F = {F_s_given_depth:.3f}, p = {p_s_given_depth:.4f}")
print(f"  F-test: {best_depth_name} adds to S?    F = {F_depth_given_s:.3f}, p = {p_depth_given_s:.4f}")

# Bootstrap
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

print(f"\n  Bootstrap CIs:")
for i, name in enumerate(['intercept', 'β_S', f'β_{best_depth_name}']):
    lo = np.percentile(boot_c[:, i], 2.5)
    hi = np.percentile(boot_c[:, i], 97.5)
    print(f"    {name:20s}: {b_c[i]:+.6f}  [{lo:+.6f}, {hi:+.6f}]")


# ============================================================
# TEST 4: Multiple depth model
# ============================================================
print(f"\n{'='*70}")
print("TEST 4: KITCHEN SINK — logR ~ S + mean_depth + depth_std + deep_frac")
print(f"{'='*70}")

X_ks = np.column_stack([np.ones(n), S_arr, mean_depth, depth_std, deep_frac])
b_ks, _, _, _ = lstsq(X_ks, logR_arr, rcond=None)
ss_ks = np.sum((logR_arr - X_ks @ b_ks)**2)
r2_ks = 1 - ss_ks / ss_tot

# Does S survive all depth controls?
X_depth_only = np.column_stack([np.ones(n), mean_depth, depth_std, deep_frac])
b_do, _, _, _ = lstsq(X_depth_only, logR_arr, rcond=None)
ss_do = np.sum((logR_arr - X_depth_only @ b_do)**2)
r2_do = 1 - ss_do / ss_tot

F_s_given_all_depth = ((ss_do - ss_ks) / 1) / (ss_ks / (n - 5))
p_s_given_all_depth = 1 - f_dist.cdf(F_s_given_all_depth, 1, n - 5)

print(f"\n  Depth-only (3 predictors):   R² = {r2_do:.4f}")
print(f"  S + Depth (4 predictors):    R² = {r2_ks:.4f}")
print(f"  β_S in kitchen sink:         {b_ks[1]:+.4f}")
print(f"  F-test: S adds to all depth? F = {F_s_given_all_depth:.3f}, p = {p_s_given_all_depth:.4f}")

if p_s_given_all_depth < 0.05:
    print(f"  → S SURVIVES controlling for all depth metrics")
else:
    print(f"  → S does NOT survive all depth controls")


# ============================================================
# TEST 5: Upwelling proxy — do canyon cells have colder nearby water?
# ============================================================
print(f"\n{'='*70}")
print("TEST 5: Canyon cells vs non-canyon — depth profile comparison")
print(f"{'='*70}")

s_pos = S_arr > 0
s_zero = S_arr == 0

from scipy.stats import mannwhitneyu
for name, arr in depth_metrics.items():
    if s_pos.sum() > 0 and s_zero.sum() > 0:
        U, p_mw = mannwhitneyu(arr[s_pos], arr[s_zero])
        print(f"  {name:<15}: S>0 mean = {arr[s_pos].mean():>8.1f}, "
              f"S=0 mean = {arr[s_zero].mean():>8.1f}, "
              f"MWU p = {p_mw:.4f}")


# ============================================================
# INTERPRETATION
# ============================================================
print(f"\n{'='*70}")
print("INTERPRETATION")
print(f"{'='*70}")

print(f"""
  If S loses significance when controlling for depth metrics:
    → Upwelling/oceanographic confound is supported (Option 2)
    → Canyon steepness is just a proxy for deep water nearby

  If S survives:
    → Canyon steepness adds information beyond depth alone
    → Pure depth/upwelling doesn't explain the pattern
    → Consistent with geometry-specific mechanism (CTH or
      canyon-channeled acoustics/optics)
""")


# ============================================================
# SAVE
# ============================================================
results = {
    "test": "Oceanographic confound (depth metrics vs S)",
    "n_cells": n,
    "correlations": correlations,
    "S_logR": {"rho": round(float(rho_sR), 4), "p": round(float(p_sR), 4)},
    "best_depth_predictor": best_depth_name,
    "models": {
        "A_S_only": {"R2": round(r2_a, 4)},
        "B_depth_only": {"predictor": best_depth_name, "R2": round(r2_b, 4)},
        "C_combined": {"R2": round(r2_c, 4)},
        "KS_all_depth": {"R2": round(r2_do, 4)},
        "KS_S_plus_depth": {"R2": round(r2_ks, 4)},
    },
    "F_tests": {
        "S_given_best_depth": {"F": round(F_s_given_depth, 3), "p": round(p_s_given_depth, 4)},
        "depth_given_S": {"F": round(F_depth_given_s, 3), "p": round(p_depth_given_s, 4)},
        "S_given_all_depth": {"F": round(F_s_given_all_depth, 3), "p": round(p_s_given_all_depth, 4)},
    },
}

out_file = os.path.join(OUT_DIR, "phase_e_ocean_confound.json")
with open(out_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_file}")

import shutil
repo_out = os.path.join(BASE_DIR, "uap-canyon-analysis", "results", "phase_ev2")
shutil.copy2(out_file, repo_out)
print(f"Copied to repo")

print(f"\nDONE ({elapsed()})")
