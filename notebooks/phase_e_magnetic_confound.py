#!/usr/bin/env python3
"""
Phase E: Magnetic Anomaly Confound Test
========================================
Tests whether the UAP-canyon correlation is explained by magnetic anomalies.

Hypothesis: Steep submarine gradients correlate with crustal magnetic anomalies
(lithological contrasts at shelf edge). If magnetic anomalies explain the UAP
excess better than canyon steepness (S), we have a candidate geophysical mechanism.

Data: EMAG2v3 (Sea Level, 2 arc-min resolution) from NOAA NCEI.
      Downloaded as regional float32 GeoTIFF subsets via ArcGIS ImageServer.

Tests:
  1. Correlation: S vs |mag_anomaly|
  2. Does |mag_anomaly| predict logR?
  3. Combined model: logR ~ S + |mag_anomaly|
  4. Does S survive controlling for magnetic anomaly?
  5. Kitchen sink: S + mag + depth
"""

import os, time, json, warnings
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import f as f_dist
from numpy.linalg import lstsq

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR = os.environ.get("UAP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "results", "phase_ev2")

GRID_DEG = 0.5

def elapsed():
    return f"{time.time()-t0:.1f}s"


print("=" * 70)
print("PHASE E: MAGNETIC ANOMALY CONFOUND TEST")
print("EMAG2v3 vs canyon steepness (S)")
print("=" * 70)


# ============================================================
# LOAD EMAG2v3 — regional float32 GeoTIFF
# ============================================================
print(f"\n[LOAD] EMAG2v3 magnetic anomaly grid (West Coast subset)... ({elapsed()})")
import rasterio

MAG_FILE = os.path.join(DATA_DIR, "emag2v3_westcoast.tif")
with rasterio.open(MAG_FILE) as src:
    mag_data = src.read(1)
    mag_transform = src.transform
    mag_nodata = src.nodata
    mag_bounds = src.bounds
    print(f"  Shape: {mag_data.shape}")
    print(f"  Bounds: {mag_bounds}")
    print(f"  dtype: {mag_data.dtype}")
    valid_mag = mag_data[np.isfinite(mag_data)]
    print(f"  Value range: {valid_mag.min():.1f} to {valid_mag.max():.1f} nT")
    print(f"  Mean |anomaly|: {np.abs(valid_mag).mean():.1f} nT")


# ============================================================
# LOAD E-RED v2 CELLS
# ============================================================
print(f"\n[LOAD] E-RED v2 cell data... ({elapsed()})")
with open(os.path.join(OUT_DIR, "phase_e_red_v2_evaluation.json")) as f:
    data = json.load(f)
cells = data["primary_200km"]["cell_details"]
print(f"  {len(cells)} testable West Coast cells")


# ============================================================
# COMPUTE MAGNETIC ANOMALY METRICS PER CELL
# ============================================================
print(f"\n[COMPUTE] Magnetic anomaly metrics per cell... ({elapsed()})")

half = GRID_DEG / 2

with rasterio.open(MAG_FILE) as src:
    for cell in cells:
        clat, clon = cell['lat'], cell['lon']

        # Get pixel window for this 0.5° cell
        try:
            row_top, col_left = src.index(clon - half, clat + half)
            row_bot, col_right = src.index(clon + half, clat - half)
        except Exception:
            cell['mag_mean'] = np.nan
            cell['mag_abs_mean'] = np.nan
            cell['mag_std'] = np.nan
            cell['mag_range'] = np.nan
            cell['mag_abs_max'] = np.nan
            cell['n_mag_pix'] = 0
            continue

        row_lo = max(0, min(row_top, row_bot))
        row_hi = min(mag_data.shape[0], max(row_top, row_bot))
        col_lo = max(0, min(col_left, col_right))
        col_hi = min(mag_data.shape[1], max(col_left, col_right))

        if row_lo < row_hi and col_lo < col_hi:
            sub = mag_data[row_lo:row_hi, col_lo:col_hi].astype(float)
            valid = sub[np.isfinite(sub)]

            if len(valid) > 0:
                cell['mag_mean'] = float(np.mean(valid))
                cell['mag_abs_mean'] = float(np.mean(np.abs(valid)))
                cell['mag_std'] = float(np.std(valid))
                cell['mag_range'] = float(np.max(valid) - np.min(valid))
                cell['mag_abs_max'] = float(np.max(np.abs(valid)))
                cell['n_mag_pix'] = int(len(valid))
            else:
                cell['mag_mean'] = np.nan
                cell['mag_abs_mean'] = np.nan
                cell['mag_std'] = np.nan
                cell['mag_range'] = np.nan
                cell['mag_abs_max'] = np.nan
                cell['n_mag_pix'] = 0
        else:
            cell['mag_mean'] = np.nan
            cell['mag_abs_mean'] = np.nan
            cell['mag_std'] = np.nan
            cell['mag_range'] = np.nan
            cell['mag_abs_max'] = np.nan
            cell['n_mag_pix'] = 0

# Filter to cells with valid magnetic data
valid_cells = [c for c in cells if np.isfinite(c.get('mag_abs_mean', np.nan)) and c['n_mag_pix'] > 0]
print(f"  {len(valid_cells)}/{len(cells)} cells with valid magnetic data")

# Stats
mag_abs = np.array([c['mag_abs_mean'] for c in valid_cells])
mag_std = np.array([c['mag_std'] for c in valid_cells])
mag_range = np.array([c['mag_range'] for c in valid_cells])
print(f"  |mag| mean: min={mag_abs.min():.1f}, max={mag_abs.max():.1f}, "
      f"mean={mag_abs.mean():.1f} nT")
print(f"  mag_std: min={mag_std.min():.1f}, max={mag_std.max():.1f}, "
      f"mean={mag_std.mean():.1f} nT")
print(f"  mag_range: min={mag_range.min():.1f}, max={mag_range.max():.1f}, "
      f"mean={mag_range.mean():.1f} nT")


# ============================================================
# ANALYSIS
# ============================================================
n = len(valid_cells)
S_arr = np.array([c['S'] for c in valid_cells])
logR_arr = np.array([c['logR'] for c in valid_cells])
R_arr = np.array([c['R_i'] for c in valid_cells])
lat_arr = np.array([c['lat'] for c in valid_cells])

mag_metrics = {
    'mag_abs_mean': mag_abs,
    'mag_std': mag_std,
    'mag_range': mag_range,
}


# ============================================================
# TEST 1: Correlations
# ============================================================
print(f"\n{'='*70}")
print("TEST 1: CORRELATIONS — magnetic anomaly vs logR and S")
print(f"{'='*70}")

print(f"\n  n = {n} cells")
print(f"\n  {'Metric':<15} {'vs logR':>12} {'p':>8} {'vs S':>12} {'p':>8}")
print(f"  {'-'*55}")

correlations = {}
for name, arr in mag_metrics.items():
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

rho_sR, p_sR = spearmanr(S_arr, logR_arr)
print(f"  {'S (ref)':<15} {rho_sR:>+8.3f} {'*':>3} {p_sR:>8.4f}")


# ============================================================
# TEST 2: Best magnetic predictor R²
# ============================================================
print(f"\n{'='*70}")
print("TEST 2: R² COMPARISON")
print(f"{'='*70}")

ss_tot = np.sum((logR_arr - logR_arr.mean())**2)

X_a = np.column_stack([np.ones(n), S_arr])
b_a, _, _, _ = lstsq(X_a, logR_arr, rcond=None)
ss_a = np.sum((logR_arr - X_a @ b_a)**2)
r2_a = 1 - ss_a / ss_tot

print(f"\n  Model A (logR ~ S):          R² = {r2_a:.4f}")

best_mag_name = None
best_mag_r2 = -999
best_mag_arr = None

for name, arr in mag_metrics.items():
    X = np.column_stack([np.ones(n), arr])
    b, _, _, _ = lstsq(X, logR_arr, rcond=None)
    ss = np.sum((logR_arr - X @ b)**2)
    r2 = 1 - ss / ss_tot
    sig = " <-- beats S!" if r2 > r2_a else ""
    print(f"  Model M_{name:<12} (logR ~ {name:<12}): R² = {r2:.4f}{sig}")

    if r2 > best_mag_r2:
        best_mag_r2 = r2
        best_mag_name = name
        best_mag_arr = arr

print(f"\n  Best magnetic predictor: {best_mag_name} (R² = {best_mag_r2:.4f})")


# ============================================================
# TEST 3: Combined model
# ============================================================
print(f"\n{'='*70}")
print(f"TEST 3: COMBINED MODEL — logR ~ S + {best_mag_name}")
print(f"{'='*70}")

X_b = np.column_stack([np.ones(n), best_mag_arr])
b_b, _, _, _ = lstsq(X_b, logR_arr, rcond=None)
ss_b = np.sum((logR_arr - X_b @ b_b)**2)
r2_b = 1 - ss_b / ss_tot

X_c = np.column_stack([np.ones(n), S_arr, best_mag_arr])
b_c, _, _, _ = lstsq(X_c, logR_arr, rcond=None)
ss_c = np.sum((logR_arr - X_c @ b_c)**2)
r2_c = 1 - ss_c / ss_tot

print(f"\n  Model A (S only):              R² = {r2_a:.4f}")
print(f"  Model B ({best_mag_name} only): R² = {r2_b:.4f}")
print(f"  Model C (S + {best_mag_name}):  R² = {r2_c:.4f}")
print(f"    beta_S = {b_c[1]:+.4f}, beta_{best_mag_name} = {b_c[2]:+.6f}")

df_full = n - 3
F_s_given_mag = ((ss_b - ss_c) / 1) / (ss_c / df_full)
p_s_given_mag = 1 - f_dist.cdf(F_s_given_mag, 1, df_full)

F_mag_given_s = ((ss_a - ss_c) / 1) / (ss_c / df_full)
p_mag_given_s = 1 - f_dist.cdf(F_mag_given_s, 1, df_full)

print(f"\n  F-test: S adds to {best_mag_name}?    F = {F_s_given_mag:.3f}, p = {p_s_given_mag:.4f}")
print(f"  F-test: {best_mag_name} adds to S?    F = {F_mag_given_s:.3f}, p = {p_mag_given_s:.4f}")

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

print(f"\n  Bootstrap CIs (95%):")
for i, bname in enumerate(['intercept', 'beta_S', f'beta_{best_mag_name}']):
    lo = np.percentile(boot_c[:, i], 2.5)
    hi = np.percentile(boot_c[:, i], 97.5)
    print(f"    {bname:20s}: {b_c[i]:+.6f}  [{lo:+.6f}, {hi:+.6f}]")


# ============================================================
# TEST 4: Kitchen sink — S + all mag + depth metrics
# ============================================================
print(f"\n{'='*70}")
print("TEST 4: S + MAGNETIC + DEPTH — does S still survive?")
print(f"{'='*70}")

# Load depth metrics from ETOPO
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

depth_std_arr = np.zeros(n)
mean_depth_arr = np.zeros(n)
for i, cell in enumerate(valid_cells):
    clat, clon = cell['lat'], cell['lon']
    lat_lo = np.searchsorted(elev_lats, clat - half)
    lat_hi = np.searchsorted(elev_lats, clat + half)
    lon_lo = np.searchsorted(elev_lons, clon - half)
    lon_hi = np.searchsorted(elev_lons, clon + half)
    sub = elevation[lat_lo:lat_hi, lon_lo:lon_hi]
    ocean = sub[sub < 0]
    if len(ocean) > 0:
        depth_std_arr[i] = np.std(ocean)
        mean_depth_arr[i] = np.mean(ocean)

# Full kitchen sink: S + best_mag + depth_std + mean_depth
X_full = np.column_stack([np.ones(n), S_arr, best_mag_arr, depth_std_arr, mean_depth_arr])
b_full, _, _, _ = lstsq(X_full, logR_arr, rcond=None)
ss_full = np.sum((logR_arr - X_full @ b_full)**2)
r2_full = 1 - ss_full / ss_tot

# Without S
X_no_s = np.column_stack([np.ones(n), best_mag_arr, depth_std_arr, mean_depth_arr])
b_no_s, _, _, _ = lstsq(X_no_s, logR_arr, rcond=None)
ss_no_s = np.sum((logR_arr - X_no_s @ b_no_s)**2)
r2_no_s = 1 - ss_no_s / ss_tot

F_s_final = ((ss_no_s - ss_full) / 1) / (ss_full / (n - 5))
p_s_final = 1 - f_dist.cdf(F_s_final, 1, n - 5)

print(f"\n  All confounds (mag + depth):  R² = {r2_no_s:.4f}")
print(f"  S + all confounds:            R² = {r2_full:.4f}")
print(f"  beta_S in full model:         {b_full[1]:+.4f}")
print(f"  F-test: S adds to everything? F = {F_s_final:.3f}, p = {p_s_final:.4f}")

if p_s_final < 0.05:
    print(f"\n  >>> S SURVIVES even after controlling for magnetic anomaly + depth")
    final_verdict = "S_SURVIVES_ALL"
else:
    print(f"\n  >>> S does NOT survive magnetic + depth controls")
    final_verdict = "S_ABSORBED"


# ============================================================
# TEST 5: Canyon vs non-canyon magnetic comparison
# ============================================================
print(f"\n{'='*70}")
print("TEST 5: Canyon cells vs non-canyon — magnetic anomaly comparison")
print(f"{'='*70}")

s_pos = S_arr > 0
s_zero = S_arr == 0
print(f"  S>0: {s_pos.sum()} cells, S=0: {s_zero.sum()} cells")

from scipy.stats import mannwhitneyu
for name, arr in mag_metrics.items():
    if s_pos.sum() > 0 and s_zero.sum() > 0:
        U, p_mw = mannwhitneyu(arr[s_pos], arr[s_zero], alternative='two-sided')
        sig = "*" if p_mw < 0.05 else ""
        print(f"  {name:<15}: S>0 mean = {arr[s_pos].mean():>8.1f}, "
              f"S=0 mean = {arr[s_zero].mean():>8.1f}, "
              f"MWU p = {p_mw:.4f} {sig}")


# ============================================================
# TEST 6: S vs mag collinearity
# ============================================================
print(f"\n{'='*70}")
print("TEST 6: COLLINEARITY CHECK — S vs magnetic metrics")
print(f"{'='*70}")

for name, arr in mag_metrics.items():
    rho, p = spearmanr(S_arr, arr)
    print(f"  S vs {name:<15}: rho = {rho:+.3f}, p = {p:.4f}")


# ============================================================
# INTERPRETATION
# ============================================================
print(f"\n{'='*70}")
print("INTERPRETATION")
print(f"{'='*70}")

if p_s_given_mag < 0.05 and p_mag_given_s >= 0.05:
    print(f"  S survives mag, mag doesn't add to S")
    print(f"  -> Magnetic anomaly is NOT a better explanation")
    print(f"  -> Canyon geometry carries unique information")
    interp = "S_DOMINANT"
elif p_s_given_mag >= 0.05 and p_mag_given_s < 0.05:
    print(f"  Mag absorbs S!")
    print(f"  -> Magnetic anomaly IS a candidate mechanism")
    print(f"  -> Canyon steepness may just proxy for mag anomalies")
    interp = "MAG_ABSORBS_S"
elif p_s_given_mag < 0.05 and p_mag_given_s < 0.05:
    print(f"  Both survive — independent contributions")
    print(f"  -> Both canyon geometry AND magnetic anomalies matter")
    interp = "BOTH_CONTRIBUTE"
else:
    print(f"  Neither significant in combined model")
    print(f"  -> Collinearity may prevent separation")
    interp = "COLLINEAR"


# ============================================================
# SAVE
# ============================================================
results = {
    "test": "Magnetic anomaly confound (EMAG2v3)",
    "data_source": "EMAG2v3 via NOAA NCEI ArcGIS ImageServer (float32 GeoTIFF, 2 arc-min)",
    "n_cells": n,
    "correlations": correlations,
    "S_vs_logR": {"rho": round(float(rho_sR), 4), "p": round(float(p_sR), 4)},
    "models": {
        "A_S_only": {"R2": round(r2_a, 4)},
        "B_mag_only": {"predictor": best_mag_name, "R2": round(best_mag_r2, 4)},
        "C_combined": {"R2": round(r2_c, 4)},
        "full_all_confounds": {"R2": round(r2_no_s, 4)},
        "full_with_S": {"R2": round(r2_full, 4)},
    },
    "F_tests": {
        "S_given_mag": {"F": round(F_s_given_mag, 3), "p": round(p_s_given_mag, 4)},
        "mag_given_S": {"F": round(F_mag_given_s, 3), "p": round(p_mag_given_s, 4)},
        "S_given_all": {"F": round(F_s_final, 3), "p": round(p_s_final, 4)},
    },
    "interpretation": interp,
    "final_verdict": final_verdict,
}

out_file = os.path.join(OUT_DIR, "phase_e_magnetic_confound.json")
with open(out_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_file}")


print(f"\nDONE ({elapsed()})")
