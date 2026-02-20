#!/usr/bin/env python3
"""
Phase E: NOAA ESI Shoreline Classification Test (Option 3 confound, clean version)
==================================================================================
Uses actual NOAA ESI shoreline classification data (not ETOPO proxy) to test
whether rocky/cliffy coastline explains the UAP excess better than underwater
canyon score (S).

ESI codes grouped as:
  ROCKY: 1A, 1B, 1C, 2A, 2B, 8A  (exposed/sheltered rock)
  SANDY: 3A, 3B, 4                (sand beaches)
  GRAVEL: 5, 6A, 6B, 6C          (gravel/mixed/riprap)
  MUD_FLAT: 7, 9A, 9B            (tidal flats, vegetated banks)
  MARSH: 10A, 10B, 10C, 10D      (marshes, wetlands, mangroves)
  MANMADE: 8B, 8C, 8D, 8E        (sheltered structures, rubble, peat)

Data: California only (Southern, Central, Northern GDBs from NOAA).
Washington ESI not available for download.

For each 0.5° cell: compute fraction of shoreline length that is ROCKY.
Then test: does rocky_frac explain logR, and does S survive controlling for it?
"""

import os, time, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from scipy.stats import f as f_dist
from numpy.linalg import lstsq

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR = os.environ.get("UAP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(BASE_DIR, "data")
ESI_DIR  = os.path.join(DATA_DIR, "esi")
OUT_DIR  = os.path.join(BASE_DIR, "phase_ev2")

GRID_DEG = 0.5

def elapsed():
    return f"{time.time()-t0:.1f}s"

# Rocky ESI codes
ROCKY_CODES = {'1A', '1B', '1C', '2A', '2B', '8A'}

print("=" * 70)
print("PHASE E: NOAA ESI SHORELINE CLASSIFICATION TEST")
print("Rocky shoreline fraction vs canyon score (S)")
print("=" * 70)

# ============================================================
# LOAD ESI SHORELINE DATA
# ============================================================
print(f"\n[LOAD] ESI shoreline data from California GDBs... ({elapsed()})")
import fiona
from shapely.geometry import shape

esi_gdbs = [
    ("SouthernCal", os.path.join(ESI_DIR, "SouthernCal/SouthernCaliforniaESI.gdb"), "esi_arc"),
    ("CentralCal", os.path.join(ESI_DIR, "CentralCal/CentralCaliforniaESI.gdb"), "esil_arc"),
    ("NorthernCal", os.path.join(ESI_DIR, "NorthernCal/NorthernCal_2008_GDB/NorthernCaliforniaESI.gdb"), "esil"),
]

all_segments = []  # list of (esi_code, length_deg, centroid_lat, centroid_lon)

for region_name, gdb_path, layer_name in esi_gdbs:
    with fiona.open(gdb_path, layer=layer_name) as src:
        n_features = len(src)
        count = 0
        for feat in src:
            esi_code = feat['properties'].get('ESI', '')
            if not esi_code:
                continue
            # Get the most sensitive classification
            most_sens = feat['properties'].get('MOSTSENSIT', esi_code)
            geom = shape(feat['geometry'])
            # Use length in degrees as proxy for relative length
            length = geom.length
            centroid = geom.centroid
            all_segments.append({
                'esi': str(most_sens).strip(),
                'length': length,
                'lat': centroid.y,
                'lon': centroid.x,
                'region': region_name,
            })
            count += 1
        print(f"  {region_name}: {count} shoreline segments from {n_features} features")

print(f"  Total: {len(all_segments)} segments")

# Show ESI code distribution
esi_codes = [s['esi'] for s in all_segments]
from collections import Counter
code_counts = Counter(esi_codes)
print(f"\n  ESI code distribution (top 15):")
for code, cnt in code_counts.most_common(15):
    is_rocky = "ROCKY" if code in ROCKY_CODES else ""
    print(f"    {code:5s}: {cnt:5d} segments  {is_rocky}")

# Mark rocky
for seg in all_segments:
    seg['is_rocky'] = seg['esi'] in ROCKY_CODES

n_rocky = sum(1 for s in all_segments if s['is_rocky'])
print(f"\n  Rocky segments: {n_rocky}/{len(all_segments)} ({100*n_rocky/len(all_segments):.1f}%)")


# ============================================================
# LOAD E-RED v2 CELL DETAILS
# ============================================================
print(f"\n[LOAD] E-RED v2 cell data... ({elapsed()})")
with open(os.path.join(OUT_DIR, "phase_e_red_v2_evaluation.json")) as f:
    data = json.load(f)
cells = data["primary_200km"]["cell_details"]
print(f"  {len(cells)} testable West Coast cells")


# ============================================================
# COMPUTE ROCKY FRACTION PER CELL
# ============================================================
print(f"\n[COMPUTE] Rocky shoreline fraction per cell... ({elapsed()})")

half = GRID_DEG / 2

for cell in cells:
    clat, clon = cell['lat'], cell['lon']
    # Find ESI segments within this cell
    cell_segs = [s for s in all_segments
                 if abs(s['lat'] - clat) <= half and abs(s['lon'] - clon) <= half]

    total_length = sum(s['length'] for s in cell_segs)
    rocky_length = sum(s['length'] for s in cell_segs if s['is_rocky'])

    cell['n_esi_segments'] = len(cell_segs)
    cell['total_esi_length'] = total_length
    cell['rocky_length'] = rocky_length
    cell['rocky_frac'] = rocky_length / total_length if total_length > 0 else None

# Count cells with ESI coverage
cells_with_esi = [c for c in cells if c['rocky_frac'] is not None]
cells_no_esi = [c for c in cells if c['rocky_frac'] is None]

print(f"  Cells with ESI data: {len(cells_with_esi)}")
print(f"  Cells without ESI data: {len(cells_no_esi)} (outside California)")

if cells_no_esi:
    print(f"  Missing cells (lat range): {min(c['lat'] for c in cells_no_esi):.1f}–{max(c['lat'] for c in cells_no_esi):.1f}")

# Show rocky fraction distribution
rf_arr = np.array([c['rocky_frac'] for c in cells_with_esi])
print(f"\n  Rocky fraction: min={rf_arr.min():.3f}, max={rf_arr.max():.3f}, "
      f"mean={rf_arr.mean():.3f}, median={np.median(rf_arr):.3f}")


# ============================================================
# ANALYSIS — California cells only (cells with ESI data)
# ============================================================
S_arr = np.array([c['S'] for c in cells_with_esi])
logR_arr = np.array([c['logR'] for c in cells_with_esi])
R_arr = np.array([c['R_i'] for c in cells_with_esi])
rocky_arr = np.array([c['rocky_frac'] for c in cells_with_esi])
lat_arr = np.array([c['lat'] for c in cells_with_esi])
n = len(cells_with_esi)

print(f"\n  Working with n = {n} California cells")
print(f"  S range: [{S_arr.min():.3f}, {S_arr.max():.3f}]")
print(f"  n with S > 0: {(S_arr > 0).sum()}")


# ============================================================
# TEST 1: S vs rocky_frac correlation
# ============================================================
print(f"\n{'='*70}")
print("TEST 1: S (underwater canyon) vs rocky_frac (shoreline type)")
print(f"{'='*70}")

rho_sr, p_sr = spearmanr(S_arr, rocky_arr)
r_sr, p_sr_p = pearsonr(S_arr, rocky_arr)
print(f"  Spearman(S, rocky_frac): rho = {rho_sr:.3f}, p = {p_sr:.4f}")
print(f"  Pearson(S, rocky_frac):  r = {r_sr:.3f}, p = {p_sr_p:.4f}")

if abs(rho_sr) > 0.8:
    print(f"  WARNING: HIGH CORRELATION — confound cannot be separated")
elif abs(rho_sr) > 0.5:
    print(f"  CAUTION: MODERATE CORRELATION — partial separation possible")
else:
    print(f"  OK: LOW-MODERATE — predictors are separable")


# ============================================================
# TEST 2: rocky_frac → logR
# ============================================================
print(f"\n{'='*70}")
print("TEST 2: rocky_frac → logR")
print(f"{'='*70}")

rho_rR, p_rR = spearmanr(rocky_arr, logR_arr)
rho_sR, p_sR = spearmanr(S_arr, logR_arr)
print(f"  Spearman(rocky_frac, logR): rho = {rho_rR:.3f}, p = {p_rR:.4f}")
print(f"  Spearman(S, logR):          rho = {rho_sR:.3f}, p = {p_sR:.4f}")


# ============================================================
# TEST 3: Competing models
# ============================================================
print(f"\n{'='*70}")
print("TEST 3: COMPETING MODELS (OLS)")
print(f"{'='*70}")

# Model A: logR ~ S only
X_a = np.column_stack([np.ones(n), S_arr])
b_a, _, _, _ = lstsq(X_a, logR_arr, rcond=None)
ss_a = np.sum((logR_arr - X_a @ b_a)**2)
ss_tot = np.sum((logR_arr - logR_arr.mean())**2)
r2_a = 1 - ss_a / ss_tot

# Model B: logR ~ rocky only
X_b = np.column_stack([np.ones(n), rocky_arr])
b_b, _, _, _ = lstsq(X_b, logR_arr, rcond=None)
ss_b = np.sum((logR_arr - X_b @ b_b)**2)
r2_b = 1 - ss_b / ss_tot

# Model C: logR ~ S + rocky
X_c = np.column_stack([np.ones(n), S_arr, rocky_arr])
b_c, _, _, _ = lstsq(X_c, logR_arr, rcond=None)
ss_c = np.sum((logR_arr - X_c @ b_c)**2)
r2_c = 1 - ss_c / ss_tot

print(f"\n  Model A (logR ~ S):            beta_S = {b_a[1]:+.4f}, R2 = {r2_a:.4f}")
print(f"  Model B (logR ~ rocky):        beta_rocky = {b_b[1]:+.4f}, R2 = {r2_b:.4f}")
print(f"  Model C (logR ~ S + rocky):    beta_S = {b_c[1]:+.4f}, beta_rocky = {b_c[2]:+.4f}, R2 = {r2_c:.4f}")

# F-tests
df_full = n - 3
F_s_given_rocky = ((ss_b - ss_c) / 1) / (ss_c / df_full) if ss_c > 0 else 0
p_s_given_rocky = 1 - f_dist.cdf(F_s_given_rocky, 1, df_full)

F_rocky_given_s = ((ss_a - ss_c) / 1) / (ss_c / df_full) if ss_c > 0 else 0
p_rocky_given_s = 1 - f_dist.cdf(F_rocky_given_s, 1, df_full)

print(f"\n  F-test: S adds to rocky-only?     F = {F_s_given_rocky:.3f}, p = {p_s_given_rocky:.4f}")
print(f"  F-test: rocky adds to S-only?     F = {F_rocky_given_s:.3f}, p = {p_rocky_given_s:.4f}")

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

print(f"\n  Model C bootstrap CIs (5000 resamples):")
for i, name in enumerate(['intercept', 'beta_S', 'beta_rocky']):
    lo = np.percentile(boot_c[:, i], 2.5)
    hi = np.percentile(boot_c[:, i], 97.5)
    print(f"    {name:15s}: {b_c[i]:+.4f}  [{lo:+.4f}, {hi:+.4f}]")


# ============================================================
# TEST 4: Model D — logR ~ S + rocky + S×rocky (interaction)
# ============================================================
print(f"\n{'='*70}")
print("TEST 4: INTERACTION MODEL logR ~ S + rocky + S*rocky")
print(f"{'='*70}")

X_d = np.column_stack([np.ones(n), S_arr, rocky_arr, S_arr * rocky_arr])
b_d, _, _, _ = lstsq(X_d, logR_arr, rcond=None)
ss_d = np.sum((logR_arr - X_d @ b_d)**2)
r2_d = 1 - ss_d / ss_tot

F_interact = ((ss_c - ss_d) / 1) / (ss_d / (n - 4))
p_interact = 1 - f_dist.cdf(F_interact, 1, n - 4)

print(f"  beta_S = {b_d[1]:+.4f}, beta_rocky = {b_d[2]:+.4f}, beta_SxR = {b_d[3]:+.4f}")
print(f"  R2 = {r2_d:.4f}")
print(f"  F-test for interaction: F = {F_interact:.3f}, p = {p_interact:.4f}")


# ============================================================
# TEST 5: Show cells by rocky_frac quartile
# ============================================================
print(f"\n{'='*70}")
print("TEST 5: Rate ratio by rocky_frac quartile")
print(f"{'='*70}")

q25, q50, q75 = np.percentile(rocky_arr, [25, 50, 75])
quartile_labels = ['Q1 (least rocky)', 'Q2', 'Q3', 'Q4 (most rocky)']
quartile_bounds = [(-0.01, q25), (q25, q50), (q50, q75), (q75, 1.01)]

print(f"  Quartile boundaries: {q25:.3f}, {q50:.3f}, {q75:.3f}")
print(f"\n  {'Quartile':<20} {'n':>4} {'mean_S':>8} {'mean_R':>8} {'med_R':>8} {'mean_rocky':>11}")

for label, (lo, hi) in zip(quartile_labels, quartile_bounds):
    mask = (rocky_arr > lo) & (rocky_arr <= hi)
    if mask.sum() == 0:
        continue
    ms = S_arr[mask].mean()
    mr = R_arr[mask].mean()
    medr = np.median(R_arr[mask])
    mrf = rocky_arr[mask].mean()
    print(f"  {label:<20} {mask.sum():>4} {ms:>8.3f} {mr:>8.2f} {medr:>8.2f} {mrf:>11.3f}")


# ============================================================
# TEST 6: Among S>0 cells — does rocky explain variation?
# ============================================================
print(f"\n{'='*70}")
print("TEST 6: Among S>0 cells only — rocky vs S")
print(f"{'='*70}")

s_pos = S_arr > 0
n_pos = s_pos.sum()

if n_pos >= 5:
    rho_sr_pos, p_sr_pos = spearmanr(S_arr[s_pos], rocky_arr[s_pos])
    rho_rR_pos, p_rR_pos = spearmanr(rocky_arr[s_pos], logR_arr[s_pos])
    rho_sR_pos, p_sR_pos = spearmanr(S_arr[s_pos], logR_arr[s_pos])

    print(f"  n = {n_pos} cells with S > 0")
    print(f"  Spearman(S, rocky):    rho = {rho_sr_pos:.3f}, p = {p_sr_pos:.4f}")
    print(f"  Spearman(rocky, logR): rho = {rho_rR_pos:.3f}, p = {p_rR_pos:.4f}")
    print(f"  Spearman(S, logR):     rho = {rho_sR_pos:.3f}, p = {p_sR_pos:.4f}")


# ============================================================
# TEST 7: Rocky fraction — S>0 vs S=0 comparison
# ============================================================
print(f"\n{'='*70}")
print("TEST 7: Rocky fraction — S>0 vs S=0 cells")
print(f"{'='*70}")

s_zero = S_arr == 0
if s_zero.sum() > 0 and s_pos.sum() > 0:
    print(f"  S=0 cells (n={s_zero.sum()}): mean rocky = {rocky_arr[s_zero].mean():.3f}, "
          f"median = {np.median(rocky_arr[s_zero]):.3f}")
    print(f"  S>0 cells (n={s_pos.sum()}): mean rocky = {rocky_arr[s_pos].mean():.3f}, "
          f"median = {np.median(rocky_arr[s_pos]):.3f}")

    U, p_mw = mannwhitneyu(rocky_arr[s_pos], rocky_arr[s_zero], alternative='greater')
    print(f"  Mann-Whitney (S>0 rockier than S=0): U={U:.0f}, p={p_mw:.4f}")

    if p_mw < 0.05:
        print(f"  WARNING: Canyon cells are significantly rockier — confound is plausible!")
    else:
        print(f"  OK: No significant difference — rocky confound less likely")


# ============================================================
# COMPARE WITH ETOPO PROXY
# ============================================================
print(f"\n{'='*70}")
print("COMPARISON: ESI rocky_frac vs ETOPO cliff_score")
print(f"{'='*70}")

# Load ETOPO proxy results
with open(os.path.join(OUT_DIR, "phase_e_shoretype_proxy.json")) as f:
    etopo_results = json.load(f)

print(f"  ETOPO proxy: Spearman(S, cliff) = {etopo_results['S_cliff_correlation']['spearman_rho']:.3f}")
print(f"  ESI actual:  Spearman(S, rocky) = {rho_sr:.3f}")
print(f"  ETOPO proxy: Spearman(cliff, logR) = {etopo_results['cliff_logR_spearman']:.3f}")
print(f"  ESI actual:  Spearman(rocky, logR) = {rho_rR:.3f}")

# Also correlate rocky_frac with cliff_score for cells that have both
cliff_scores = {}
for cell in cells:
    key = (cell['lat'], cell['lon'])
    cliff_scores[key] = cell.get('cliff_score', None)

both = [(c['rocky_frac'], cliff_scores.get((c['lat'], c['lon'])))
        for c in cells_with_esi
        if cliff_scores.get((c['lat'], c['lon'])) is not None]

if both:
    rf_both = np.array([b[0] for b in both])
    cs_both = np.array([b[1] for b in both])
    rho_rc, p_rc = spearmanr(rf_both, cs_both)
    print(f"\n  ESI rocky_frac vs ETOPO cliff_score: Spearman = {rho_rc:.3f}, p = {p_rc:.4f}")
    print(f"  (n = {len(both)} cells with both measures)")


# ============================================================
# SAVE
# ============================================================
results = {
    "test": "NOAA ESI shoreline classification (California only)",
    "n_cells": n,
    "n_esi_segments": len(all_segments),
    "rocky_codes": sorted(list(ROCKY_CODES)),
    "S_rocky_correlation": {
        "spearman_rho": round(float(rho_sr), 4),
        "spearman_p": round(float(p_sr), 4),
        "pearson_r": round(float(r_sr), 4),
    },
    "rocky_logR_spearman": round(float(rho_rR), 4),
    "S_logR_spearman": round(float(rho_sR), 4),
    "models": {
        "A_S_only": {"beta_S": round(float(b_a[1]), 4), "R2": round(r2_a, 4)},
        "B_rocky_only": {"beta_rocky": round(float(b_b[1]), 4), "R2": round(r2_b, 4)},
        "C_both": {
            "beta_S": round(float(b_c[1]), 4),
            "beta_rocky": round(float(b_c[2]), 4),
            "R2": round(r2_c, 4),
        },
    },
    "F_tests": {
        "S_given_rocky": {"F": round(F_s_given_rocky, 3), "p": round(p_s_given_rocky, 4)},
        "rocky_given_S": {"F": round(F_rocky_given_s, 3), "p": round(p_rocky_given_s, 4)},
    },
    "interaction": {
        "beta_SxR": round(float(b_d[3]), 4),
        "R2": round(r2_d, 4),
        "F": round(F_interact, 3),
        "p": round(p_interact, 4),
    },
}

out_file = os.path.join(OUT_DIR, "phase_e_esi_shoretype.json")
with open(out_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_file}")

import shutil
repo_out = os.path.join(BASE_DIR, "uap-canyon-analysis", "results", "phase_ev2")
shutil.copy2(out_file, repo_out)
print(f"Copied to repo")

print(f"\nDONE ({elapsed()})")
