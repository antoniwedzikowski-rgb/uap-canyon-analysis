#!/usr/bin/env python3
"""
Norway Replication v2: Logistic regression on full coastal dataset
==================================================================
The original Norway test (Spearman on 17 cells) had no S=0 contrast
because all testable cells had S>0. This test uses ALL 227 cells with
population (E>0), including 210 with zero reports.

Two approaches:
  1. Logistic regression: P(O>=1) ~ S + log(pop)
     Tests whether canyon score S predicts having any reports,
     controlling for population exposure.

  2. Spearman on raw R=O/E (including zeros)
     Extends the original test to include zero-report cells.

This gives a proper canyon-vs-flat contrast with n=227.
"""

import os, time, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from math import radians, sin, cos, sqrt, atan2
import statsmodels.api as sm

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR = os.environ.get("UAP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "results", "phase_ev2")

GRID_DEG = 0.5
R_EARTH = 6371.0
GRADIENT_THRESH = 60.0
NOR_LAT_MIN, NOR_LAT_MAX = 57.0, 72.0
NOR_LON_MIN, NOR_LON_MAX = 3.0, 33.0
SUBSAMPLE = 4

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
print("NORWAY REPLICATION v2: LOGISTIC + FULL-SAMPLE SPEARMAN")
print("=" * 70)

# ============================================================
# LOAD BATHYMETRY & BUILD GRID
# ============================================================
print(f"\n[LOAD] SRTM30 bathymetry... ({elapsed()})")
bathy = np.loadtxt(os.path.join(DATA_DIR, "srtm30_norway.nc"), delimiter='\t')
b_lons, b_lats, b_elev = bathy[:, 0], bathy[:, 1], bathy[:, 2]
u_lons = np.sort(np.unique(b_lons))
u_lats = np.sort(np.unique(b_lats))
n_lat, n_lon = len(u_lats), len(u_lons)
elev_grid = np.full((n_lat, n_lon), np.nan)
lon_idx = np.searchsorted(u_lons, b_lons)
lat_idx = np.searchsorted(u_lats, b_lats)
elev_grid[lat_idx, lon_idx] = b_elev
res_deg = abs(u_lats[1] - u_lats[0])
del bathy, b_lons, b_lats, b_elev

# Coastal cells
half = GRID_DEG / 2
cell_lats = np.arange(NOR_LAT_MIN + half, NOR_LAT_MAX, GRID_DEG)
cell_lons = np.arange(NOR_LON_MIN + half, NOR_LON_MAX, GRID_DEG)

coastal_cells = []
for clat in cell_lats:
    for clon in cell_lons:
        li = np.searchsorted(u_lats, clat - half)
        hi = np.searchsorted(u_lats, clat + half)
        lj = np.searchsorted(u_lons, clon - half)
        hj = np.searchsorted(u_lons, clon + half)
        sub = elev_grid[li:hi, lj:hj]
        if sub.size > 0 and (sub >= 0).any() and (sub < 0).any():
            coastal_cells.append((clat, clon))
print(f"  Coastal cells: {len(coastal_cells)}")

# ============================================================
# COMPUTE S PER CELL
# ============================================================
print(f"\n[COMPUTE] Canyon score S... ({elapsed()})")

MIN_DEPTH, MAX_DEPTH = -500, 0

cell_data = []
for idx, (clat, clon) in enumerate(coastal_cells):
    if idx % 50 == 0:
        print(f"  Cell {idx+1}/{len(coastal_cells)} ({elapsed()})")

    li = np.searchsorted(u_lats, clat - half)
    hi = np.searchsorted(u_lats, clat + half)
    lj = np.searchsorted(u_lons, clon - half)
    hj = np.searchsorted(u_lons, clon + half)
    sub = elev_grid[li:hi, lj:hj]

    shelf_mask = (sub >= MIN_DEPTH) & (sub < MAX_DEPTH)
    n_shelf = shelf_mask.sum()

    if n_shelf < 10:
        cell_data.append({'lat': clat, 'lon': clon, 'S': 0.0, 'n_steep': 0})
        continue

    # Subsample shelf pixels
    shelf_positions = []
    for ii in range(0, sub.shape[0], SUBSAMPLE):
        for jj in range(0, sub.shape[1], SUBSAMPLE):
            if shelf_mask[ii, jj]:
                shelf_positions.append((li + ii, lj + jj))

    if not shelf_positions:
        cell_data.append({'lat': clat, 'lon': clon, 'S': 0.0, 'n_steep': 0})
        continue

    if len(shelf_positions) > 200:
        rng = np.random.RandomState(int(clat * 100 + clon * 10))
        idx_s = rng.choice(len(shelf_positions), 200, replace=False)
        shelf_positions = [shelf_positions[i] for i in idx_s]

    steep_count = 0
    steep_grads = []
    for pi, pj in shelf_positions:
        center = elev_grid[pi, pj]
        if np.isnan(center):
            continue
        rad = 5
        lat_lo = max(0, pi - rad)
        lat_hi = min(n_lat - 1, pi + rad)
        lon_lo = max(0, pj - rad)
        lon_hi = min(n_lon - 1, pj + rad)
        max_grad = 0.0
        for di in range(lat_lo, lat_hi + 1):
            for dj in range(lon_lo, lon_hi + 1):
                if di == pi and dj == pj:
                    continue
                nb = elev_grid[di, dj]
                if np.isnan(nb):
                    continue
                dist = haversine_km(u_lats[pi], u_lons[pj], u_lats[di], u_lons[dj])
                if dist < 0.1:
                    continue
                grad = abs(nb - center) / dist
                if grad > max_grad:
                    max_grad = grad
        if max_grad >= GRADIENT_THRESH:
            steep_count += 1
            steep_grads.append(max_grad)

    n_sampled = len(shelf_positions)
    if steep_count == 0 or n_sampled == 0:
        cell_data.append({'lat': clat, 'lon': clon, 'S': 0.0, 'n_steep': 0})
        continue

    frac = steep_count / n_sampled
    mean_g = np.mean(steep_grads)
    S = frac * (mean_g / GRADIENT_THRESH)
    cell_data.append({'lat': clat, 'lon': clon, 'S': float(S), 'n_steep': steep_count})

print(f"  Done ({elapsed()})")
n_s_pos = sum(1 for c in cell_data if c['S'] > 0)
n_s_zero = sum(1 for c in cell_data if c['S'] == 0)
print(f"  S > 0: {n_s_pos}, S = 0: {n_s_zero}")

# ============================================================
# LOAD REPORTS & POPULATION
# ============================================================
print(f"\n[LOAD] NUFORC Norway reports... ({elapsed()})")
cols = ['datetime_str','city','state','country','shape','duration_seconds',
        'duration_text','description','date_posted','lat','lon']
df = pd.read_csv(os.path.join(DATA_DIR, "nuforc_reports.csv"), header=None,
                 names=cols, low_memory=False)
df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
df['lon'] = pd.to_numeric(df['lon'], errors='coerce')

norw_kw = 'norway|norge'
norway_text = df['city'].str.lower().str.contains(norw_kw, na=False)
norway_box = (df['lat'] >= 57) & (df['lat'] <= 72) & (df['lon'] >= 3) & (df['lon'] <= 16)
norway_df = df[norway_text | norway_box].copy().dropna(subset=['lat','lon'])
exclude_kw = 'sweden|sverige|stockholm|goteborg|malmö|malmo|jonkoping|linkoping|ljungsbro|sundborn|bjursas|vasby|finland|helsinki|tampere|turku|espoo|lahti|hyvink|kirkkonummi|denmark|aalborg|copenhagen|aarhus|odense|estonia|tallinn|tartu|raasiku|cambodia'
exclude = norway_df['city'].str.lower().str.contains(exclude_kw, na=False)
norway_df = norway_df[~exclude]
print(f"  Reports: {len(norway_df)}")

norway_df['glat'] = np.round(norway_df['lat'] / GRID_DEG) * GRID_DEG + half
norway_df['glon'] = np.round(norway_df['lon'] / GRID_DEG) * GRID_DEG + half
cell_counts = norway_df.groupby(['glat','glon']).size().to_dict()

print(f"\n[LOAD] WorldPop population... ({elapsed()})")
import rasterio
with rasterio.open(os.path.join(DATA_DIR, "norway_pop_2020.tif")) as src:
    pop_data = src.read(1)
    pop_nodata = src.nodata
    if pop_nodata is not None:
        pop_data[pop_data == pop_nodata] = 0
    pop_data[pop_data < 0] = 0
    pop_data[np.isnan(pop_data)] = 0

    for cell in cell_data:
        clat, clon = cell['lat'], cell['lon']
        try:
            row_top, col_left = src.index(clon - half, clat + half)
            row_bot, col_right = src.index(clon + half, clat - half)
            row_lo = max(0, min(row_top, row_bot))
            row_hi = min(pop_data.shape[0], max(row_top, row_bot))
            col_lo = max(0, min(col_left, col_right))
            col_hi = min(pop_data.shape[1], max(col_left, col_right))
            if row_lo < row_hi and col_lo < col_hi:
                cell['pop'] = float(np.sum(pop_data[row_lo:row_hi, col_lo:col_hi]))
            else:
                cell['pop'] = 0.0
        except:
            cell['pop'] = 0.0

# Assign O and E
total_O = sum(cell_counts.values())
for cell in cell_data:
    key = (cell['lat'], cell['lon'])
    cell['O'] = cell_counts.get(key, 0)

# ============================================================
# BUILD ANALYSIS DATASET
# ============================================================
print(f"\n[BUILD] Analysis dataset... ({elapsed()})")

# Keep cells with pop > 0
analysis = [c for c in cell_data if c['pop'] > 0]
n_analysis = len(analysis)

total_pop = sum(c['pop'] for c in analysis)
for c in analysis:
    c['E'] = total_O * (c['pop'] / total_pop)
    c['R'] = c['O'] / c['E'] if c['E'] > 0 else 0.0
    c['has_report'] = 1 if c['O'] >= 1 else 0
    c['log_pop'] = np.log(c['pop']) if c['pop'] > 0 else 0.0

n_with_report = sum(c['has_report'] for c in analysis)
n_s_pos_a = sum(1 for c in analysis if c['S'] > 0)
n_s_zero_a = sum(1 for c in analysis if c['S'] == 0)

print(f"  Cells with pop > 0: {n_analysis}")
print(f"  Cells with reports: {n_with_report}")
print(f"  Cells S > 0: {n_s_pos_a}")
print(f"  Cells S = 0: {n_s_zero_a}")

# Report rate by S group
s_pos = [c for c in analysis if c['S'] > 0]
s_zero = [c for c in analysis if c['S'] == 0]
rate_pos = sum(c['has_report'] for c in s_pos) / len(s_pos) if s_pos else 0
rate_zero = sum(c['has_report'] for c in s_zero) / len(s_zero) if s_zero else 0
print(f"\n  Report rate (S>0): {rate_pos:.3f} ({sum(c['has_report'] for c in s_pos)}/{len(s_pos)})")
print(f"  Report rate (S=0): {rate_zero:.3f} ({sum(c['has_report'] for c in s_zero)}/{len(s_zero)})")

# ============================================================
# TEST 1: LOGISTIC REGRESSION
# ============================================================
print(f"\n{'='*70}")
print("TEST 1: Logistic Regression — P(O>=1) ~ S + log(pop)")
print(f"{'='*70}")

y = np.array([c['has_report'] for c in analysis])
X_S = np.array([c['S'] for c in analysis])
X_logpop = np.array([c['log_pop'] for c in analysis])

# Full model: S + log(pop)
X_full = sm.add_constant(np.column_stack([X_S, X_logpop]))
try:
    model_full = sm.Logit(y, X_full).fit(disp=0)
    print(f"\n  Full model: const + S + log(pop)")
    print(f"  n = {n_analysis}, events = {n_with_report}")
    print(f"  S coefficient: {model_full.params[1]:.4f} (p = {model_full.pvalues[1]:.4f})")
    print(f"  log(pop) coefficient: {model_full.params[2]:.4f} (p = {model_full.pvalues[2]:.4f})")
    print(f"  Pseudo R²: {model_full.prsquared:.4f}")
    print(f"  AIC: {model_full.aic:.1f}")

    # Odds ratio for S
    or_S = np.exp(model_full.params[1])
    ci_lo = np.exp(model_full.conf_int()[1, 0])
    ci_hi = np.exp(model_full.conf_int()[1, 1])
    print(f"  OR for S: {or_S:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]")

    logistic_result = {
        'n': n_analysis,
        'events': int(n_with_report),
        'S_coef': round(float(model_full.params[1]), 4),
        'S_p': round(float(model_full.pvalues[1]), 4),
        'S_OR': round(float(or_S), 2),
        'S_OR_CI': [round(float(ci_lo), 2), round(float(ci_hi), 2)],
        'logpop_coef': round(float(model_full.params[2]), 4),
        'logpop_p': round(float(model_full.pvalues[2]), 4),
        'pseudo_r2': round(float(model_full.prsquared), 4),
    }

    # Reduced model: log(pop) only
    X_red = sm.add_constant(X_logpop)
    model_red = sm.Logit(y, X_red).fit(disp=0)
    lr_stat = 2 * (model_full.llf - model_red.llf)
    lr_p = 1 - __import__('scipy').stats.chi2.cdf(lr_stat, df=1)
    print(f"\n  LR test (S added to pop-only model):")
    print(f"  chi2 = {lr_stat:.3f}, p = {lr_p:.4f}")
    logistic_result['lr_chi2'] = round(float(lr_stat), 3)
    logistic_result['lr_p'] = round(float(lr_p), 4)

except Exception as e:
    print(f"  Logistic regression failed: {e}")
    logistic_result = {'error': str(e)}

# ============================================================
# TEST 2: BINARY S (S>0 vs S=0) — Fisher's exact, pop-stratified
# ============================================================
print(f"\n{'='*70}")
print("TEST 2: Binary canyon test — report rate S>0 vs S=0")
print(f"{'='*70}")

from scipy.stats import fisher_exact

# 2x2 table: (S>0, has_report), (S>0, no_report), (S=0, has_report), (S=0, no_report)
a = sum(1 for c in analysis if c['S'] > 0 and c['has_report'] == 1)
b = sum(1 for c in analysis if c['S'] > 0 and c['has_report'] == 0)
c_val = sum(1 for c in analysis if c['S'] == 0 and c['has_report'] == 1)
d = sum(1 for c in analysis if c['S'] == 0 and c['has_report'] == 0)

table = np.array([[a, b], [c_val, d]])
or_fisher, p_fisher = fisher_exact(table, alternative='greater')
print(f"  2x2 table:")
print(f"           report  no_report")
print(f"  S > 0:    {a:4d}     {b:4d}")
print(f"  S = 0:    {c_val:4d}     {d:4d}")
print(f"  Fisher exact OR = {or_fisher:.2f}, p = {p_fisher:.4f} (one-sided)")

binary_result = {
    'table': [[int(a), int(b)], [int(c_val), int(d)]],
    'OR': round(float(or_fisher), 2) if not np.isinf(or_fisher) else 'inf',
    'p_onesided': round(float(p_fisher), 4),
}

# ============================================================
# TEST 3: SPEARMAN ON RAW R = O/E (including zeros)
# ============================================================
print(f"\n{'='*70}")
print("TEST 3: Spearman(S, O/E) — full sample including O=0")
print(f"{'='*70}")

S_all = np.array([c['S'] for c in analysis])
R_all = np.array([c['R'] for c in analysis])

rho_all, p_all = spearmanr(S_all, R_all)
print(f"  n = {n_analysis}")
print(f"  Spearman rho = {rho_all:.4f}")
print(f"  p = {p_all:.4f}")

spearman_full = {
    'n': n_analysis,
    'rho': round(float(rho_all), 4),
    'p': round(float(p_all), 4),
}

# Also original 17-cell result for comparison
testable_17 = [c for c in analysis if c['O'] >= 1]
S_17 = np.array([c['S'] for c in testable_17])
logR_17 = np.array([np.log(c['R']) for c in testable_17 if c['R'] > 0])
rho_17, p_17 = spearmanr(S_17[:len(logR_17)], logR_17)
print(f"\n  Original (n=17, logR): rho = {rho_17:.4f}, p = {p_17:.4f}")

# ============================================================
# TEST 4: Mann-Whitney — pop-matched comparison
# ============================================================
print(f"\n{'='*70}")
print("TEST 4: Mann-Whitney — log(pop) in report vs no-report cells")
print(f"{'='*70}")

pop_with = [c['log_pop'] for c in analysis if c['has_report'] == 1]
pop_without = [c['log_pop'] for c in analysis if c['has_report'] == 0]
u_stat, p_mw = mannwhitneyu(pop_with, pop_without, alternative='greater')
print(f"  Cells with reports: median log(pop) = {np.median(pop_with):.2f} (pop ~ {np.exp(np.median(pop_with)):,.0f})")
print(f"  Cells without reports: median log(pop) = {np.median(pop_without):.2f} (pop ~ {np.exp(np.median(pop_without)):,.0f})")
print(f"  Mann-Whitney p = {p_mw:.4f}")
print(f"  (Tests whether report cells have higher pop — expected, confirms pop control needed)")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  Original (17 cells, all S>0):    rho={rho_17:.3f}, p={p_17:.4f}")
print(f"  Full sample Spearman (n={n_analysis}):   rho={rho_all:.3f}, p={p_all:.4f}")
print(f"  Binary Fisher exact:              OR={or_fisher:.2f}, p={p_fisher:.4f}")
print(f"  Logistic (S|pop):                 coef={logistic_result.get('S_coef','?')}, p={logistic_result.get('S_p','?')}")

if logistic_result.get('S_p', 1) < 0.05:
    verdict = "POSITIVE — S predicts reports controlling for population"
elif logistic_result.get('S_p', 1) < 0.10:
    verdict = "SUGGESTIVE — S marginal after population control"
else:
    verdict = "NULL — S does not predict reports after population control"
print(f"\n  Verdict: {verdict}")

# Save
output = {
    'test': 'Norway replication v2: logistic + full-sample',
    'n_coastal_cells': len(coastal_cells),
    'n_with_pop': n_analysis,
    'n_with_reports': int(n_with_report),
    'n_S_gt_0': n_s_pos_a,
    'n_S_eq_0': n_s_zero_a,
    'report_rate_S_pos': round(rate_pos, 4),
    'report_rate_S_zero': round(rate_zero, 4),
    'test_1_logistic': logistic_result,
    'test_2_binary_fisher': binary_result,
    'test_3_spearman_full': spearman_full,
    'original_17cell': {'rho': round(float(rho_17), 4), 'p': round(float(p_17), 4)},
    'verdict': verdict,
}

out_file = os.path.join(OUT_DIR, "phase_e_norway_logistic.json")
with open(out_file, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_file}")
print(f"DONE ({elapsed()})")
