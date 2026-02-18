#!/usr/bin/env python3
"""
Phase E-RED v2: Haversine-corrected evaluation
================================================
Fixes CRITICAL-3 from cross-phase audit:
  Old code used cKDTree(lat,lon) distance * 111 km/deg for coastal and
  county distance. This is Euclidean in degree-space, which overestimates
  E-W distances at high latitudes (33% at 48°N Puget Sound).

  Fix: use haversine for all distance computations in E_i.

Also runs secondary at BOTH 20km and 25km to answer "which is better?"

Scoring (S) frozen at phase-ev2-frozen (commit c2366d2, 60 m/km threshold).
"""

import os, time, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, rankdata
from numpy.linalg import lstsq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR = "/Users/antoniwedzikowski/Desktop/UAP research"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "phase_ev2")
os.makedirs(OUT_DIR, exist_ok=True)

R_EARTH = 6371.0
GRID_DEG = 0.5
WINSORIZE_PCT = 95
N_BOOTSTRAP = 2000
RNG_SEED = 42
MIN_REPORTS = 20

# West Coast definition
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


def haversine_from_tree_idx(query_lats, query_lons, ref_lats, ref_lons, tree_indices):
    """Compute haversine distances from query points to indexed reference points."""
    return haversine_km(query_lats, query_lons,
                        ref_lats[tree_indices], ref_lons[tree_indices])


# ============================================================
# LOAD FROZEN GRID
# ============================================================
print("=" * 70)
print("PHASE E-RED v2: HAVERSINE-CORRECTED EVALUATION")
print("Scoring frozen at phase-ev2-frozen (60 m/km)")
print("Fixes: CRITICAL-3 (latitude bias in E_i)")
print("=" * 70)

with open(os.path.join(OUT_DIR, "phase_ev2_grid.json")) as f:
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
coast_lats_arr = elev_lats[coast_i].astype(np.float64)
coast_lons_arr = elev_lons[coast_j].astype(np.float64)
coast_tree = cKDTree(np.column_stack([coast_lats_arr, coast_lons_arr]))

counties_df = pd.read_csv(os.path.join(DATA_DIR, "county_centroids_pop.csv"))
county_lats = counties_df['lat'].values.astype(np.float64)
county_lons = counties_df['lon'].values.astype(np.float64)
county_tree = cKDTree(np.column_stack([county_lats, county_lons]))
counties_pop = counties_df['pop'].values.astype(np.float64)

print(f"[LOAD] Geometry complete ({elapsed()})")


# ============================================================
# LOAD NUFORC
# ============================================================
print(f"\n[LOAD] NUFORC... ({elapsed()})")
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
print(f"  Total CONUS reports: {len(df_raw):,}")


# ============================================================
# FUNCTION: compute E_i (expected reports) — HAVERSINE CORRECTED
# ============================================================
def compute_expected(cell_lat, cell_lon, half_deg, coastal_band_km):
    """
    Population-weighted expected report density for a cell.

    HAVERSINE FIX (v2): uses haversine for all distance computations,
    replacing the old cd_grid * 111.0 approximation that systematically
    biased high-latitude cells (Puget: ~33% overestimate of E-W distance).
    """
    # Create grid of points within cell
    grid_lat = np.linspace(cell_lat - half_deg, cell_lat + half_deg, 30)
    grid_lon = np.linspace(cell_lon - half_deg, cell_lon + half_deg, 30)
    glat, glon = np.meshgrid(grid_lat, grid_lon, indexing='ij')
    gf, gln = glat.flatten(), glon.flatten()

    # Coastal filter — HAVERSINE
    # Use cKDTree for nearest-neighbor index, then haversine for true distance
    _, coast_idx = coast_tree.query(np.column_stack([gf, gln]), k=1)
    coast_dist_km = haversine_km(gf, gln,
                                  coast_lats_arr[coast_idx],
                                  coast_lons_arr[coast_idx])
    coastal_mask = coast_dist_km <= coastal_band_km
    gc_lat = gf[coastal_mask]
    gc_lon = gln[coastal_mask]

    if len(gc_lat) < 5:
        return 0.0

    # Population weighting — HAVERSINE
    n_k = min(10, len(counties_pop))
    _, ci_county = county_tree.query(np.column_stack([gc_lat, gc_lon]), k=n_k)

    weights = np.zeros(len(gc_lat))
    for k in range(n_k):
        d_km = haversine_km(gc_lat, gc_lon,
                            county_lats[ci_county[:, k]],
                            county_lons[ci_county[:, k]])
        d_km = d_km + 1.0  # avoid division by zero
        weights += counties_pop[ci_county[:, k]] / (d_km**2)

    # Land/ocean weighting
    lat_idx = np.clip(np.searchsorted(elev_lats, gc_lat), 0, len(elev_lats)-1)
    lon_idx = np.clip(np.searchsorted(elev_lons, gc_lon), 0, len(elev_lons)-1)
    ge = elevation[lat_idx, lon_idx]
    lw = np.where(ge >= 0, 3.0, 0.05)
    weights *= lw

    return float(weights.sum())


# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================
def run_evaluation(coastal_band_km, label):
    print(f"\n{'='*70}")
    print(f"E-RED v2: {label} (coastal band = {coastal_band_km} km)")
    print(f"{'='*70}")

    # Filter NUFORC to coastal band — already uses haversine
    _, c_idx = coast_tree.query(np.column_stack([df_raw['latitude'].values,
                                                  df_raw['longitude'].values]), k=1)
    d_coast = haversine_km(df_raw['latitude'].values, df_raw['longitude'].values,
                            coast_lats_arr[c_idx], coast_lons_arr[c_idx])
    df_band = df_raw[d_coast <= coastal_band_km].copy().reset_index(drop=True)
    print(f"  NUFORC in {coastal_band_km}km band: {len(df_band):,}")

    # West Coast filter
    df_west = df_band[(df_band['longitude'] <= WEST_COAST_LON_MAX) &
                       (df_band['latitude'] >= WEST_COAST_LAT_MIN)].copy()
    print(f"  West Coast: {len(df_west):,}")

    # Count O_i per cell (West Coast only)
    print(f"\n  Computing O_i per cell... ({elapsed()})")
    half = GRID_DEG / 2
    cell_data = []

    for cell in grid_data:
        lat_c, lon_c, S = cell['lat'], cell['lon'], cell['S']

        # West Coast filter
        if lon_c > WEST_COAST_LON_MAX or lat_c < WEST_COAST_LAT_MIN:
            continue

        # Count UAP in cell
        in_cell = ((df_west['latitude'].values >= lat_c - half) &
                   (df_west['latitude'].values < lat_c + half) &
                   (df_west['longitude'].values >= lon_c - half) &
                   (df_west['longitude'].values < lon_c + half))
        O_i = int(in_cell.sum())

        cell_data.append({
            'lat': lat_c, 'lon': lon_c, 'S': S, 'O_i': O_i,
            'n_steep': cell['n_steep_cells'],
        })

    print(f"  West Coast grid cells: {len(cell_data)}")

    # Compute E_i
    print(f"  Computing E_i (population-weighted, HAVERSINE)... ({elapsed()})")
    for cd in cell_data:
        cd['E_i_raw'] = compute_expected(cd['lat'], cd['lon'], half, coastal_band_km)

    # Normalize E_i so sum(E_i) = sum(O_i)
    total_O = sum(cd['O_i'] for cd in cell_data)
    total_E_raw = sum(cd['E_i_raw'] for cd in cell_data)
    if total_E_raw == 0:
        print("  ERROR: total expected = 0")
        return None

    scale = total_O / total_E_raw
    for cd in cell_data:
        cd['E_i'] = cd['E_i_raw'] * scale

    # Filter to testable cells (N >= MIN_REPORTS)
    testable = [cd for cd in cell_data if cd['O_i'] >= MIN_REPORTS]
    print(f"  Testable cells (N >= {MIN_REPORTS}): {len(testable)}")
    n_hot = sum(1 for cd in testable if cd['S'] > 0)
    n_cold = sum(1 for cd in testable if cd['S'] == 0)
    print(f"    S > 0 (hot): {n_hot}")
    print(f"    S = 0 (cold): {n_cold}")

    if len(testable) < 10:
        print("  Too few testable cells for analysis.")
        return {"label": label, "n_testable": len(testable), "verdict": "INSUFFICIENT_DATA"}

    # Compute R_i = O_i / E_i
    for cd in testable:
        cd['R_i'] = cd['O_i'] / cd['E_i'] if cd['E_i'] > 0 else np.nan
        cd['logR'] = np.log(cd['R_i']) if cd['R_i'] > 0 else np.nan

    # Remove NaN
    testable = [cd for cd in testable if not np.isnan(cd['logR'])]
    print(f"  Valid R_i: {len(testable)}")

    S_arr = np.array([cd['S'] for cd in testable])
    logR_arr = np.array([cd['logR'] for cd in testable])
    R_arr = np.array([cd['R_i'] for cd in testable])
    O_arr = np.array([cd['O_i'] for cd in testable])
    E_arr = np.array([cd['E_i'] for cd in testable])
    lat_arr = np.array([cd['lat'] for cd in testable])

    # ---- METRIC 1: Spearman ----
    rho, p_val = spearmanr(S_arr, logR_arr)
    print(f"\n  Spearman(S, log(R)): rho = {rho:.3f}, p = {p_val:.4f} (n = {len(testable)})")

    # Bootstrap CI on Spearman
    rng = np.random.RandomState(RNG_SEED)
    boot_rhos = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(testable), len(testable), replace=True)
        if len(np.unique(S_arr[idx])) > 1:
            r, _ = spearmanr(S_arr[idx], logR_arr[idx])
            boot_rhos.append(r)
    rho_ci_lo = np.percentile(boot_rhos, 2.5)
    rho_ci_hi = np.percentile(boot_rhos, 97.5)
    print(f"  Bootstrap 95% CI: [{rho_ci_lo:.3f}, {rho_ci_hi:.3f}]")

    # ---- METRIC 2: Precision@K ----
    K_values = [5, 10]
    R_threshold_80 = np.percentile(R_arr, 80)
    precision_results = {}
    for K in K_values:
        if K > len(testable):
            continue
        top_k_idx = np.argsort(-S_arr)[:K]
        n_in_top_R = sum(1 for i in top_k_idx if R_arr[i] >= R_threshold_80)
        prec = n_in_top_R / K
        precision_results[K] = {'precision': round(prec, 3), 'n_hit': n_in_top_R,
                                 'R_threshold_80': round(R_threshold_80, 3)}
        print(f"  Precision@{K}: {n_in_top_R}/{K} = {prec:.1%} "
              f"(top-20% R threshold = {R_threshold_80:.2f})")

    # ---- METRIC 3: Decile plot ----
    n_bins = min(5, len(testable) // 5)
    if n_bins >= 3:
        S_ranks = rankdata(S_arr)
        bin_edges = np.linspace(0, len(testable), n_bins + 1).astype(int)
        decile_means = []
        decile_cis = []
        decile_S_means = []

        for b in range(n_bins):
            idx_sorted = np.argsort(S_ranks)
            lo, hi = bin_edges[b], bin_edges[b+1]
            bin_idx = idx_sorted[lo:hi]
            bin_logR = logR_arr[bin_idx]
            bin_S = S_arr[bin_idx]

            mean_logR = np.mean(bin_logR)
            boots = []
            for _ in range(N_BOOTSTRAP):
                bi = rng.choice(len(bin_logR), len(bin_logR), replace=True)
                boots.append(np.mean(bin_logR[bi]))
            ci_lo = np.percentile(boots, 2.5)
            ci_hi = np.percentile(boots, 97.5)

            decile_means.append(mean_logR)
            decile_cis.append((ci_lo, ci_hi))
            decile_S_means.append(np.mean(bin_S))

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        x = range(1, n_bins + 1)
        ax.bar(x, decile_means, color='steelblue', alpha=0.7, edgecolor='k')
        for i in range(n_bins):
            ax.plot([x[i], x[i]], [decile_cis[i][0], decile_cis[i][1]],
                    color='black', linewidth=2)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel(f'S quintile (1=lowest, {n_bins}=highest)')
        ax.set_ylabel('Mean log(R)  [R = O_i / E_i]')
        ax.set_title(f'E-RED v2 {label}: Geometric Score vs Excess Report Rate\n'
                     f'Spearman ρ={rho:.3f} (p={p_val:.4f}), n={len(testable)}'
                     f'\n[Haversine-corrected E_i]')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Q{i+1}\n(S̄={decile_S_means[i]:.2f})' for i in range(n_bins)])
        plt.tight_layout()
        plot_file = os.path.join(OUT_DIR, f"e_red_v2_{label.replace(' ', '_').lower()}.png")
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"  Saved plot: {plot_file}")
    else:
        print(f"  Too few cells for decile plot (n_bins={n_bins})")
        decile_means = []
        decile_cis = []

    # ---- METRIC 4: Poisson proxy (OLS on log(R) ~ S) ----
    X = np.column_stack([np.ones(len(testable)), S_arr])
    beta, residuals, _, _ = lstsq(X, logR_arr, rcond=None)
    alpha_hat, beta_hat = beta[0], beta[1]
    print(f"\n  Poisson proxy (OLS on log(R) ~ S):")
    print(f"    α = {alpha_hat:.3f}, β = {beta_hat:.3f}")

    # Bootstrap β
    boot_betas = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(testable), len(testable), replace=True)
        Xb = np.column_stack([np.ones(len(idx)), S_arr[idx]])
        try:
            b, _, _, _ = lstsq(Xb, logR_arr[idx], rcond=None)
            boot_betas.append(b[1])
        except:
            pass
    beta_ci_lo = np.percentile(boot_betas, 2.5)
    beta_ci_hi = np.percentile(boot_betas, 97.5)
    print(f"    β 95% CI: [{beta_ci_lo:.3f}, {beta_ci_hi:.3f}]")
    print(f"    Interpretation: 1 unit increase in S → exp(β) = {np.exp(beta_hat):.2f}x rate ratio")

    # ---- DETAIL TABLE ----
    print(f"\n  {'lat':>6} {'lon':>7} {'S':>6} {'O_i':>5} {'E_i':>7} {'R_i':>6} {'logR':>7} {'n_steep':>7}")
    print(f"  {'-'*55}")
    for cd in sorted(testable, key=lambda x: -x['S']):
        print(f"  {cd['lat']:6.1f} {cd['lon']:7.1f} {cd['S']:6.3f} {cd['O_i']:5d} "
              f"{cd['E_i']:7.1f} {cd['R_i']:6.2f} {cd['logR']:7.3f} {cd['n_steep']:7d}")

    # ---- Spearman without Puget ----
    puget_mask = ~((S_arr > 1.8) & (lat_arr > 46) & (lat_arr < 50))
    if puget_mask.sum() > 5:
        rho_nopuget, p_nopuget = spearmanr(S_arr[puget_mask], logR_arr[puget_mask])
        print(f"\n  Spearman WITHOUT Puget cluster: rho = {rho_nopuget:.3f}, "
              f"p = {p_nopuget:.4f} (n = {puget_mask.sum()})")
    else:
        rho_nopuget, p_nopuget = None, None

    return {
        "label": label,
        "coastal_band_km": coastal_band_km,
        "n_west_coast_cells": len(cell_data),
        "n_testable": len(testable),
        "n_hot": n_hot,
        "n_cold": n_cold,
        "spearman": {
            "rho": round(rho, 4),
            "p": round(p_val, 4),
            "ci_lo": round(rho_ci_lo, 4),
            "ci_hi": round(rho_ci_hi, 4),
            "n": len(testable),
        },
        "spearman_no_puget": {
            "rho": round(rho_nopuget, 4) if rho_nopuget is not None else None,
            "p": round(p_nopuget, 4) if p_nopuget is not None else None,
        },
        "precision": precision_results,
        "poisson_proxy": {
            "alpha": round(alpha_hat, 4),
            "beta": round(beta_hat, 4),
            "beta_ci_lo": round(beta_ci_lo, 4),
            "beta_ci_hi": round(beta_ci_hi, 4),
            "exp_beta": round(np.exp(beta_hat), 4),
        },
        "decile_means": [round(d, 4) for d in decile_means],
        "decile_cis": [(round(c[0], 4), round(c[1], 4)) for c in decile_cis],
        "cell_details": [
            {"lat": cd['lat'], "lon": cd['lon'], "S": round(cd['S'], 4),
             "O_i": cd['O_i'], "E_i": round(cd['E_i'], 2),
             "R_i": round(cd['R_i'], 4), "logR": round(cd['logR'], 4),
             "n_steep": cd['n_steep']}
            for cd in sorted(testable, key=lambda x: -x['S'])
        ],
    }


# ============================================================
# RUN ALL THREE PASSES
# ============================================================
res_primary = run_evaluation(200, "primary 200km")
res_sec_20  = run_evaluation(20, "secondary 20km")
res_sec_25  = run_evaluation(25, "secondary 25km")


# ============================================================
# COMPARISON: v1 vs v2 (for Puget cells)
# ============================================================
print(f"\n{'='*70}")
print("COMPARISON: degree*111 (v1) vs haversine (v2)")
print(f"{'='*70}")

# Find Puget cells in primary results to show the difference
if res_primary and 'cell_details' in res_primary:
    puget_cells = [c for c in res_primary['cell_details']
                   if c['lat'] > 46 and c['lat'] < 50 and c['S'] > 0]
    if puget_cells:
        print(f"\n  Puget Sound cells (lat 46-50, S > 0):")
        print(f"  {'lat':>6} {'lon':>7} {'S':>6} {'O_i':>5} {'E_i':>7} {'R_i':>6}")
        for c in puget_cells:
            print(f"  {c['lat']:6.1f} {c['lon']:7.1f} {c['S']:6.3f} {c['O_i']:5d} {c['E_i']:7.1f} {c['R_i']:6.2f}")


# ============================================================
# 20km vs 25km COMPARISON
# ============================================================
print(f"\n{'='*70}")
print("SECONDARY BAND COMPARISON: 20km vs 25km")
print(f"{'='*70}")

for tag, res in [("20km", res_sec_20), ("25km", res_sec_25)]:
    if res is None or res.get('n_testable', 0) < 10:
        print(f"\n  {tag}: insufficient data (n={res.get('n_testable', 0) if res else 0})")
        continue
    sp = res['spearman']
    snp = res['spearman_no_puget']
    poi = res['poisson_proxy']
    print(f"\n  SECONDARY {tag}:")
    print(f"    Testable cells: {res['n_testable']} (hot={res['n_hot']}, cold={res['n_cold']})")
    print(f"    Spearman(S, logR): rho={sp['rho']:.3f} [{sp['ci_lo']:.3f}, {sp['ci_hi']:.3f}] p={sp['p']:.4f}")
    if snp['rho'] is not None:
        print(f"    Spearman no Puget: rho={snp['rho']:.3f} p={snp['p']:.4f}")
    for K, pr in res['precision'].items():
        print(f"    Precision@{K}: {pr['precision']:.1%}")
    print(f"    β(S): {poi['beta']:.3f} [{poi['beta_ci_lo']:.3f}, {poi['beta_ci_hi']:.3f}]"
          f"  exp(β)={poi['exp_beta']:.2f}x")


# ============================================================
# COMBINED SUMMARY
# ============================================================
print(f"\n{'='*70}")
print("COMBINED SUMMARY: E-RED v2 (haversine-corrected)")
print(f"{'='*70}")

for tag, res in [("PRIMARY (200km)", res_primary),
                 ("SECONDARY (20km)", res_sec_20),
                 ("SECONDARY (25km)", res_sec_25)]:
    if res is None or res.get('n_testable', 0) < 10:
        print(f"\n  {tag}: insufficient data")
        continue
    sp = res['spearman']
    snp = res['spearman_no_puget']
    poi = res['poisson_proxy']
    print(f"\n  {tag}:")
    print(f"    Testable cells: {res['n_testable']} (hot={res['n_hot']}, cold={res['n_cold']})")
    print(f"    Spearman(S, logR): rho={sp['rho']:.3f} [{sp['ci_lo']:.3f}, {sp['ci_hi']:.3f}] p={sp['p']:.4f}")
    if snp['rho'] is not None:
        print(f"    Spearman no Puget: rho={snp['rho']:.3f} p={snp['p']:.4f}")
    for K, pr in res['precision'].items():
        print(f"    Precision@{K}: {pr['precision']:.1%}")
    print(f"    β(S): {poi['beta']:.3f} [{poi['beta_ci_lo']:.3f}, {poi['beta_ci_hi']:.3f}]"
          f"  exp(β)={poi['exp_beta']:.2f}x")


# ============================================================
# SAVE
# ============================================================
combined = {
    "metadata": {
        "script": "phase_e_red_v2.py",
        "timestamp": datetime.now().isoformat(),
        "frozen_tag": "phase-ev2-frozen",
        "frozen_commit": "c2366d2",
        "threshold_mkm": 60,
        "west_coast_def": f"lon <= {WEST_COAST_LON_MAX}, lat >= {WEST_COAST_LAT_MIN}",
        "min_reports": MIN_REPORTS,
        "n_bootstrap": N_BOOTSTRAP,
        "fix": "CRITICAL-3: haversine replaces degree*111 for coastal and county distances in E_i",
        "description": "Phase E-RED v2: haversine-corrected E_i. Also compares 20km vs 25km secondary.",
    },
    "primary_200km": res_primary,
    "secondary_20km": res_sec_20,
    "secondary_25km": res_sec_25,
}

out_file = os.path.join(OUT_DIR, "phase_e_red_v2_evaluation.json")
with open(out_file, 'w') as f:
    json.dump(combined, f, indent=2, default=str)
print(f"\n  Saved: {out_file}")

import shutil
repo_out = os.path.join(BASE_DIR, "uap-canyon-analysis", "results", "phase_ev2")
os.makedirs(repo_out, exist_ok=True)
shutil.copy2(out_file, repo_out)
for png in [f for f in os.listdir(OUT_DIR) if f.startswith('e_red_v2_') and f.endswith('.png')]:
    shutil.copy2(os.path.join(OUT_DIR, png), repo_out)
print(f"  Copied to repo")

print(f"\n{'='*70}")
print(f"DONE ({elapsed()})")
print(f"{'='*70}")
