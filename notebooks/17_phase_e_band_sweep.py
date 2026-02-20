#!/usr/bin/env python3
"""
Phase E: Coastal Band Sensitivity Sweep
=========================================
Same grid, same S, same E_i methodology — vary only the coastal band width.
Bands: 10, 25, 50, 100, 200 km
Regions: West Coast and East Coast separately.

Answers: "Is the effect nearshore or just coastal-region?"
"""

import os, time, json, warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from numpy.linalg import lstsq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR = os.environ.get("UAP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "phase_ev2")

R_EARTH = 6371.0
GRID_DEG = 0.5
N_BOOTSTRAP = 2000
RNG_SEED = 42
MIN_REPORTS = 20

BANDS = [10, 25, 50, 100, 200]

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
print("PHASE E: COASTAL BAND SENSITIVITY SWEEP")
print(f"Bands: {BANDS} km")
print("=" * 70)

with open(os.path.join(OUT_DIR, "phase_ev2_grid.json")) as f:
    grid_data = json.load(f)

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

print(f"[LOAD] NUFORC... ({elapsed()})")
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
print(f"  CONUS reports: {len(df_raw):,}")

# Precompute coast distances for all NUFORC reports
print(f"  Precomputing coast distances... ({elapsed()})")
_, all_c_idx = coast_tree.query(np.column_stack([df_raw['latitude'].values,
                                                  df_raw['longitude'].values]), k=1)
all_d_coast = haversine_km(df_raw['latitude'].values, df_raw['longitude'].values,
                            coast_lats_arr[all_c_idx], coast_lons_arr[all_c_idx])
print(f"  Done ({elapsed()})")


# ============================================================
# E_i function (haversine)
# ============================================================
def compute_expected(cell_lat, cell_lon, half_deg, coastal_band_km):
    grid_lat = np.linspace(cell_lat - half_deg, cell_lat + half_deg, 30)
    grid_lon = np.linspace(cell_lon - half_deg, cell_lon + half_deg, 30)
    glat, glon = np.meshgrid(grid_lat, grid_lon, indexing='ij')
    gf, gln = glat.flatten(), glon.flatten()

    _, coast_idx = coast_tree.query(np.column_stack([gf, gln]), k=1)
    coast_dist_km = haversine_km(gf, gln, coast_lats_arr[coast_idx], coast_lons_arr[coast_idx])
    coastal_mask = coast_dist_km <= coastal_band_km
    gc_lat = gf[coastal_mask]
    gc_lon = gln[coastal_mask]

    if len(gc_lat) < 5:
        return 0.0

    n_k = min(10, len(counties_pop))
    _, ci_county = county_tree.query(np.column_stack([gc_lat, gc_lon]), k=n_k)
    weights = np.zeros(len(gc_lat))
    for k in range(n_k):
        d_km = haversine_km(gc_lat, gc_lon, county_lats[ci_county[:, k]], county_lons[ci_county[:, k]])
        d_km = d_km + 1.0
        weights += counties_pop[ci_county[:, k]] / (d_km**2)

    lat_idx = np.clip(np.searchsorted(elev_lats, gc_lat), 0, len(elev_lats)-1)
    lon_idx = np.clip(np.searchsorted(elev_lons, gc_lon), 0, len(elev_lons)-1)
    ge = elevation[lat_idx, lon_idx]
    lw = np.where(ge >= 0, 3.0, 0.05)
    weights *= lw

    return float(weights.sum())


# ============================================================
# SWEEP FUNCTION
# ============================================================
def sweep_region(region_name, lon_range, lat_range):
    """Run E-RED at each band width for a given region."""
    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range

    results = []
    for band_km in BANDS:
        print(f"\n  --- {region_name} @ {band_km} km ---")

        # Filter NUFORC to band
        df_band = df_raw[all_d_coast <= band_km].copy()

        # Region filter
        df_region = df_band[(df_band['longitude'] >= lon_min) &
                            (df_band['longitude'] <= lon_max) &
                            (df_band['latitude'] >= lat_min) &
                            (df_band['latitude'] <= lat_max)].copy()
        print(f"    NUFORC in band: {len(df_region):,}")

        # Count O_i
        half = GRID_DEG / 2
        cell_data = []
        for cell in grid_data:
            lat_c, lon_c, S = cell['lat'], cell['lon'], cell['S']
            if lon_c < lon_min or lon_c > lon_max or lat_c < lat_min or lat_c > lat_max:
                continue

            in_cell = ((df_region['latitude'].values >= lat_c - half) &
                       (df_region['latitude'].values < lat_c + half) &
                       (df_region['longitude'].values >= lon_c - half) &
                       (df_region['longitude'].values < lon_c + half))
            O_i = int(in_cell.sum())
            cell_data.append({
                'lat': lat_c, 'lon': lon_c, 'S': S, 'O_i': O_i,
                'n_steep': cell['n_steep_cells'],
            })

        # E_i
        for cd in cell_data:
            cd['E_i_raw'] = compute_expected(cd['lat'], cd['lon'], half, band_km)
        total_O = sum(cd['O_i'] for cd in cell_data)
        total_E_raw = sum(cd['E_i_raw'] for cd in cell_data)
        if total_E_raw == 0:
            results.append({'band_km': band_km, 'n_testable': 0, 'rho': None, 'p': None})
            continue
        scale = total_O / total_E_raw
        for cd in cell_data:
            cd['E_i'] = cd['E_i_raw'] * scale

        # Testable
        testable = [cd for cd in cell_data if cd['O_i'] >= MIN_REPORTS]
        for cd in testable:
            cd['R_i'] = cd['O_i'] / cd['E_i'] if cd['E_i'] > 0 else np.nan
            cd['logR'] = np.log(cd['R_i']) if cd['R_i'] > 0 else np.nan
        testable = [cd for cd in testable if not np.isnan(cd['logR'])]

        n_test = len(testable)
        n_hot = sum(1 for cd in testable if cd['S'] > 0)
        n_cold = n_test - n_hot

        if n_test < 10 or n_hot < 2:
            print(f"    Testable: {n_test} (hot={n_hot}) — too few")
            results.append({
                'band_km': band_km, 'n_testable': n_test, 'n_hot': n_hot,
                'n_cold': n_cold, 'rho': None, 'p': None,
            })
            continue

        S_arr = np.array([cd['S'] for cd in testable])
        logR_arr = np.array([cd['logR'] for cd in testable])
        R_arr = np.array([cd['R_i'] for cd in testable])
        O_arr_t = np.array([cd['O_i'] for cd in testable])
        E_arr_t = np.array([cd['E_i'] for cd in testable])

        rho, p_val = spearmanr(S_arr, logR_arr)

        # Bootstrap
        rng = np.random.RandomState(RNG_SEED)
        boot_rhos = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(n_test, n_test, replace=True)
            if len(np.unique(S_arr[idx])) > 1:
                r, _ = spearmanr(S_arr[idx], logR_arr[idx])
                boot_rhos.append(r)
        ci_lo = np.percentile(boot_rhos, 2.5) if boot_rhos else None
        ci_hi = np.percentile(boot_rhos, 97.5) if boot_rhos else None

        # β
        X = np.column_stack([np.ones(n_test), S_arr])
        beta, _, _, _ = lstsq(X, logR_arr, rcond=None)

        # Canyon uplift
        has_c = S_arr > 0
        if has_c.sum() > 0 and (~has_c).sum() > 0:
            rate_s1 = O_arr_t[has_c].sum() / E_arr_t[has_c].sum()
            rate_s0 = O_arr_t[~has_c].sum() / E_arr_t[~has_c].sum()
            uplift = rate_s1 / rate_s0 if rate_s0 > 0 else None
        else:
            rate_s1 = rate_s0 = uplift = None

        print(f"    Testable: {n_test} (hot={n_hot}, cold={n_cold})")
        print(f"    Spearman: rho={rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] p={p_val:.4f}")
        print(f"    β_S = {beta[1]:.3f}, exp(β) = {np.exp(beta[1]):.2f}x")
        if uplift:
            print(f"    Canyon uplift: {uplift:.2f}×")

        results.append({
            'band_km': band_km,
            'n_testable': n_test,
            'n_hot': n_hot,
            'n_cold': n_cold,
            'rho': round(float(rho), 4),
            'p': round(float(p_val), 4),
            'ci_lo': round(float(ci_lo), 4) if ci_lo else None,
            'ci_hi': round(float(ci_hi), 4) if ci_hi else None,
            'beta_S': round(float(beta[1]), 4),
            'exp_beta': round(float(np.exp(beta[1])), 4),
            'canyon_uplift': round(float(uplift), 4) if uplift else None,
        })

    return results


# ============================================================
# RUN
# ============================================================
print(f"\n{'='*70}")
print("WEST COAST SWEEP")
print(f"{'='*70}")
wc_results = sweep_region("West Coast", (-135, -115), (30, 50))

print(f"\n{'='*70}")
print("EAST COAST SWEEP")
print(f"{'='*70}")
ec_results = sweep_region("East Coast", (-82, -65), (25, 45))


# ============================================================
# SUMMARY TABLE
# ============================================================
print(f"\n{'='*70}")
print("SUMMARY: COASTAL BAND SENSITIVITY")
print(f"{'='*70}")

print(f"\n  {'Band':>6} │ {'West Coast':^42} │ {'East Coast':^42}")
print(f"  {'(km)':>6} │ {'n':>4} {'n_hot':>5} {'rho':>6} {'p':>7} {'uplift':>7} │ {'n':>4} {'n_hot':>5} {'rho':>6} {'p':>7} {'uplift':>7}")
print(f"  {'─'*6}─┼{'─'*42}─┼{'─'*42}")

for wc, ec in zip(wc_results, ec_results):
    band = wc['band_km']

    # West Coast
    if wc['rho'] is not None:
        wc_str = f"{wc['n_testable']:4d} {wc['n_hot']:5d} {wc['rho']:+6.3f} {wc['p']:7.4f} {wc['canyon_uplift'] or 0:7.2f}×"
    else:
        wc_str = f"{wc['n_testable']:4d} {wc.get('n_hot', 0):5d} {'—':>6} {'—':>7} {'—':>7}"

    # East Coast
    if ec['rho'] is not None:
        ec_str = f"{ec['n_testable']:4d} {ec['n_hot']:5d} {ec['rho']:+6.3f} {ec['p']:7.4f} {ec['canyon_uplift'] or 0:7.2f}×"
    else:
        ec_str = f"{ec['n_testable']:4d} {ec.get('n_hot', 0):5d} {'—':>6} {'—':>7} {'—':>7}"

    print(f"  {band:6d} │ {wc_str} │ {ec_str}")


# ============================================================
# PLOT
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, results, title in [(axes[0], wc_results, 'West Coast'),
                             (axes[1], ec_results, 'East Coast')]:
    bands_plot = [r['band_km'] for r in results if r['rho'] is not None]
    rhos_plot = [r['rho'] for r in results if r['rho'] is not None]
    ci_lo_plot = [r['ci_lo'] for r in results if r['rho'] is not None]
    ci_hi_plot = [r['ci_hi'] for r in results if r['rho'] is not None]
    pvals_plot = [r['p'] for r in results if r['rho'] is not None]

    if bands_plot:
        colors = ['green' if p < 0.05 else 'gray' for p in pvals_plot]
        ax.bar(range(len(bands_plot)), rhos_plot, color=colors, alpha=0.7, edgecolor='k')
        for i in range(len(bands_plot)):
            if ci_lo_plot[i] is not None:
                ax.plot([i, i], [ci_lo_plot[i], ci_hi_plot[i]], color='black', linewidth=2)
        ax.set_xticks(range(len(bands_plot)))
        ax.set_xticklabels([f"{b} km\n(n={r['n_testable']})"
                            for b, r in zip(bands_plot, [r for r in results if r['rho'] is not None])])

    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Spearman ρ(S, log R)')
    ax.set_title(f'{title}\nGreen = p < 0.05')
    ax.set_ylim(-0.2, 0.6)

plt.tight_layout()
plot_file = os.path.join(OUT_DIR, "e_red_band_sweep.png")
plt.savefig(plot_file, dpi=150)
plt.close()
print(f"\nSaved plot: {plot_file}")


# ============================================================
# SAVE
# ============================================================
combined = {
    "test": "Coastal band sensitivity sweep",
    "bands_km": BANDS,
    "west_coast": wc_results,
    "east_coast": ec_results,
}
out_file = os.path.join(OUT_DIR, "phase_e_band_sweep.json")
with open(out_file, 'w') as f:
    json.dump(combined, f, indent=2)
print(f"Saved: {out_file}")

import shutil
repo_out = os.path.join(BASE_DIR, "uap-canyon-analysis", "results", "phase_ev2")
os.makedirs(repo_out, exist_ok=True)
shutil.copy2(out_file, repo_out)
shutil.copy2(plot_file, repo_out)
print(f"Copied to repo")

print(f"\nDONE ({elapsed()})")
