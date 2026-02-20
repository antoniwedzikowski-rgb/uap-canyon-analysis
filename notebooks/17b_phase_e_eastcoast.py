#!/usr/bin/env python3
"""
Phase E: East Coast E-RED — same methodology as West Coast
============================================================
Run the identical rate ratio analysis (R_i = O_i / E_i) on East Coast
to answer: does S predict UAP excess there?

Uses the same frozen grid (phase-ev2-frozen) and haversine E_i.
"""

import os, time, json, warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, rankdata, mannwhitneyu
from numpy.linalg import lstsq

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
print("PHASE E: EAST COAST E-RED CHECK")
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
# RUN for East Coast (200km band)
# ============================================================
COASTAL_BAND = 200
EAST_COAST_LON_MIN = -82.0  # east of this is East Coast
EAST_COAST_LAT_MIN = 25.0
EAST_COAST_LAT_MAX = 45.0

print(f"\n{'='*70}")
print(f"EAST COAST (lon >= {EAST_COAST_LON_MIN}, lat {EAST_COAST_LAT_MIN}-{EAST_COAST_LAT_MAX})")
print(f"Coastal band: {COASTAL_BAND} km")
print(f"{'='*70}")

# Filter NUFORC to 200km coastal band
_, c_idx = coast_tree.query(np.column_stack([df_raw['latitude'].values,
                                              df_raw['longitude'].values]), k=1)
d_coast = haversine_km(df_raw['latitude'].values, df_raw['longitude'].values,
                        coast_lats_arr[c_idx], coast_lons_arr[c_idx])
df_band = df_raw[d_coast <= COASTAL_BAND].copy().reset_index(drop=True)

# East Coast filter
df_east = df_band[(df_band['longitude'] >= EAST_COAST_LON_MIN) &
                   (df_band['latitude'] >= EAST_COAST_LAT_MIN) &
                   (df_band['latitude'] <= EAST_COAST_LAT_MAX)].copy()
print(f"  NUFORC in East Coast 200km band: {len(df_east):,}")

# Count O_i per cell
half = GRID_DEG / 2
cell_data = []

for cell in grid_data:
    lat_c, lon_c, S = cell['lat'], cell['lon'], cell['S']
    if lon_c < EAST_COAST_LON_MIN or lat_c < EAST_COAST_LAT_MIN or lat_c > EAST_COAST_LAT_MAX:
        continue

    in_cell = ((df_east['latitude'].values >= lat_c - half) &
               (df_east['latitude'].values < lat_c + half) &
               (df_east['longitude'].values >= lon_c - half) &
               (df_east['longitude'].values < lon_c + half))
    O_i = int(in_cell.sum())

    cell_data.append({
        'lat': lat_c, 'lon': lon_c, 'S': S, 'O_i': O_i,
        'n_steep': cell['n_steep_cells'],
    })

print(f"  East Coast grid cells: {len(cell_data)}")
print(f"  S > 0 cells: {sum(1 for c in cell_data if c['S'] > 0)}")
print(f"  S = 0 cells: {sum(1 for c in cell_data if c['S'] == 0)}")

# Compute E_i
print(f"  Computing E_i (haversine)... ({elapsed()})")
for cd in cell_data:
    cd['E_i_raw'] = compute_expected(cd['lat'], cd['lon'], half, COASTAL_BAND)

total_O = sum(cd['O_i'] for cd in cell_data)
total_E_raw = sum(cd['E_i_raw'] for cd in cell_data)
scale = total_O / total_E_raw if total_E_raw > 0 else 0
for cd in cell_data:
    cd['E_i'] = cd['E_i_raw'] * scale

# Testable cells
testable = [cd for cd in cell_data if cd['O_i'] >= MIN_REPORTS]
for cd in testable:
    cd['R_i'] = cd['O_i'] / cd['E_i'] if cd['E_i'] > 0 else np.nan
    cd['logR'] = np.log(cd['R_i']) if cd['R_i'] > 0 else np.nan
testable = [cd for cd in testable if not np.isnan(cd['logR'])]

n_test = len(testable)
n_hot = sum(1 for cd in testable if cd['S'] > 0)
n_cold = sum(1 for cd in testable if cd['S'] == 0)
print(f"\n  Testable (N >= {MIN_REPORTS}): {n_test}")
print(f"    S > 0 (hot): {n_hot}")
print(f"    S = 0 (cold): {n_cold}")

if n_test >= 10:
    S_arr = np.array([cd['S'] for cd in testable])
    logR_arr = np.array([cd['logR'] for cd in testable])
    R_arr = np.array([cd['R_i'] for cd in testable])
    O_arr = np.array([cd['O_i'] for cd in testable])
    E_arr = np.array([cd['E_i'] for cd in testable])
    has_canyon = S_arr > 0

    # Spearman
    rho, p_val = spearmanr(S_arr, logR_arr)
    print(f"\n  Spearman(S, logR): rho = {rho:.3f}, p = {p_val:.4f} (n = {n_test})")

    # Bootstrap CI
    rng = np.random.RandomState(RNG_SEED)
    boot_rhos = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n_test, n_test, replace=True)
        if len(np.unique(S_arr[idx])) > 1:
            r, _ = spearmanr(S_arr[idx], logR_arr[idx])
            boot_rhos.append(r)
    print(f"  Bootstrap 95% CI: [{np.percentile(boot_rhos, 2.5):.3f}, {np.percentile(boot_rhos, 97.5):.3f}]")

    # Poisson proxy
    X = np.column_stack([np.ones(n_test), S_arr])
    beta, _, _, _ = lstsq(X, logR_arr, rcond=None)
    print(f"\n  OLS log(R) ~ S:")
    print(f"    α = {beta[0]:.4f}, β_S = {beta[1]:.4f}")
    print(f"    exp(β) = {np.exp(beta[1]):.2f}x per unit S")

    # Bootstrap β
    boot_betas = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n_test, n_test, replace=True)
        Xb = np.column_stack([np.ones(len(idx)), S_arr[idx]])
        try:
            b, _, _, _ = lstsq(Xb, logR_arr[idx], rcond=None)
            boot_betas.append(b[1])
        except:
            pass
    print(f"    β 95% CI: [{np.percentile(boot_betas, 2.5):.3f}, {np.percentile(boot_betas, 97.5):.3f}]")

    # Four-group comparison (canyon/no-canyon)
    print(f"\n  Rate comparison:")
    for label, mask in [("S > 0", has_canyon), ("S = 0", ~has_canyon)]:
        if mask.sum() > 0:
            agg = O_arr[mask].sum() / E_arr[mask].sum()
            print(f"    {label}: n={mask.sum()}, mean R={R_arr[mask].mean():.3f}, "
                  f"median R={np.median(R_arr[mask]):.3f}, ΣO/ΣE={agg:.3f}")

    # Mann-Whitney
    if has_canyon.sum() >= 3 and (~has_canyon).sum() >= 3:
        U, p_mw = mannwhitneyu(R_arr[has_canyon], R_arr[~has_canyon], alternative='greater')
        print(f"\n  Mann-Whitney (S>0 vs S=0): U={U:.0f}, p={p_mw:.4f}")

    # Detail table: S > 0 cells
    print(f"\n  East Coast S > 0 cells (sorted by S):")
    print(f"  {'lat':>6} {'lon':>7} {'S':>6} {'O_i':>5} {'E_i':>7} {'R_i':>6} {'logR':>7}")
    print(f"  {'-'*50}")
    for cd in sorted([c for c in testable if c['S'] > 0], key=lambda x: -x['S']):
        print(f"  {cd['lat']:6.1f} {cd['lon']:7.1f} {cd['S']:6.3f} {cd['O_i']:5d} "
              f"{cd['E_i']:7.1f} {cd['R_i']:6.2f} {cd['logR']:7.3f}")

    # Compare with West Coast results
    print(f"\n{'='*70}")
    print("COMPARISON: EAST COAST vs WEST COAST")
    print(f"{'='*70}")
    print(f"  West Coast (from E-RED v2):")
    print(f"    Spearman: rho = 0.374, p = 0.0001, n = 102")
    print(f"    Canyon uplift (Puget): 9.54×")
    print(f"    Canyon uplift (Other WC): 1.76×")
    print(f"\n  East Coast:")
    print(f"    Spearman: rho = {rho:.3f}, p = {p_val:.4f}, n = {n_test}")
    if has_canyon.sum() > 0 and (~has_canyon).sum() > 0:
        s1_rate = O_arr[has_canyon].sum() / E_arr[has_canyon].sum()
        s0_rate = O_arr[~has_canyon].sum() / E_arr[~has_canyon].sum()
        print(f"    Canyon uplift: {s1_rate/s0_rate:.2f}×")

else:
    print("  Too few testable cells for East Coast analysis.")

# Save
results = {
    "test": "East Coast E-RED check",
    "region": f"lon >= {EAST_COAST_LON_MIN}, lat {EAST_COAST_LAT_MIN}-{EAST_COAST_LAT_MAX}",
    "coastal_band_km": COASTAL_BAND,
    "n_testable": n_test,
    "n_hot": n_hot,
    "n_cold": n_cold,
}
if n_test >= 10:
    results["spearman_rho"] = round(float(rho), 4)
    results["spearman_p"] = round(float(p_val), 4)
    results["beta_S"] = round(float(beta[1]), 4)

out_file = os.path.join(OUT_DIR, "phase_e_eastcoast_check.json")
with open(out_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_file}")

print(f"\nDONE ({elapsed()})")
