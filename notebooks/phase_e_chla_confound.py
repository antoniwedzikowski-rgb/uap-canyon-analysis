#!/usr/bin/env python3
"""
Phase E: MODIS Chlorophyll-a Upwelling Confound Test
=====================================================
Tests whether the UAP-canyon correlation is explained by coastal
upwelling (measured via satellite chlorophyll-a concentration).

Hypothesis: Canyons amplify coastal upwelling (California Current) ->
  fog, thermal inversions, bioluminescence, fata morgana -> UAP reports.
  If Chl-a (direct upwelling proxy) explains logR better than canyon
  steepness (S), the geophysical mechanism is supported.

Data: MODIS Aqua L3 mapped chlorophyll-a, monthly 4km composites,
      climatological mean 2003-2020, from NASA ERDDAP (CoastWatch).

Tests:
  1. Correlations: Chl-a metrics vs logR and vs S
  2. R-squared comparison: S vs Chl-a as predictor of logR
  3. Combined model + nested F-tests: does S survive controlling for Chl-a?
  4. Kitchen sink: S + Chl-a + depth metrics
  5. Mann-Whitney: canyon (S>0) vs non-canyon cells on Chl-a
  6. Collinearity check: S vs Chl-a, Chl-a vs depth
"""

import os, time, json, warnings
import numpy as np
import xarray as xr
from scipy.stats import spearmanr, mannwhitneyu
from scipy.stats import f as f_dist
from numpy.linalg import lstsq
import netCDF4 as nc

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR = os.environ.get("UAP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "phase_ev2")

GRID_DEG = 0.5
CHLA_FILE = os.path.join(DATA_DIR, "modis_chla_westcoast.nc")

# ERDDAP dataset for MODIS Aqua L3 monthly 4km chlorophyll-a
ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
ERDDAP_DATASET = "erdMH1chlamday"

def elapsed():
    return f"{time.time()-t0:.1f}s"


print("=" * 70)
print("PHASE E: MODIS CHLOROPHYLL-a UPWELLING CONFOUND TEST")
print("Chl-a (direct upwelling proxy) vs canyon steepness (S)")
print("=" * 70)


# ============================================================
# DOWNLOAD MODIS Chl-a (if not cached)
# ============================================================
if not os.path.exists(CHLA_FILE):
    print(f"\n[DOWNLOAD] MODIS Aqua Chl-a from ERDDAP... ({elapsed()})")
    print(f"  Dataset: {ERDDAP_DATASET}")
    print(f"  Region: 30-50N, 130-115W")
    print(f"  Period: 2003-01 to 2020-12 (climatological mean)")

    downloaded = False

    # Approach 1: OPeNDAP via xarray (most robust)
    try:
        print(f"  Trying OPeNDAP...")
        ds_remote = xr.open_dataset(
            f"{ERDDAP_BASE}/{ERDDAP_DATASET}",
            engine='netcdf4'
        )

        # Detect coordinate names
        if 'latitude' in ds_remote.coords:
            lat_dim, lon_dim = 'latitude', 'longitude'
        else:
            lat_dim, lon_dim = 'lat', 'lon'

        # Latitude may be descending (90 -> -90), so detect order
        lats_remote = ds_remote.coords[lat_dim].values
        if lats_remote[0] > lats_remote[-1]:
            lat_slice = slice(50.0, 30.0)   # descending
        else:
            lat_slice = slice(30.0, 50.0)   # ascending

        chla_subset = ds_remote['chlorophyll'].sel(
            time=slice('2003-01-01', '2020-12-31'),
            **{lat_dim: lat_slice, lon_dim: slice(-130.0, -115.0)}
        )
        print(f"  Loaded {chla_subset.sizes.get('time', 0)} monthly composites")
        print(f"  Spatial shape: {dict((d, chla_subset.sizes[d]) for d in chla_subset.dims if d != 'time')}")
        print(f"  Computing climatological mean...")
        chla_mean = chla_subset.mean(dim='time', skipna=True)

        # Save as Dataset (not raw DataArray) to preserve coords
        chla_mean.name = 'chlorophyll'
        chla_mean.to_dataset().to_netcdf(CHLA_FILE)
        ds_remote.close()
        print(f"  Saved climatological mean: {CHLA_FILE}")
        downloaded = True

    except Exception as e:
        print(f"  OPeNDAP failed: {e}")

    # Approach 2: Direct HTTP download
    if not downloaded:
        try:
            print(f"  Trying direct HTTP download...")
            import requests

            url = (
                f"{ERDDAP_BASE}/{ERDDAP_DATASET}.nc?"
                "chlorophyll"
                "[(2003-01-16T00:00:00Z):1:(2020-12-16T00:00:00Z)]"
                "[(30.0):1:(50.0)]"
                "[(-130.0):1:(-115.0)]"
            )
            resp = requests.get(url, timeout=600)
            resp.raise_for_status()

            # Save raw download, then compute mean
            raw_file = CHLA_FILE + ".raw"
            with open(raw_file, 'wb') as f:
                f.write(resp.content)
            print(f"  Downloaded {len(resp.content)/1e6:.1f} MB")

            # Compute temporal mean and save
            ds_raw = xr.open_dataset(raw_file)
            var_name = [v for v in ds_raw.data_vars if 'chlor' in v.lower()][0]
            chla_mean = ds_raw[var_name].mean(dim='time', skipna=True)
            chla_mean.to_netcdf(CHLA_FILE)
            ds_raw.close()
            os.remove(raw_file)
            print(f"  Saved climatological mean: {CHLA_FILE}")
            downloaded = True

        except Exception as e:
            print(f"  HTTP download failed: {e}")

    if not downloaded:
        print("\n  ERROR: Could not download MODIS Chl-a data.")
        print("  Please download manually from:")
        print(f"    {ERDDAP_BASE}/{ERDDAP_DATASET}.html")
        print(f"  Save as: {CHLA_FILE}")
        raise SystemExit(1)

else:
    print(f"\n[CACHED] {CHLA_FILE}")


# ============================================================
# LOAD MODIS Chl-a
# ============================================================
print(f"\n[LOAD] MODIS Chl-a climatological mean... ({elapsed()})")

ds_chla = xr.open_dataset(CHLA_FILE)

# Find the chlorophyll variable (name varies by download method)
chla_var = None
for vname in ds_chla.data_vars:
    if 'chlor' in vname.lower():
        chla_var = vname
        break
if chla_var is None:
    chla_var = list(ds_chla.data_vars)[0]

chla_data = ds_chla[chla_var]

# If time dimension survived, collapse it
if 'time' in chla_data.dims:
    print(f"  Collapsing {chla_data.sizes['time']} time steps to mean...")
    chla_data = chla_data.mean(dim='time', skipna=True)

# Detect coordinate names
lat_name = 'latitude' if 'latitude' in chla_data.coords else 'lat'
lon_name = 'longitude' if 'longitude' in chla_data.coords else 'lon'

chla_lats = chla_data.coords[lat_name].values
chla_lons = chla_data.coords[lon_name].values
chla_grid = chla_data.values  # 2D array (lat, lon)

ds_chla.close()

print(f"  Variable: {chla_var}")
print(f"  Shape: {chla_grid.shape}")
print(f"  Lat range: {chla_lats.min():.2f} to {chla_lats.max():.2f}")
print(f"  Lon range: {chla_lons.min():.2f} to {chla_lons.max():.2f}")

valid_pix = chla_grid[np.isfinite(chla_grid) & (chla_grid > 0)]
print(f"  Valid ocean pixels: {len(valid_pix):,} / {chla_grid.size:,}")
print(f"  Chl-a range: {valid_pix.min():.3f} to {valid_pix.max():.1f} mg/m3")
print(f"  Chl-a median: {np.median(valid_pix):.3f} mg/m3")


# ============================================================
# LOAD E-RED v2 CELLS
# ============================================================
print(f"\n[LOAD] E-RED v2 cell data... ({elapsed()})")

with open(os.path.join(OUT_DIR, "phase_e_red_v2_evaluation.json")) as f:
    data = json.load(f)
cells = data["primary_200km"]["cell_details"]
print(f"  {len(cells)} testable West Coast cells")


# ============================================================
# BUILD OCEAN Chl-a LOOKUP (cKDTree for nearest-ocean queries)
# ============================================================
print(f"\n[BUILD] Ocean Chl-a spatial index... ({elapsed()})")

R_EARTH = 6371.0
SEARCH_RADIUS_KM = 200.0  # match the study's 200km coastal band definition

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.asarray, [lat1, lon1, lat2, lon2])
    r = np.radians
    dlat = r(lat2 - lat1); dlon = r(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(r(lat1))*np.cos(r(lat2))*np.sin(dlon/2)**2
    return R_EARTH * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# Extract all valid ocean Chl-a pixels with their coordinates
chla_lat_grid, chla_lon_grid = np.meshgrid(chla_lats, chla_lons, indexing='ij')
ocean_mask = np.isfinite(chla_grid) & (chla_grid > 0)
ocean_lat = chla_lat_grid[ocean_mask]
ocean_lon = chla_lon_grid[ocean_mask]
ocean_chla = chla_grid[ocean_mask]

print(f"  {len(ocean_chla):,} valid ocean pixels")

# Build cKDTree on ocean pixels (degree-space for fast neighbor lookup)
from scipy.spatial import cKDTree
ocean_tree = cKDTree(np.column_stack([ocean_lat, ocean_lon]))

# Degree search radius (approximate, refined with haversine below)
deg_search = SEARCH_RADIUS_KM / 80.0  # ~0.625 deg at mid-latitudes

# ============================================================
# COMPUTE Chl-a METRICS PER 0.5-deg CELL
# ============================================================
print(f"\n[COMPUTE] Chl-a metrics per cell (nearest ocean within {SEARCH_RADIUS_KM} km)... ({elapsed()})")

half = GRID_DEG / 2

for cell in cells:
    clat, clon = cell['lat'], cell['lon']

    # Find ocean pixels within degree-space bounding box (fast pre-filter)
    candidates = ocean_tree.query_ball_point([clat, clon], deg_search)

    if len(candidates) > 0:
        c_lats = ocean_lat[candidates]
        c_lons = ocean_lon[candidates]
        c_vals = ocean_chla[candidates]

        # Refine with haversine distance
        dists = haversine_km(clat, clon, c_lats, c_lons)
        within = dists <= SEARCH_RADIUS_KM
        valid = c_vals[within]
    else:
        valid = np.array([])

    if len(valid) >= 3:
        cell['chla_mean'] = float(np.mean(valid))
        cell['chla_median'] = float(np.median(valid))
        cell['chla_log_mean'] = float(np.mean(np.log(valid)))
        cell['chla_std'] = float(np.std(valid))
        cell['chla_max'] = float(np.max(valid))
        cell['n_chla_pix'] = int(len(valid))
    else:
        cell['chla_mean'] = np.nan
        cell['chla_median'] = np.nan
        cell['chla_log_mean'] = np.nan
        cell['chla_std'] = np.nan
        cell['chla_max'] = np.nan
        cell['n_chla_pix'] = int(len(valid))


# ============================================================
# FILTER TO CELLS WITH VALID Chl-a
# ============================================================
valid_cells = [c for c in cells
               if np.isfinite(c.get('chla_log_mean', np.nan))
               and c['n_chla_pix'] >= 3]
n_lost = len(cells) - len(valid_cells)

print(f"  Valid cells: {len(valid_cells)}/{len(cells)} ({n_lost} dropped)")

if n_lost > 0:
    print(f"  Dropped cells:")
    for c in cells:
        if c not in valid_cells:
            print(f"    ({c['lat']:.1f}, {c['lon']:.1f}) — {c['n_chla_pix']} Chl-a pixels")

n = len(valid_cells)
S_arr = np.array([c['S'] for c in valid_cells])
logR_arr = np.array([c['logR'] for c in valid_cells])
R_arr = np.array([c['R_i'] for c in valid_cells])
lat_arr = np.array([c['lat'] for c in valid_cells])

# Primary: log-mean (geometric mean proxy — Chl-a is log-normal)
chla_logmean_arr = np.array([c['chla_log_mean'] for c in valid_cells])
chla_mean_arr = np.array([c['chla_mean'] for c in valid_cells])
chla_std_arr = np.array([c['chla_std'] for c in valid_cells])
chla_max_arr = np.array([c['chla_max'] for c in valid_cells])

chla_metrics = {
    'chla_log_mean': chla_logmean_arr,
    'chla_mean': chla_mean_arr,
    'chla_std': chla_std_arr,
    'chla_max': chla_max_arr,
}

print(f"\n  Chl-a log-mean range: [{chla_logmean_arr.min():.3f}, {chla_logmean_arr.max():.3f}]")
print(f"  Chl-a mean range:     [{chla_mean_arr.min():.3f}, {chla_mean_arr.max():.1f}] mg/m3")


# ============================================================
# TEST 1: CORRELATIONS
# ============================================================
print(f"\n{'='*70}")
print("TEST 1: CORRELATIONS — Chl-a metrics vs logR and S")
print(f"{'='*70}")
print(f"\n  n = {n} cells")

print(f"\n  {'Metric':<15} {'vs logR':>12} {'p':>8} {'vs S':>12} {'p':>8}")
print(f"  {'-'*55}")

correlations = {}
for name, arr in chla_metrics.items():
    rho_R, p_R = spearmanr(arr, logR_arr)
    rho_S, p_S = spearmanr(arr, S_arr)
    sig_R = "*" if p_R < 0.05 else ""
    sig_S = "*" if p_S < 0.05 else ""
    print(f"  {name:<15} {rho_R:>+8.3f} {sig_R:>3} {p_R:>8.4f}"
          f"  {rho_S:>+8.3f} {sig_S:>3} {p_S:>8.4f}")
    correlations[name] = {
        'rho_logR': round(float(rho_R), 4),
        'p_logR': round(float(p_R), 4),
        'rho_S': round(float(rho_S), 4),
        'p_S': round(float(p_S), 4),
    }

rho_sR, p_sR = spearmanr(S_arr, logR_arr)
print(f"  {'S (ref)':<15} {rho_sR:>+8.3f} {'*':>3} {p_sR:>8.4f}")


# ============================================================
# TEST 2: R-SQUARED COMPARISON
# ============================================================
print(f"\n{'='*70}")
print("TEST 2: R-SQUARED COMPARISON — best Chl-a predictor vs S")
print(f"{'='*70}")

ss_tot = np.sum((logR_arr - logR_arr.mean())**2)

# Model A: logR ~ S
X_a = np.column_stack([np.ones(n), S_arr])
b_a, _, _, _ = lstsq(X_a, logR_arr, rcond=None)
ss_a = np.sum((logR_arr - X_a @ b_a)**2)
r2_a = 1 - ss_a / ss_tot

print(f"\n  Model A (logR ~ S):           R2 = {r2_a:.4f}")

best_chla_name = None
best_chla_r2 = -999
best_chla_arr = None

for name, arr in chla_metrics.items():
    X = np.column_stack([np.ones(n), arr])
    b, _, _, _ = lstsq(X, logR_arr, rcond=None)
    ss = np.sum((logR_arr - X @ b)**2)
    r2 = 1 - ss / ss_tot
    sig = " <-- beats S!" if r2 > r2_a else ""
    print(f"  Model Chl_{name:<12} (logR ~ {name:<12}): R2 = {r2:.4f}{sig}")

    if r2 > best_chla_r2:
        best_chla_r2 = r2
        best_chla_name = name
        best_chla_arr = arr

print(f"\n  Best Chl-a predictor: {best_chla_name} (R2 = {best_chla_r2:.4f})")
print(f"  S R2 = {r2_a:.4f}")

if best_chla_r2 > r2_a:
    print(f"  -> Chl-a metric beats S in R2!")
else:
    print(f"  -> S beats all Chl-a metrics in R2")


# ============================================================
# TEST 3: COMBINED MODEL + NESTED F-TESTS (decisive test)
# ============================================================
print(f"\n{'='*70}")
print(f"TEST 3: COMBINED MODEL — logR ~ S + {best_chla_name}")
print(f"{'='*70}")

# Model B: logR ~ best_chla only
X_b = np.column_stack([np.ones(n), best_chla_arr])
b_b, _, _, _ = lstsq(X_b, logR_arr, rcond=None)
ss_b = np.sum((logR_arr - X_b @ b_b)**2)
r2_b = 1 - ss_b / ss_tot

# Model C: logR ~ S + best_chla
X_c = np.column_stack([np.ones(n), S_arr, best_chla_arr])
b_c, _, _, _ = lstsq(X_c, logR_arr, rcond=None)
ss_c = np.sum((logR_arr - X_c @ b_c)**2)
r2_c = 1 - ss_c / ss_tot

print(f"\n  Model A (S only):                 R2 = {r2_a:.4f}")
print(f"  Model B ({best_chla_name} only):  R2 = {r2_b:.4f}")
print(f"  Model C (S + {best_chla_name}):   R2 = {r2_c:.4f}")
print(f"    beta_S = {b_c[1]:+.4f}, beta_{best_chla_name} = {b_c[2]:+.6f}")

# Nested F-tests
df_full = n - 3
F_s_given_chla = ((ss_b - ss_c) / 1) / (ss_c / df_full)
p_s_given_chla = 1 - f_dist.cdf(F_s_given_chla, 1, df_full)

F_chla_given_s = ((ss_a - ss_c) / 1) / (ss_c / df_full)
p_chla_given_s = 1 - f_dist.cdf(F_chla_given_s, 1, df_full)

print(f"\n  F-test: S adds to {best_chla_name}?     F = {F_s_given_chla:.3f}, p = {p_s_given_chla:.4f}")
print(f"  F-test: {best_chla_name} adds to S?     F = {F_chla_given_s:.3f}, p = {p_chla_given_s:.4f}")

# Bootstrap CIs
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

print(f"\n  Bootstrap CIs (95%, n_boot={N_BOOT}):")
for i, bname in enumerate(['intercept', 'beta_S', f'beta_{best_chla_name}']):
    lo = np.percentile(boot_c[:, i], 2.5)
    hi = np.percentile(boot_c[:, i], 97.5)
    print(f"    {bname:25s}: {b_c[i]:+.6f}  [{lo:+.6f}, {hi:+.6f}]")


# ============================================================
# TEST 4: KITCHEN SINK — S + Chl-a + depth metrics
# ============================================================
print(f"\n{'='*70}")
print("TEST 4: KITCHEN SINK — logR ~ S + Chl-a + depth_std + mean_depth")
print(f"{'='*70}")

# Load ETOPO for depth metrics
print(f"\n  Loading ETOPO... ({elapsed()})")
ds_etopo = nc.Dataset(os.path.join(DATA_DIR, "etopo_subset.nc"))
if 'y' in ds_etopo.variables:
    elev_lats = ds_etopo.variables['y'][:]
    elev_lons = ds_etopo.variables['x'][:]
else:
    elev_lats = ds_etopo.variables['lat'][:]
    elev_lons = ds_etopo.variables['lon'][:]
elevation = ds_etopo.variables['z'][:]
ds_etopo.close()

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

# Full kitchen sink: S + best_chla + depth_std + mean_depth
X_full = np.column_stack([np.ones(n), S_arr, best_chla_arr,
                          depth_std_arr, mean_depth_arr])
b_full, _, _, _ = lstsq(X_full, logR_arr, rcond=None)
ss_full = np.sum((logR_arr - X_full @ b_full)**2)
r2_full = 1 - ss_full / ss_tot

# Without S
X_no_s = np.column_stack([np.ones(n), best_chla_arr,
                          depth_std_arr, mean_depth_arr])
b_no_s, _, _, _ = lstsq(X_no_s, logR_arr, rcond=None)
ss_no_s = np.sum((logR_arr - X_no_s @ b_no_s)**2)
r2_no_s = 1 - ss_no_s / ss_tot

F_s_final = ((ss_no_s - ss_full) / 1) / (ss_full / (n - 5))
p_s_final = 1 - f_dist.cdf(F_s_final, 1, n - 5)

print(f"\n  All confounds (Chl-a + depth):   R2 = {r2_no_s:.4f}")
print(f"  S + all confounds:               R2 = {r2_full:.4f}")
print(f"  beta_S in full model:            {b_full[1]:+.4f}")
print(f"  F-test: S adds to everything?    F = {F_s_final:.3f}, p = {p_s_final:.4f}")

if p_s_final < 0.05:
    print(f"\n  >>> S SURVIVES even after controlling for Chl-a + depth metrics")
    final_verdict = "S_SURVIVES_ALL"
else:
    print(f"\n  >>> S does NOT survive Chl-a + depth controls")
    final_verdict = "S_ABSORBED"


# ============================================================
# TEST 5: Canyon vs non-canyon — Chl-a comparison
# ============================================================
print(f"\n{'='*70}")
print("TEST 5: Canyon (S>0) vs non-canyon (S=0) — Chl-a levels")
print(f"{'='*70}")

s_pos = S_arr > 0
s_zero = S_arr == 0
print(f"\n  S>0: {s_pos.sum()} cells, S=0: {s_zero.sum()} cells")

mw_results = {}
for name, arr in chla_metrics.items():
    if s_pos.sum() > 0 and s_zero.sum() > 0:
        U, p_mw = mannwhitneyu(arr[s_pos], arr[s_zero], alternative='two-sided')
        sig = "*" if p_mw < 0.05 else ""
        print(f"  {name:<15}: S>0 mean = {arr[s_pos].mean():>8.3f}, "
              f"S=0 mean = {arr[s_zero].mean():>8.3f}, "
              f"MWU p = {p_mw:.4f} {sig}")
        mw_results[name] = {
            'S_pos_mean': round(float(arr[s_pos].mean()), 4),
            'S_zero_mean': round(float(arr[s_zero].mean()), 4),
            'U': round(float(U), 1),
            'p': round(float(p_mw), 4),
        }


# ============================================================
# TEST 6: COLLINEARITY CHECK
# ============================================================
print(f"\n{'='*70}")
print("TEST 6: COLLINEARITY CHECK — S vs Chl-a, Chl-a vs depth")
print(f"{'='*70}")

collinearity = {}

print(f"\n  S vs Chl-a metrics:")
for name, arr in chla_metrics.items():
    rho, p = spearmanr(S_arr, arr)
    print(f"    S vs {name:<15}: rho = {rho:+.3f}, p = {p:.4f}")
    collinearity[f"S_vs_{name}"] = {'rho': round(float(rho), 4),
                                     'p': round(float(p), 4)}

print(f"\n  Chl-a vs depth metrics:")
for name, arr in chla_metrics.items():
    rho_d, p_d = spearmanr(arr, mean_depth_arr)
    rho_ds, p_ds = spearmanr(arr, depth_std_arr)
    print(f"    {name:<15} vs mean_depth: rho = {rho_d:+.3f}, p = {p_d:.4f}")
    print(f"    {name:<15} vs depth_std:  rho = {rho_ds:+.3f}, p = {p_ds:.4f}")


# ============================================================
# TEST 7: RADIUS SENSITIVITY SWEEP
# ============================================================
print(f"\n{'='*70}")
print("TEST 7: RADIUS SENSITIVITY — F(S|Chl-a) at different search radii")
print(f"{'='*70}")

radius_results = {}
for test_radius in [50, 100, 150, 200]:
    deg_r = test_radius / 80.0

    # Recompute Chl-a per cell at this radius
    r_cells = []
    for cell in cells:
        clat, clon = cell['lat'], cell['lon']
        cands = ocean_tree.query_ball_point([clat, clon], deg_r)
        if len(cands) > 0:
            c_lats = ocean_lat[cands]
            c_lons = ocean_lon[cands]
            c_vals = ocean_chla[cands]
            dists = haversine_km(clat, clon, c_lats, c_lons)
            valid = c_vals[dists <= test_radius]
        else:
            valid = np.array([])

        if len(valid) >= 3:
            r_cells.append({
                'S': cell['S'], 'logR': cell['logR'],
                'chla_log_mean': float(np.mean(np.log(valid))),
            })

    nr = len(r_cells)
    if nr < 15:
        print(f"\n  Radius {test_radius:3d} km: n={nr} cells — SKIP (too few)")
        radius_results[test_radius] = {'n': nr, 'verdict': 'SKIP'}
        continue

    S_r = np.array([c['S'] for c in r_cells])
    logR_r = np.array([c['logR'] for c in r_cells])
    chla_r = np.array([c['chla_log_mean'] for c in r_cells])

    ss_tot_r = np.sum((logR_r - logR_r.mean())**2)
    X_s = np.column_stack([np.ones(nr), S_r])
    X_ch = np.column_stack([np.ones(nr), chla_r])
    X_both = np.column_stack([np.ones(nr), S_r, chla_r])

    b_s, _, _, _ = lstsq(X_s, logR_r, rcond=None)
    b_ch, _, _, _ = lstsq(X_ch, logR_r, rcond=None)
    b_both, _, _, _ = lstsq(X_both, logR_r, rcond=None)

    ss_s = np.sum((logR_r - X_s @ b_s)**2)
    ss_ch = np.sum((logR_r - X_ch @ b_ch)**2)
    ss_both = np.sum((logR_r - X_both @ b_both)**2)

    F_s = ((ss_ch - ss_both) / 1) / (ss_both / (nr - 3))
    p_s = 1 - f_dist.cdf(F_s, 1, nr - 3)

    F_ch = ((ss_s - ss_both) / 1) / (ss_both / (nr - 3))
    p_ch = 1 - f_dist.cdf(F_ch, 1, nr - 3)

    rho_sc, _ = spearmanr(S_r, chla_r)

    label = "S_DOM" if (p_s < 0.05 and p_ch >= 0.05) else \
            "CHL_DOM" if (p_s >= 0.05 and p_ch < 0.05) else \
            "BOTH" if (p_s < 0.05 and p_ch < 0.05) else "COLLIN"

    print(f"\n  Radius {test_radius:3d} km: n={nr}, rho(S,Chl-a)={rho_sc:+.3f}")
    print(f"    F(S|Chl-a) = {F_s:.3f}, p = {p_s:.4f}")
    print(f"    F(Chl-a|S) = {F_ch:.3f}, p = {p_ch:.4f}")
    print(f"    -> {label}")

    radius_results[test_radius] = {
        'n': nr, 'rho_S_chla': round(float(rho_sc), 4),
        'F_S_given_chla': round(F_s, 3), 'p_S_given_chla': round(p_s, 4),
        'F_chla_given_S': round(F_ch, 3), 'p_chla_given_S': round(p_ch, 4),
        'verdict': label,
    }


# ============================================================
# INTERPRETATION
# ============================================================
print(f"\n{'='*70}")
print("INTERPRETATION")
print(f"{'='*70}")

if p_s_given_chla < 0.05 and p_chla_given_s >= 0.05:
    interp = "S_DOMINANT"
    print("""
  S SURVIVES Chl-a control, Chl-a does NOT add to S.
  -> Upwelling (Chl-a) is NOT a better explanation.
  -> Canyon geometry carries unique info beyond upwelling.
  -> The UAP-canyon association is not reducible to upwelling optics.
""")
elif p_s_given_chla >= 0.05 and p_chla_given_s < 0.05:
    interp = "CHLA_ABSORBS_S"
    print("""
  Chl-a ABSORBS S!
  -> Upwelling IS a candidate geophysical explanation.
  -> Canyon steepness may just proxy for upwelling intensity.
  -> The UAP pattern may reflect atmospheric/optical phenomena from
     cold upwelled water (fog, inversions, fata morgana).
""")
elif p_s_given_chla < 0.05 and p_chla_given_s < 0.05:
    interp = "BOTH_CONTRIBUTE"
    print("""
  Both S and Chl-a contribute independently.
  -> Upwelling explains part of the variance but not all.
  -> Canyon geometry adds information beyond upwelling.
  -> Partial upwelling confound — some signal is geophysical.
""")
else:
    interp = "COLLINEAR"
    print("""
  Neither significant in combined model.
  -> High collinearity prevents separation.
  -> Cannot determine whether S or Chl-a is the true driver.
  -> Would need instrumental variable or different approach.
""")


# ============================================================
# SAVE
# ============================================================
print(f"\n{'='*70}")
print("SAVE RESULTS")
print(f"{'='*70}")

results = {
    "test": "MODIS Chl-a upwelling confound",
    "data_source": "MODIS Aqua L3 mapped Chl-a, monthly 4km, mean 2003-2020, NASA ERDDAP",
    "n_cells_original": len(cells),
    "n_cells_valid": n,
    "n_cells_dropped": n_lost,
    "correlations": correlations,
    "S_vs_logR": {"rho": round(float(rho_sR), 4), "p": round(float(p_sR), 4)},
    "best_chla_predictor": best_chla_name,
    "models": {
        "A_S_only": {"R2": round(r2_a, 4)},
        "B_chla_only": {"predictor": best_chla_name, "R2": round(best_chla_r2, 4)},
        "C_combined": {"R2": round(r2_c, 4)},
        "full_confounds_no_S": {"R2": round(r2_no_s, 4)},
        "full_with_S": {"R2": round(r2_full, 4)},
    },
    "F_tests": {
        "S_given_chla": {"F": round(F_s_given_chla, 3),
                         "p": round(p_s_given_chla, 4)},
        "chla_given_S": {"F": round(F_chla_given_s, 3),
                         "p": round(p_chla_given_s, 4)},
        "S_given_all_confounds": {"F": round(F_s_final, 3),
                                   "p": round(p_s_final, 4)},
    },
    "bootstrap": {
        "n_boot": N_BOOT,
        "beta_S": {
            "estimate": round(float(b_c[1]), 6),
            "ci_lo": round(float(np.percentile(boot_c[:, 1], 2.5)), 6),
            "ci_hi": round(float(np.percentile(boot_c[:, 1], 97.5)), 6),
        },
        f"beta_{best_chla_name}": {
            "estimate": round(float(b_c[2]), 6),
            "ci_lo": round(float(np.percentile(boot_c[:, 2], 2.5)), 6),
            "ci_hi": round(float(np.percentile(boot_c[:, 2], 97.5)), 6),
        },
    },
    "canyon_vs_noncanyon": mw_results,
    "radius_sensitivity": {str(k): v for k, v in radius_results.items()},
    "search_radius_km": SEARCH_RADIUS_KM,
    "interpretation": interp,
    "final_verdict": final_verdict,
}

out_file = os.path.join(OUT_DIR, "phase_e_chla_confound.json")
with open(out_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved: {out_file}")

import shutil
repo_out = os.path.join(BASE_DIR, "uap-canyon-analysis", "results", "phase_ev2")
os.makedirs(repo_out, exist_ok=True)
shutil.copy2(out_file, repo_out)
print(f"  Copied to repo: {repo_out}")

print(f"\n{'='*70}")
print(f"VERDICT: {interp}")
print(f"KITCHEN SINK: {final_verdict}")
print(f"{'='*70}")

print(f"\nDONE ({elapsed()})")
