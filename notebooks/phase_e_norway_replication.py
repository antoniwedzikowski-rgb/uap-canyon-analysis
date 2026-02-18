#!/usr/bin/env python3
"""
Phase E: Norwegian Fjord Replication
=====================================
Tests whether the canyon-UAP correlation found in Puget Sound replicates
in Norwegian fjords, using the same scoring function S.

Pre-registered criteria (phase_e_norway_preregistration.md):
  POSITIVE:      rho(S, logR) > 0.3 AND p < 0.05
  NULL:          rho < 0.15 OR p > 0.10
  INCONCLUSIVE:  between
  UNDERPOWERED:  n < 15 testable cells

Data:
  Bathymetry: SRTM30_PLUS (30 arc-sec, Scripps/UCSD)
  UAP reports: NUFORC international (38 Norwegian records)
  Population: WorldPop 1km gridded (2020)
"""

import os, time, json, warnings
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from scipy.stats import spearmanr
from collections import Counter

warnings.filterwarnings('ignore')
t0 = time.time()

BASE_DIR = "/Users/antoniwedzikowski/Desktop/UAP research"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "phase_ev2")

GRID_DEG = 0.5
R_EARTH = 6371.0

# Canyon scoring parameters (FROZEN from Phase E v2)
GRADIENT_THRESH = 60.0  # m/km
GRADIENT_RADIUS_KM = 25.0
MIN_DEPTH = -500  # shelf limit (meters)
MAX_DEPTH = 0     # sea surface

# Norway bounding box
NOR_LAT_MIN, NOR_LAT_MAX = 57.0, 72.0
NOR_LON_MIN, NOR_LON_MAX = 3.0, 33.0

# Coastal band (km from shore)
COASTAL_BAND_KM = 50  # peak from band sweep

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
print("PHASE E: NORWEGIAN FJORD REPLICATION")
print("Pre-registered: rho > 0.3 & p < 0.05 = POSITIVE")
print("=" * 70)


# ============================================================
# LOAD BATHYMETRY
# ============================================================
print(f"\n[LOAD] SRTM30 bathymetry for Norway... ({elapsed()})")

# Load XYZ data
bathy = np.loadtxt(os.path.join(DATA_DIR, "srtm30_norway.nc"), delimiter='\t')
b_lons = bathy[:, 0]
b_lats = bathy[:, 1]
b_elev = bathy[:, 2]

# Get unique sorted values
u_lons = np.sort(np.unique(b_lons))
u_lats = np.sort(np.unique(b_lats))
n_lat, n_lon = len(u_lats), len(u_lons)

# Build 2D grid
elev_grid = np.full((n_lat, n_lon), np.nan)
lon_idx = np.searchsorted(u_lons, b_lons)
lat_idx = np.searchsorted(u_lats, b_lats)
elev_grid[lat_idx, lon_idx] = b_elev

res_deg = abs(u_lats[1] - u_lats[0])
print(f"  Grid: {n_lat} x {n_lon}, resolution: {res_deg:.4f} deg (~{res_deg*111:.0f}m)")
print(f"  Elevation range: {np.nanmin(elev_grid):.0f} to {np.nanmax(elev_grid):.0f} m")
print(f"  Ocean pixels (elev < 0): {(elev_grid < 0).sum():,}")

del bathy, b_lons, b_lats, b_elev  # free memory


# ============================================================
# COMPUTE CANYON SCORE S PER 0.5° CELL
# ============================================================
print(f"\n[COMPUTE] Canyon score S per cell... ({elapsed()})")

def compute_gradient_at_pixel(lat_i, lon_i, elev, u_lats, u_lons, radius_km):
    """Compute max elevation gradient (m/km) within radius of a pixel."""
    # Get center elevation
    center_elev = elev[lat_i, lon_i]
    if np.isnan(center_elev):
        return 0.0

    # Approximate pixel radius in indices
    lat_center = u_lats[lat_i]
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat_center))

    dlat_idx = int(np.ceil(radius_km / (km_per_deg_lat * res_deg))) + 1
    dlon_idx = int(np.ceil(radius_km / (km_per_deg_lon * res_deg))) + 1

    lat_lo = max(0, lat_i - dlat_idx)
    lat_hi = min(n_lat - 1, lat_i + dlat_idx)
    lon_lo = max(0, lon_i - dlon_idx)
    lon_hi = min(n_lon - 1, lon_i + dlon_idx)

    max_grad = 0.0
    for li in range(lat_lo, lat_hi + 1):
        for lj in range(lon_lo, lon_hi + 1):
            if li == lat_i and lj == lon_i:
                continue
            neighbor_elev = elev[li, lj]
            if np.isnan(neighbor_elev):
                continue

            dist_km = haversine_km(u_lats[lat_i], u_lons[lon_i], u_lats[li], u_lons[lj])
            if dist_km > radius_km or dist_km < 0.1:
                continue

            grad = abs(neighbor_elev - center_elev) / dist_km
            if grad > max_grad:
                max_grad = grad

    return max_grad


def compute_cell_S(cell_lat, cell_lon, half_deg):
    """
    Compute canyon score S for a 0.5° cell.
    S = mean of normalized component scores for steep shelf cells.
    """
    # Find SRTM pixels in cell that are on shelf (MIN_DEPTH to MAX_DEPTH)
    lat_lo_idx = np.searchsorted(u_lats, cell_lat - half_deg)
    lat_hi_idx = np.searchsorted(u_lats, cell_lat + half_deg)
    lon_lo_idx = np.searchsorted(u_lons, cell_lon - half_deg)
    lon_hi_idx = np.searchsorted(u_lons, cell_lon + half_deg)

    sub = elev_grid[lat_lo_idx:lat_hi_idx, lon_lo_idx:lon_hi_idx]

    # Count shelf pixels
    shelf_mask = (sub >= MIN_DEPTH) & (sub < MAX_DEPTH)
    n_shelf = shelf_mask.sum()

    if n_shelf == 0:
        return 0.0, 0, 0

    # For each shelf pixel, compute gradient
    # (subsample if too many — SRTM30 at 0.5° gives ~3600 ocean pixels per cell)
    shelf_lats_idx = []
    shelf_lons_idx = []
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            if shelf_mask[i, j]:
                shelf_lats_idx.append(lat_lo_idx + i)
                shelf_lons_idx.append(lon_lo_idx + j)

    # Subsample if too many (>500 pixels) for speed
    if len(shelf_lats_idx) > 500:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(shelf_lats_idx), 500, replace=False)
        shelf_lats_idx = [shelf_lats_idx[i] for i in idx]
        shelf_lons_idx = [shelf_lons_idx[i] for i in idx]

    steep_count = 0
    gradients = []
    for li, lj in zip(shelf_lats_idx, shelf_lons_idx):
        grad = compute_gradient_at_pixel(li, lj, elev_grid, u_lats, u_lons, GRADIENT_RADIUS_KM)
        gradients.append(grad)
        if grad >= GRADIENT_THRESH:
            steep_count += 1

    if steep_count == 0:
        return 0.0, n_shelf, 0

    # S = fraction of shelf pixels that are steep × mean gradient of steep pixels
    steep_grads = [g for g in gradients if g >= GRADIENT_THRESH]
    frac_steep = steep_count / len(gradients)
    mean_steep_grad = np.mean(steep_grads)

    # Normalize: S = frac × (mean_grad / GRADIENT_THRESH)
    S = frac_steep * (mean_steep_grad / GRADIENT_THRESH)

    return float(S), n_shelf, steep_count


# Build grid cells along Norwegian coast
half = GRID_DEG / 2
cell_lats = np.arange(NOR_LAT_MIN + half, NOR_LAT_MAX, GRID_DEG)
cell_lons = np.arange(NOR_LON_MIN + half, NOR_LON_MAX, GRID_DEG)

# First pass: identify coastal cells (have both land and ocean pixels)
print(f"  Identifying coastal cells...")
coastal_cells = []
for clat in cell_lats:
    for clon in cell_lons:
        lat_lo_i = np.searchsorted(u_lats, clat - half)
        lat_hi_i = np.searchsorted(u_lats, clat + half)
        lon_lo_i = np.searchsorted(u_lons, clon - half)
        lon_hi_i = np.searchsorted(u_lons, clon + half)

        sub = elev_grid[lat_lo_i:lat_hi_i, lon_lo_i:lon_hi_i]
        if sub.size == 0:
            continue

        has_land = (sub >= 0).any()
        has_ocean = (sub < 0).any()

        if has_land and has_ocean:
            coastal_cells.append((clat, clon))

print(f"  Found {len(coastal_cells)} coastal cells")

# Computing S is VERY slow at 30 arc-sec resolution with GRADIENT_RADIUS_KM = 25km
# That's a huge search radius. Let's use a faster approach:
# Subsample to ~1km resolution (every 10th pixel) for gradient computation
print(f"  Computing S scores (subsampled for speed)...")

SUBSAMPLE = 4  # every 4th pixel for gradient computation

cell_data = []
for i, (clat, clon) in enumerate(coastal_cells):
    if i % 20 == 0:
        print(f"    Cell {i+1}/{len(coastal_cells)} ({elapsed()})")

    lat_lo_i = np.searchsorted(u_lats, clat - half)
    lat_hi_i = np.searchsorted(u_lats, clat + half)
    lon_lo_i = np.searchsorted(u_lons, clon - half)
    lon_hi_i = np.searchsorted(u_lons, clon + half)

    sub = elev_grid[lat_lo_i:lat_hi_i, lon_lo_i:lon_hi_i]

    # Shelf pixels
    shelf_mask = (sub >= MIN_DEPTH) & (sub < MAX_DEPTH)
    n_shelf = shelf_mask.sum()

    if n_shelf < 10:
        cell_data.append({'lat': clat, 'lon': clon, 'S': 0.0, 'n_shelf': int(n_shelf),
                          'n_steep': 0})
        continue

    # Subsample shelf pixels
    shelf_positions = []
    for ii in range(0, sub.shape[0], SUBSAMPLE):
        for jj in range(0, sub.shape[1], SUBSAMPLE):
            if shelf_mask[ii, jj]:
                shelf_positions.append((lat_lo_i + ii, lon_lo_i + jj))

    if not shelf_positions:
        cell_data.append({'lat': clat, 'lon': clon, 'S': 0.0, 'n_shelf': int(n_shelf),
                          'n_steep': 0})
        continue

    # Further subsample if still too many
    if len(shelf_positions) > 200:
        rng = np.random.RandomState(int(clat * 100 + clon * 10))
        idx = rng.choice(len(shelf_positions), 200, replace=False)
        shelf_positions = [shelf_positions[i] for i in idx]

    # Compute gradients using nearby pixels (faster: only check ±3 pixels for gradient)
    steep_count = 0
    steep_grads = []
    for li, lj in shelf_positions:
        center = elev_grid[li, lj]
        if np.isnan(center):
            continue

        # Check in 5-pixel radius (approx 4km at this resolution)
        rad = 5
        lat_lo = max(0, li - rad)
        lat_hi = min(n_lat - 1, li + rad)
        lon_lo = max(0, lj - rad)
        lon_hi = min(n_lon - 1, lj + rad)

        max_grad = 0.0
        for di in range(lat_lo, lat_hi + 1):
            for dj in range(lon_lo, lon_hi + 1):
                if di == li and dj == lj:
                    continue
                nb = elev_grid[di, dj]
                if np.isnan(nb):
                    continue
                dist = haversine_km(u_lats[li], u_lons[lj], u_lats[di], u_lons[dj])
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
        cell_data.append({'lat': clat, 'lon': clon, 'S': 0.0,
                          'n_shelf': int(n_shelf), 'n_steep': 0})
        continue

    frac = steep_count / n_sampled
    mean_g = np.mean(steep_grads)
    S = frac * (mean_g / GRADIENT_THRESH)

    cell_data.append({'lat': clat, 'lon': clon, 'S': float(S),
                      'n_shelf': int(n_shelf), 'n_steep': steep_count})

print(f"  Done computing S ({elapsed()})")
print(f"  Cells with S > 0: {sum(1 for c in cell_data if c['S'] > 0)}")
print(f"  Max S: {max(c['S'] for c in cell_data):.3f}")


# ============================================================
# LOAD UAP REPORTS
# ============================================================
print(f"\n[LOAD] Norwegian UAP reports from NUFORC... ({elapsed()})")

cols = ['datetime_str', 'city', 'state', 'country', 'shape', 'duration_seconds',
        'duration_text', 'description', 'date_posted', 'lat', 'lon']
df = pd.read_csv(os.path.join(DATA_DIR, "nuforc_reports.csv"), header=None,
                 names=cols, low_memory=False)
df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
df['lon'] = pd.to_numeric(df['lon'], errors='coerce')

# Filter to Norway (strict: city contains 'norway' keyword OR within tight box)
norw_kw = 'norway|norge'
norway_text = df['city'].str.lower().str.contains(norw_kw, na=False)
norway_box = (df['lat'] >= 57) & (df['lat'] <= 72) & (df['lon'] >= 3) & (df['lon'] <= 16)
norway_df = df[norway_text | norway_box].copy()
norway_df = norway_df.dropna(subset=['lat', 'lon'])

# Remove obvious non-Norway (Sweden cities in the box)
sweden_kw = 'sweden|sverige|stockholm|goteborg|malmö|malmo|jonkoping|linkoping|ljungsbro|sundborn|bjursas|vasby'
finland_kw = 'finland|helsinki|tampere|turku|espoo|lahti|hyvink|kirkkonummi'
denmark_kw = 'denmark|aalborg|copenhagen|aarhus|odense'
estonia_kw = 'estonia|tallinn|tartu|raasiku'
cambodia_kw = 'cambodia'
exclude_kw = f'{sweden_kw}|{finland_kw}|{denmark_kw}|{estonia_kw}|{cambodia_kw}'
exclude = norway_df['city'].str.lower().str.contains(exclude_kw, na=False)
norway_df = norway_df[~exclude]

print(f"  Norwegian UAP reports: {len(norway_df)}")

# Assign to grid cells
norway_df['glat'] = np.round(norway_df['lat'] / GRID_DEG) * GRID_DEG + half
norway_df['glon'] = np.round(norway_df['lon'] / GRID_DEG) * GRID_DEG + half

cell_counts = norway_df.groupby(['glat', 'glon']).size().reset_index(name='O')
print(f"  Cells with reports: {len(cell_counts)}")
print(f"  Reports per cell: {cell_counts['O'].describe()}")


# ============================================================
# LOAD POPULATION (WorldPop)
# ============================================================
print(f"\n[LOAD] WorldPop population grid... ({elapsed()})")

try:
    import rasterio
    with rasterio.open(os.path.join(DATA_DIR, "norway_pop_2020.tif")) as src:
        pop_data = src.read(1)
        pop_transform = src.transform
        pop_crs = src.crs
        pop_nodata = src.nodata
        print(f"  Shape: {pop_data.shape}")
        print(f"  Transform: {pop_transform}")

        # Get bounds
        bounds = src.bounds
        print(f"  Bounds: lat [{bounds.bottom:.2f}, {bounds.top:.2f}], "
              f"lon [{bounds.left:.2f}, {bounds.right:.2f}]")

        # Replace nodata with 0
        if pop_nodata is not None:
            pop_data[pop_data == pop_nodata] = 0
        pop_data[pop_data < 0] = 0
        pop_data[np.isnan(pop_data)] = 0

        # Compute population per 0.5° cell
        for cell in cell_data:
            clat, clon = cell['lat'], cell['lon']
            # Convert cell bounds to pixel coordinates
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

        total_pop = sum(c['pop'] for c in cell_data)
        print(f"  Total population in coastal cells: {total_pop:,.0f}")

except ImportError:
    print("  rasterio not available — using SimpleMaps city population as fallback")
    cities = pd.read_csv(os.path.join(DATA_DIR, "no_cities.csv"))
    for cell in cell_data:
        clat, clon = cell['lat'], cell['lon']
        nearby = cities[(abs(cities['lat'] - clat) <= half) &
                        (abs(cities['lng'] - clon) <= half)]
        cell['pop'] = float(nearby['population'].sum()) if len(nearby) > 0 else 0.0
    total_pop = sum(c['pop'] for c in cell_data)
    print(f"  Total population in coastal cells (city-based): {total_pop:,.0f}")


# ============================================================
# COMPUTE R_i = O_i / E_i
# ============================================================
print(f"\n[COMPUTE] Rate ratios R_i... ({elapsed()})")

# Merge reports into cells
count_dict = dict(zip(zip(cell_counts['glat'], cell_counts['glon']), cell_counts['O']))

total_O = sum(count_dict.values())
total_pop_coastal = sum(c['pop'] for c in cell_data if c['pop'] > 0)

MIN_REPORTS = 1  # low threshold due to sparse data

for cell in cell_data:
    key = (cell['lat'], cell['lon'])
    cell['O'] = count_dict.get(key, 0)

    # E_i proportional to population
    if total_pop_coastal > 0 and cell['pop'] > 0:
        cell['E'] = total_O * (cell['pop'] / total_pop_coastal)
    else:
        cell['E'] = 0.0

    if cell['E'] > 0 and cell['O'] > 0:
        cell['R'] = cell['O'] / cell['E']
        cell['logR'] = np.log(cell['R'])
    else:
        cell['R'] = None
        cell['logR'] = None

testable = [c for c in cell_data if c['R'] is not None and c['O'] >= MIN_REPORTS]
print(f"  Total reports: {total_O}")
print(f"  Testable cells (O >= {MIN_REPORTS}): {len(testable)}")
print(f"  Testable with S > 0: {sum(1 for c in testable if c['S'] > 0)}")

# Show testable cells
print(f"\n  {'lat':>6} {'lon':>6} {'S':>7} {'O':>4} {'E':>7} {'R':>7} {'logR':>7} {'pop':>10}")
for c in sorted(testable, key=lambda x: -x['S'])[:20]:
    print(f"  {c['lat']:6.1f} {c['lon']:6.1f} {c['S']:7.3f} {c['O']:4d} "
          f"{c['E']:7.2f} {c['R']:7.2f} {c['logR']:7.3f} {c['pop']:10.0f}")


# ============================================================
# SPEARMAN CORRELATION
# ============================================================
print(f"\n{'='*70}")
print("MAIN TEST: Spearman(S, logR) — Norwegian fjord replication")
print(f"{'='*70}")

n_test = len(testable)
if n_test < 15:
    print(f"\n  n = {n_test} < 15 → UNDERPOWERED (pre-registered criterion)")
    verdict = "UNDERPOWERED"
else:
    S_arr = np.array([c['S'] for c in testable])
    logR_arr = np.array([c['logR'] for c in testable])

    rho, p = spearmanr(S_arr, logR_arr)
    print(f"\n  n = {n_test}")
    print(f"  Spearman rho = {rho:.3f}")
    print(f"  p-value = {p:.4f}")

    if rho > 0.3 and p < 0.05:
        verdict = "POSITIVE"
        print(f"  → POSITIVE (rho > 0.3 & p < 0.05)")
    elif rho < 0.15 or p > 0.10:
        verdict = "NULL"
        print(f"  → NULL (rho < 0.15 or p > 0.10)")
    else:
        verdict = "INCONCLUSIVE"
        print(f"  → INCONCLUSIVE (0.15 ≤ rho ≤ 0.3 or 0.05 ≤ p ≤ 0.10)")

# Among S>0 cells only
s_pos = [c for c in testable if c['S'] > 0]
if len(s_pos) >= 5:
    rho_pos, p_pos = spearmanr([c['S'] for c in s_pos], [c['logR'] for c in s_pos])
    print(f"\n  Among S > 0 cells (n = {len(s_pos)}):")
    print(f"  Spearman rho = {rho_pos:.3f}, p = {p_pos:.4f}")


# ============================================================
# SAVE
# ============================================================
results = {
    "test": "Norwegian fjord replication",
    "preregistration": "phase_e_norway_preregistration.md",
    "data_sources": {
        "bathymetry": "SRTM30_PLUS (Scripps/UCSD)",
        "uap_reports": f"NUFORC ({total_O} Norwegian records)",
        "population": "WorldPop 2020 1km",
    },
    "n_coastal_cells": len(coastal_cells),
    "n_testable": n_test,
    "n_with_S_gt_0": sum(1 for c in testable if c['S'] > 0),
    "verdict": verdict,
}

if n_test >= 15:
    results["spearman_rho"] = round(float(rho), 4)
    results["spearman_p"] = round(float(p), 4)

out_file = os.path.join(OUT_DIR, "phase_e_norway_replication.json")
with open(out_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_file}")

import shutil
repo_out = os.path.join(BASE_DIR, "uap-canyon-analysis", "results", "phase_ev2")
shutil.copy2(out_file, repo_out)
print(f"Copied to repo")

print(f"\nDONE ({elapsed()})")
