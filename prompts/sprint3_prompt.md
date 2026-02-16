# Sprint 3 (v2): Temporal Clustering + Dose-Response

## Context

Sprints 1–2 established a robust spatial association between UAP reports and submarine canyons (β ≈ -0.16, surviving port/marina controls, cluster bootstrap, GAM continuous model, and geocoding jitter).

Sprint 3 asks two new, independent questions using existing data:

1. **Temporal clustering:** Do near-canyon UAP reports show episodic bursts ("flaps") beyond what their higher rate predicts?
2. **Dose-response:** Do steeper canyons predict more excess UAP reports than shallow ones?

Both are independent of Sprints 1–2. Each can kill or strengthen the finding.

## Existing files

- Sprint 1/2 scripts and results
- `data/port_coords_cache.npz` — 7,747 port/marina locations
- Original arrays: uap_coords, control_coords, canyon_cells
- NUFORC data with datetime field
- ETOPO 2022 bathymetry (OPeNDAP or cached)
- `dist_to_canyon_uap`, `dist_to_canyon_ctrl`

---

## PRE-REGISTERED ANALYSIS PLAN

Declare primary specifications BEFORE seeing results. Everything else is sensitivity.

```
PRIMARY SPECIFICATIONS (report as main results):

  TEMPORAL:
    Spatial radius:     50 km
    Temporal window:    ±7 days
    Canyon threshold:   25 km
    Null model:         time permutation within year (1000 iterations)
    Metric:             excess_near − excess_far (observed/expected ratio)

  DOSE-RESPONSE:
    Gradient feature:   p95_gradient_within_25km (95th percentile, not max)
    Transform:          log1p
    Model:              Model B + gradient feature (logistic regression, LR test)
    Primary plot:       OR by gradient bins (6 bins)

SENSITIVITY (report as supplementary):
    Temporal windows:   3, 14, 30 days
    Spatial radii:      25, 100 km
    Canyon thresholds:  10, 50 km
    Gradient features:  max, mean_top10%, count_steep (gradient > 50 m/km)
    BH-FDR correction across sensitivity grid
```

Print this plan at the top of console output, timestamped, BEFORE any results.

---

## PART A: Temporal Clustering

### Hypothesis

If a spatially localized source exists near canyons, near-canyon reports should show temporal bursts — multiple reports from the same area within a short time window — more than far-from-canyon reports, *even after accounting for higher overall reporting rate*.

### Step A1: Parse NUFORC datetimes

```python
import pandas as pd
import numpy as np

# Adjust column name/format to actual dataset
df_uap['datetime'] = pd.to_datetime(df_uap['date_time'], format='mixed', dayfirst=False)
df_uap['year'] = df_uap['datetime'].dt.year

# Convert to numeric days since epoch for fast arithmetic
uap_days = (df_uap['datetime'] - pd.Timestamp('2000-01-01')).dt.total_seconds() / 86400.0
uap_days = uap_days.values
uap_years = df_uap['year'].values

print(f"Date range: {df_uap['datetime'].min()} to {df_uap['datetime'].max()}")
print(f"Reports per year: min={df_uap['year'].value_counts().min()}, max={df_uap['year'].value_counts().max()}")
```

### Step A2: Define groups

```python
CANYON_THRESHOLD_PRIMARY = 25  # km

near_mask = dist_to_canyon_uap < CANYON_THRESHOLD_PRIMARY
far_mask = ~near_mask

print(f"Near-canyon (< {CANYON_THRESHOLD_PRIMARY} km): {near_mask.sum()}")
print(f"Far-canyon (>= {CANYON_THRESHOLD_PRIMARY} km): {far_mask.sum()}")
```

### Step A3: Compute observed temporal density

For each report, count how many other reports occurred within SPATIAL_RADIUS km AND ±TEMPORAL_WINDOW days.

```python
from sklearn.neighbors import BallTree

SPATIAL_RADIUS = 50  # km (primary)
TEMPORAL_WINDOW = 7  # days (primary)

uap_tree = BallTree(np.radians(uap_coords), metric='haversine')
spatial_radius_rad = SPATIAL_RADIUS / 6371.0

# Find spatial neighbors for all points (indices)
spatial_neighbors = uap_tree.query_radius(np.radians(uap_coords), r=spatial_radius_rad)

# Count temporal neighbors within window
temporal_density = np.zeros(len(uap_coords))
n_spatial_neighbors = np.zeros(len(uap_coords))

for i in range(len(uap_coords)):
    neighbors = spatial_neighbors[i]
    neighbors = neighbors[neighbors != i]  # exclude self
    n_spatial_neighbors[i] = len(neighbors)

    if len(neighbors) == 0:
        temporal_density[i] = 0
        continue

    time_diffs = np.abs(uap_days[neighbors] - uap_days[i])
    temporal_density[i] = np.sum(time_diffs <= TEMPORAL_WINDOW)

    if (i + 1) % 5000 == 0:
        print(f"Temporal density: {i+1}/{len(uap_coords)}")
```

**Performance:** If this is too slow (>30 min), subsample to 20k reports stratified by near/far canyon and note in results.

### Step A4: Time-permutation null (PRIMARY NULL MODEL)

The key test. Permute report dates within each year to create a null that preserves:
- Spatial geometry (who is near what)
- Annual reporting rate (boom years stay boomy)
- Seasonal effects within year are disrupted (this is the signal we're testing)

```python
N_PERMUTATIONS = 1000

# Group indices by year for within-year permutation
year_groups = {}
for yr in np.unique(uap_years):
    year_groups[yr] = np.where(uap_years == yr)[0]

def compute_excess_near_minus_far(temporal_density_arr, n_spatial_arr, near_mask, far_mask,
                                   temporal_window, year_groups, uap_years):
    """
    Compute Poisson excess ratio for near vs far canyon groups.
    Expected = n_spatial_neighbors × (2 × window / days_in_year_of_this_report)
    Excess = observed / expected
    Returns: median_excess_near - median_excess_far
    """
    # Days per year for each report
    days_in_year = np.array([365 + (1 if yr % 4 == 0 and (yr % 100 != 0 or yr % 400 == 0) else 0)
                              for yr in uap_years])

    expected = n_spatial_arr * (2 * temporal_window / days_in_year)

    # Avoid division by zero
    valid = expected > 0
    excess = np.full(len(temporal_density_arr), np.nan)
    excess[valid] = temporal_density_arr[valid] / expected[valid]

    excess_near = excess[near_mask & valid]
    excess_far = excess[far_mask & valid]

    return np.nanmedian(excess_near) - np.nanmedian(excess_far)

# Observed statistic
observed_diff = compute_excess_near_minus_far(
    temporal_density, n_spatial_neighbors, near_mask, far_mask,
    TEMPORAL_WINDOW, year_groups, uap_years
)
print(f"Observed excess difference (near − far): {observed_diff:.4f}")

# Permutation loop
perm_diffs = []
for p in range(N_PERMUTATIONS):
    # Permute dates within each year
    perm_days = uap_days.copy()
    for yr, indices in year_groups.items():
        perm_days[indices] = np.random.permutation(perm_days[indices])

    # Recompute temporal density with permuted dates
    # (spatial neighbors stay the same — only time changes)
    td_perm = np.zeros(len(uap_coords))
    for i in range(len(uap_coords)):
        neighbors = spatial_neighbors[i]
        neighbors = neighbors[neighbors != i]
        if len(neighbors) == 0:
            continue
        time_diffs = np.abs(perm_days[neighbors] - perm_days[i])
        td_perm[i] = np.sum(time_diffs <= TEMPORAL_WINDOW)

    perm_diff = compute_excess_near_minus_far(
        td_perm, n_spatial_neighbors, near_mask, far_mask,
        TEMPORAL_WINDOW, year_groups, uap_years
    )
    perm_diffs.append(perm_diff)

    if (p + 1) % 100 == 0:
        print(f"Permutation: {p+1}/{N_PERMUTATIONS}")

perm_diffs = np.array(perm_diffs)
p_value_perm = np.mean(perm_diffs >= observed_diff)
print(f"\nPermutation test: observed = {observed_diff:.4f}, "
      f"p = {p_value_perm:.4f} ({N_PERMUTATIONS} permutations)")
print(f"Permutation null distribution: mean = {perm_diffs.mean():.4f}, "
      f"std = {perm_diffs.std():.4f}, 95th pct = {np.percentile(perm_diffs, 95):.4f}")
```

**WARNING:** This is O(N_PERMUTATIONS × N_reports × mean_neighbors). At 1000 perms × 42k reports this will be very slow. Optimization strategies:
- Reduce to 200 permutations for initial run, increase if significant
- Subsample to 20k reports (stratified) and note this
- Vectorize the inner loop: precompute `perm_days[spatial_neighbors[i]]` as padded arrays

```python
# Optimization: vectorize temporal counting with padded neighbor arrays
# Precompute once:
max_neighbors = max(len(sn) for sn in spatial_neighbors)
neighbor_matrix = np.full((len(uap_coords), max_neighbors), -1, dtype=int)
for i, sn in enumerate(spatial_neighbors):
    sn_no_self = sn[sn != i]
    neighbor_matrix[i, :len(sn_no_self)] = sn_no_self

# In permutation loop:
# perm_days_neighbors = perm_days[neighbor_matrix]  # shape (N, max_neighbors)
# perm_days_self = perm_days[:, None]  # shape (N, 1)
# time_diffs = np.abs(perm_days_neighbors - perm_days_self)
# time_diffs[neighbor_matrix == -1] = np.inf  # mask padding
# td_perm = np.sum(time_diffs <= TEMPORAL_WINDOW, axis=1)
```

### Step A5: Sensitivity grid

Run with pre-registered alternative parameters. Report as supplementary.

```python
temporal_windows = [3, 7, 14, 30]  # primary = 7
spatial_radii = [25, 50, 100]       # primary = 50
canyon_thresholds = [10, 25, 50]    # primary = 25

# For sensitivity, use fewer permutations (200) to keep runtime manageable
N_PERM_SENSITIVITY = 200

sensitivity_results = {}
for tw in temporal_windows:
    for sr in spatial_radii:
        for ct in canyon_thresholds:
            # Recompute with these parameters
            # Skip if it's the primary spec (already computed above with 1000 perms)
            if tw == 7 and sr == 50 and ct == 25:
                sensitivity_results[(tw, sr, ct)] = {
                    'p': p_value_perm,
                    'observed_diff': observed_diff,
                    'is_primary': True
                }
                continue

            near_mask_s = dist_to_canyon_uap < ct
            far_mask_s = ~near_mask_s

            # Recompute spatial neighbors at this radius
            sr_rad = sr / 6371.0
            sn_s = uap_tree.query_radius(np.radians(uap_coords), r=sr_rad)
            # ... compute temporal density, permutation test ...

            sensitivity_results[(tw, sr, ct)] = {
                'p': p_val, 'observed_diff': obs_diff, 'is_primary': False
            }

# Report: fraction of combinations with same sign as primary + BH-FDR
from statsmodels.stats.multitest import multipletests

all_pvals = [v['p'] for v in sensitivity_results.values()]
_, fdr_pvals, _, _ = multipletests(all_pvals, method='fdr_bh')

n_significant = sum(1 for fp in fdr_pvals if fp < 0.05)
n_same_sign = sum(1 for v in sensitivity_results.values()
                   if np.sign(v['observed_diff']) == np.sign(observed_diff))
print(f"\nSensitivity: {n_same_sign}/{len(sensitivity_results)} combinations same sign as primary")
print(f"Sensitivity: {n_significant}/{len(sensitivity_results)} significant after BH-FDR at 0.05")
```

### Step A6: Identify specific flaps (DESCRIPTIVE ONLY)

Find top temporal clusters near canyons as illustrative case studies. Mark clearly as descriptive/exploratory.

```python
# Find reports with high temporal density AND near canyon
flap_threshold = np.percentile(temporal_density[temporal_density > 0], 95)
flap_mask = (temporal_density >= flap_threshold) & near_mask

print(f"\n--- DESCRIPTIVE: Top temporal clusters near canyons ---")
print(f"(These are illustrative case studies, not statistical evidence.)")
print(f"N reports in top-5% temporal density near canyons: {flap_mask.sum()}")

if flap_mask.sum() > 10:
    from sklearn.cluster import DBSCAN

    flap_coords = uap_coords[flap_mask]
    flap_times = uap_days[flap_mask]

    # DBSCAN: eps in degrees, time scaled to match spatial
    flap_features = np.column_stack([
        flap_coords[:, 0],
        flap_coords[:, 1],
        flap_times / 7.0 * 0.45  # 7 days ≈ 50 km in clustering space
    ])

    clustering = DBSCAN(eps=0.5, min_samples=3).fit(flap_features)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Distinct flap episodes (DBSCAN): {n_clusters}")

    # Top clusters by size
    for label in range(min(n_clusters, 5)):
        members = np.where(labels == label)[0]
        lats = flap_coords[members, 0]
        lons = flap_coords[members, 1]
        times = pd.to_datetime(uap_days[flap_mask][members], unit='D', origin='2000-01-01')
        canyon_dists = dist_to_canyon_uap[flap_mask][members]
        print(f"\n  Episode #{label+1}: {len(members)} reports")
        print(f"    Location: {lats.mean():.2f}°N, {lons.mean():.2f}°W")
        print(f"    Time span: {times.min().date()} to {times.max().date()}")
        print(f"    Nearest canyon: {canyon_dists.min():.1f} km")
```

---

## PART B: Dose-Response (Canyon Magnitude)

### Hypothesis

If the association is driven by the canyons themselves (not coastline type), steeper canyons should produce a stronger UAP excess. This is a dose-response test.

### Step B1: Compute canyon magnitude — p95 gradient (NOT max)

Use 95th percentile of bathymetric gradient within 25 km. This is more robust than max (which is dominated by single-pixel noise).

**Implementation: gridded lookup for speed.**

```python
# Create aggregated gradient grid (0.1° resolution) instead of querying full shelf BallTree
# This is ~10x faster than query_radius on 100k+ shelf cells

# Load ETOPO gradient data
# gradient_grid: 2D array of gradient values (m/km) at each ETOPO cell
# lat_grid, lon_grid: coordinate arrays

GRID_RES = 0.1  # degrees (~11 km)

# Aggregate: for each 0.1° cell, store the full distribution of gradient values
from collections import defaultdict

gradient_by_cell = defaultdict(list)
for i in range(len(shelf_coords)):
    lat_bin = round(shelf_coords[i, 0] / GRID_RES) * GRID_RES
    lon_bin = round(shelf_coords[i, 1] / GRID_RES) * GRID_RES
    gradient_by_cell[(lat_bin, lon_bin)].append(shelf_gradients[i])

# For each UAP/control point, find all grid cells within 25 km and compute p95
def get_p95_gradient(points, gradient_by_cell, radius_km=25):
    """
    For each point, gather gradient values from grid cells within radius,
    return 95th percentile. Returns 0 if no shelf cells nearby.
    """
    radius_deg = radius_km / 111.0  # approximate
    p95_gradients = np.zeros(len(points))

    for i in range(len(points)):
        lat, lon = points[i]
        # Gather gradients from nearby grid cells
        all_grads = []
        for dlat in np.arange(-radius_deg, radius_deg + GRID_RES, GRID_RES):
            for dlon in np.arange(-radius_deg, radius_deg + GRID_RES, GRID_RES):
                cell = (round((lat + dlat) / GRID_RES) * GRID_RES,
                        round((lon + dlon) / GRID_RES) * GRID_RES)
                if cell in gradient_by_cell:
                    all_grads.extend(gradient_by_cell[cell])

        if len(all_grads) > 0:
            p95_gradients[i] = np.percentile(all_grads, 95)
        else:
            p95_gradients[i] = 0.0

        if (i + 1) % 5000 == 0:
            print(f"Gradient p95 lookup: {i+1}/{len(points)}")

    return p95_gradients

p95_gradient_uap = get_p95_gradient(uap_coords, gradient_by_cell)
p95_gradient_ctrl = get_p95_gradient(control_coords, gradient_by_cell)

print(f"UAP gradient p95: mean={p95_gradient_uap.mean():.1f}, "
      f"median={np.median(p95_gradient_uap):.1f}, "
      f"zeros={np.sum(p95_gradient_uap == 0)} ({np.mean(p95_gradient_uap == 0)*100:.1f}%)")
print(f"Control gradient p95: mean={p95_gradient_ctrl.mean():.1f}, "
      f"median={np.median(p95_gradient_ctrl):.1f}, "
      f"zeros={np.sum(p95_gradient_ctrl == 0)} ({np.mean(p95_gradient_ctrl == 0)*100:.1f}%)")
```

### Step B2: Handle zero-inflation

Most points far from the shelf will have gradient = 0. Report this explicitly and handle in modeling.

```python
all_gradient = np.concatenate([p95_gradient_uap, p95_gradient_ctrl])
pct_zero = np.mean(all_gradient == 0) * 100

print(f"\nZero-inflation: {pct_zero:.1f}% of points have p95_gradient = 0")
print("(These are points with no shelf cells within 25 km — expected for inland/far-from-shelf locations)")

# Primary approach: log1p transform (handles zeros naturally)
log_gradient = np.log1p(all_gradient)

# Sensitivity: two-part model
# Part 1: is_any_steep = binary (gradient > 0)
is_any_steep = (all_gradient > 0).astype(float)
# Part 2: log_gradient_positive = log1p(gradient) for non-zero only
```

### Step B3: Dose-response bins

```python
bins = [0, 0.01, 5, 10, 20, 50, 100, 500]  # m/km — first bin catches true zeros
bin_labels = ['0 (no shelf)', '0-5', '5-10', '10-20', '20-50', '50-100', '100+']

uap_bins = np.digitize(p95_gradient_uap, bins) - 1
ctrl_bins = np.digitize(p95_gradient_ctrl, bins) - 1

print("\nDose-response: UAP fraction by p95 canyon gradient within 25 km")
print(f"{'Gradient bin':>15} {'N_UAP':>8} {'N_ctrl':>8} {'UAP_frac':>10} {'OR vs ref':>12}")

# Reference: first non-zero bin with sufficient N
ref_bin = 1  # '0-5 m/km'
ref_uap = (uap_bins == ref_bin).sum()
ref_ctrl = (ctrl_bins == ref_bin).sum()
ref_odds = ref_uap / ref_ctrl if ref_ctrl > 0 else np.nan

for b in range(len(bin_labels)):
    n_uap = (uap_bins == b).sum()
    n_ctrl = (ctrl_bins == b).sum()
    frac = n_uap / (n_uap + n_ctrl) if (n_uap + n_ctrl) > 0 else np.nan
    odds = n_uap / n_ctrl if n_ctrl > 0 else np.nan
    or_vs_ref = odds / ref_odds if (ref_odds and ref_odds > 0) else np.nan
    print(f"{bin_labels[b]:>15} {n_uap:>8} {n_ctrl:>8} {frac:>10.3f} {or_vs_ref:>12.2f}")

# Monotonic trend test (Jonckheere-Terpstra or Spearman on bin medians)
from scipy.stats import spearmanr
bin_midpoints = [0, 2.5, 7.5, 15, 35, 75, 200]
bin_ors = []
for b in range(len(bin_labels)):
    n_uap = (uap_bins == b).sum()
    n_ctrl = (ctrl_bins == b).sum()
    odds = n_uap / n_ctrl if n_ctrl > 0 else np.nan
    bin_ors.append(odds / ref_odds if ref_odds > 0 else np.nan)

valid_bins = [(m, o) for m, o in zip(bin_midpoints, bin_ors) if not np.isnan(o)]
if len(valid_bins) >= 3:
    rho, p_trend = spearmanr([v[0] for v in valid_bins], [v[1] for v in valid_bins])
    print(f"\nMonotonic trend test: Spearman rho = {rho:.3f}, p = {p_trend:.4f}")
```

### Step B4: Continuous model — logistic regression

```python
import statsmodels.api as sm
from scipy.stats import zscore
from scipy.stats import chi2

# Model B covariates + gradient
features_dose = pd.DataFrame({
    'dist_to_canyon': dist_all,       # keep however Model B had it
    'dist_to_military': mil_all,
    'pop_density': pop_all,
    'dist_to_coast': coast_all,
    'dist_to_nearest_port': port_dist_all,
    'log_port_count_25km': log_port_all,
    'log_p95_gradient': log_gradient,  # NEW — log1p(p95_gradient_within_25km)
})

y = np.concatenate([np.ones(len(uap_coords)), np.zeros(len(control_coords))])

X_dose = sm.add_constant(features_dose.apply(zscore))
model_dose = sm.Logit(y, X_dose).fit()
print("\n--- Full model with gradient ---")
print(model_dose.summary())

# LR test: does gradient improve over Model B?
# Fit Model B (without gradient)
X_modelb = sm.add_constant(features_dose.drop(columns=['log_p95_gradient']).apply(zscore))
model_b = sm.Logit(y, X_modelb).fit()

lr_stat = 2 * (model_dose.llf - model_b.llf)
lr_pvalue = 1 - chi2.cdf(lr_stat, df=1)
print(f"\nLR test for gradient: chi2 = {lr_stat:.2f}, p = {lr_pvalue:.2e}")
print(f"Gradient coefficient: {model_dose.params['log_p95_gradient']:.4f} "
      f"(95% CI: [{model_dose.conf_int().loc['log_p95_gradient'][0]:.4f}, "
      f"{model_dose.conf_int().loc['log_p95_gradient'][1]:.4f}])")
print(f"\nKey interpretation: gradient adds predictive signal BEYOND simple canyon proximity "
      f"(dist_to_canyon is also in the model).")
```

### Step B5: Port-stratified check

```python
from scipy.stats import spearmanr as spearman_check

# Correlation between gradient and port density
rho_gp, p_gp = spearman_check(p95_gradient_uap[p95_gradient_uap > 0],
                                port_count_25km_uap[p95_gradient_uap > 0])
print(f"\nSpearman(gradient, port_count) among non-zero: rho={rho_gp:.3f}, p={p_gp:.2e}")

# Stratified OR: high gradient vs low gradient within port density terciles
port_all = np.concatenate([port_count_25km_uap, port_count_25km_ctrl])
gradient_all = np.concatenate([p95_gradient_uap, p95_gradient_ctrl])

port_terciles = pd.qcut(port_all, 3, labels=[1, 2, 3]).astype(int)

print("\nGradient effect stratified by port density:")
for t in [1, 2, 3]:
    t_mask = port_terciles == t
    t_uap = t_mask & (y == 1)
    t_ctrl = t_mask & (y == 0)

    high_grad = gradient_all > np.percentile(gradient_all[gradient_all > 0], 75)
    low_grad = (gradient_all > 0) & (gradient_all <= np.percentile(gradient_all[gradient_all > 0], 25))

    n_high_uap = (t_uap & high_grad).sum()
    n_high_ctrl = (t_ctrl & high_grad).sum()
    n_low_uap = (t_uap & low_grad).sum()
    n_low_ctrl = (t_ctrl & low_grad).sum()

    if n_high_ctrl > 0 and n_low_ctrl > 0 and n_low_uap > 0 and n_high_uap > 0:
        or_high = n_high_uap / n_high_ctrl
        or_low = n_low_uap / n_low_ctrl
        print(f"  Port tercile {t}: OR(high gradient)/OR(low gradient) = {or_high/or_low:.2f}")
    else:
        print(f"  Port tercile {t}: insufficient data in one cell")
```

### Step B6: GAM for gradient (if pyGAM available from Sprint 2)

```python
from pygam import LogisticGAM, s, l

# Spline on log_p95_gradient (raw, not z-scored), linear on everything else (z-scored)
# INCLUDE dist_to_canyon as linear — tests whether gradient adds BEYOND proximity

X_gam_dose = np.column_stack([
    log_gradient,                    # col 0: raw log1p(p95_gradient)
    zscore(dist_all),                # col 1: dist_to_canyon
    zscore(mil_all),                 # col 2
    zscore(pop_all),                 # col 3
    zscore(coast_all),               # col 4
    zscore(port_dist_all),           # col 5
    zscore(log_port_all),            # col 6
])

gam_dose = LogisticGAM(
    s(0, n_splines=15) +  # gradient spline
    l(1) + l(2) + l(3) + l(4) + l(5) + l(6)
)
gam_dose.gridsearch(X_gam_dose, y)

XX_grad = gam_dose.generate_X_grid(term=0, n=100)
pdep_grad, confi_grad = gam_dose.partial_dependence(term=0, X=XX_grad, width=0.95)

# Plot: x-axis = exp(log1p_gradient) - 1 = original gradient scale (m/km)
# or keep as log scale with original values on secondary axis
```

### Step B7: Sensitivity — alternative gradient features

```python
# Compute alternative features for sensitivity
def get_max_gradient(points, gradient_by_cell, radius_km=25):
    # Same as p95 but np.max instead of np.percentile
    ...

def get_mean_top10_gradient(points, gradient_by_cell, radius_km=25):
    # Top 10% mean
    ...

def get_count_steep(points, gradient_by_cell, radius_km=25, steep_threshold=50):
    # Count of cells with gradient > threshold
    ...

# Run logistic regression with each alternative
for feature_name, feature_uap, feature_ctrl in [
    ('max_gradient', max_grad_uap, max_grad_ctrl),
    ('mean_top10_gradient', top10_grad_uap, top10_grad_ctrl),
    ('count_steep_50', count_steep_uap, count_steep_ctrl),
]:
    feat_all = np.log1p(np.concatenate([feature_uap, feature_ctrl]))
    features_alt = features_dose.copy()
    features_alt['log_p95_gradient'] = feat_all  # replace
    X_alt = sm.add_constant(features_alt.apply(zscore))
    model_alt = sm.Logit(y, X_alt).fit(disp=0)
    lr_alt = 2 * (model_alt.llf - model_b.llf)
    p_alt = 1 - chi2.cdf(lr_alt, df=1)
    print(f"  {feature_name}: β={model_alt.params.iloc[-1]:.4f}, LR p={p_alt:.2e}")
```

---

## PART C: Combined Summary

```
SPRINT 3 RESULTS
================

PRE-REGISTERED PRIMARY SPECIFICATIONS:
  Temporal: 50 km spatial, ±7 days, 25 km canyon threshold, 1000 time permutations within year
  Dose-response: p95_gradient_within_25km, log1p transform, LR test vs Model B

PART A: TEMPORAL CLUSTERING
----------------------------

1. Observed temporal density:
   Near-canyon: mean = X.XX neighbors (50 km, ±7 days), median = X.XX
   Far-canyon:  mean = X.XX, median = X.XX

2. Time-permutation test (PRIMARY):
   Observed excess_near − excess_far = X.XXXX
   Permutation p-value = X.XXXX (1000 permutations within year)
   Permutation null: mean = X.XXXX, std = X.XXXX

3. VERDICT:
   [TEMPORAL CLUSTERING CONFIRMED — near-canyon reports show excess temporal
    clustering beyond rate difference, p = X.XX from time-permutation null]
   or
   [NO TEMPORAL SIGNAL — permutation null not exceeded, the spatial association
    reflects a rate difference only without episodic structure]

4. Sensitivity: X/36 parameter combinations same sign, X/36 significant after BH-FDR

5. Illustrative flap episodes (DESCRIPTIVE, not statistical evidence):
   Episode #1: X reports, location, date range, nearest canyon
   Episode #2: ...

PART B: DOSE-RESPONSE
----------------------

1. Feature: p95_gradient_within_25km
   Zero-inflation: X.X% of points have gradient = 0

2. Dose-response table:
   Gradient 0 (no shelf): OR = X.XX
   Gradient 0-5 m/km:     OR = 1.00 (reference)
   Gradient 5-10:          OR = X.XX
   Gradient 10-20:         OR = X.XX
   Gradient 20-50:         OR = X.XX
   Gradient 50-100:        OR = X.XX
   Gradient 100+:          OR = X.XX
   Monotonic trend: Spearman rho = X.XX, p = X.XX

3. Continuous model (PRIMARY):
   log_p95_gradient β = X.XXX (95% CI: [X.XXX, X.XXX])
   LR test vs Model B: chi2 = X.XX, p = X.Xe-XX
   NOTE: dist_to_canyon is also in the model — gradient adds BEYOND proximity

4. Port-stratified:
   Port tercile 1 (low):    gradient OR ratio = X.XX
   Port tercile 2 (medium): gradient OR ratio = X.XX
   Port tercile 3 (high):   gradient OR ratio = X.XX

5. Sensitivity (alternative features):
   max_gradient:       β = X.XX, p = X.Xe-XX
   mean_top10:         β = X.XX, p = X.Xe-XX
   count_steep_50:     β = X.XX, p = X.Xe-XX

6. VERDICT:
   [DOSE-RESPONSE CONFIRMED — steeper canyons predict more UAP excess,
    monotonically, beyond simple proximity and port infrastructure.
    Mechanism unknown; gradient may proxy for observational conditions.]
   or
   [NO DOSE-RESPONSE — canyon proximity matters but magnitude does not.
    The association may reflect coastline type, not canyon depth.]

COMBINED SPRINT 3 VERDICT:
   Temporal:      [CONFIRMED / NEGATIVE]
   Dose-response: [CONFIRMED / NEGATIVE]

   Both confirmed:    Two new independent evidence lines. Qualitatively different paper.
   One confirmed:     Partial support. Report with caveats.
   Neither confirmed: Sprint 1-2 finding = spatial rate difference only. Still publishable.
```

## Output files

- `sprint3_temporal_doseresponse.py` — full script
- `sprint3_results.json` — all numerical results
- `figures/sprint3_temporal_permutation.png` — histogram of permutation null + observed line
- `figures/sprint3_temporal_sensitivity.png` — heatmap of p-values across parameter grid
- `figures/sprint3_flap_map.png` — map of illustrative flap episodes (labeled DESCRIPTIVE)
- `figures/sprint3_dose_response_bins.png` — bar chart of OR by gradient bin
- `figures/sprint3_dose_response_gam.png` — GAM partial dependence for gradient
- `figures/sprint3_dose_response_stratified.png` — gradient OR by port density tercile

## Performance Notes

- **Temporal permutation is the bottleneck.** 1000 perms × 42k points × mean ~50 neighbors = billions of comparisons. Use the vectorized padded-array approach. If still >2 hours, reduce to 200 perms (still valid for p-value > 0.005) or subsample to 20k.
- **Gradient grid lookup** is fast (~minutes). Don't use BallTree query_radius on full shelf — use the aggregated 0.1° grid.
- Total runtime estimate: 2–4 hours (dominated by temporal permutations).
