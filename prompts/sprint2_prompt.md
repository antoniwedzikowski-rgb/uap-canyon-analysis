# Sprint 2 (v2): Continuous Model, Uncertainty Quantification & Geocoding Robustness

## Context

Sprint 1 established that UAP reports cluster near submarine canyons (logistic regression β ≈ -0.16, p < 10⁻⁹⁶) and this effect survives controls for port/marina proximity, port density, coastline morphology, and multiple imputation strategies. The preferred model is **Model B** (full sample n=61,985, 6 covariates including port variables, no coastline morphology).

Sprint 2 addresses three reviewer objections:
1. **Reviewer #3 §1:** Report effects as CI/bootstrapped uncertainty, not just p-values
2. **Reviewer #3 §3:** Show results in a continuous model (GAM/splines) to eliminate "threshold fishing" accusation
3. **Reviewer #2 §3:** Geocoding noise — test stability under location jittering

## Existing files

- `sprint1_observer_controls.py` — Sprint 1 analysis
- `sprint1_fix_nan_bias.py` — NaN fix with Models A–G
- `sprint1_fix_results.json` — all Sprint 1 fix results
- `data/port_coords_cache.npz` — 7,747 port/marina locations
- Original data arrays: uap_coords, control_coords, canyon_cells, dist_to_canyon_uap/ctrl, etc.

Use Model B specification throughout (6 covariates, full sample, no coast_complexity).

---

## Task 1: GAM — Continuous Distance-Response Curve

Fit a Generalized Additive Model with a spline term for distance-to-canyon.

### CRITICAL: Variable scaling

- `dist_to_canyon` in **raw kilometers** (NOT z-scored) — the partial dependence plot must be readable in km
- All other covariates: z-scored (helps fitting stability)

### Implementation

```python
# Option A (preferred): pyGAM
from pygam import LogisticGAM, s, l

# Feature matrix: column 0 = dist_to_canyon in RAW KM, columns 1-5 = z-scored other covariates
gam = LogisticGAM(
    s(0, n_splines=20) +  # dist_to_canyon — spline, raw km
    l(1) +                 # dist_to_military (z-scored)
    l(2) +                 # pop_density (z-scored)
    l(3) +                 # dist_to_coast (z-scored)
    l(4) +                 # dist_to_nearest_port (z-scored)
    l(5)                   # log_port_count_25km (z-scored)
)
gam.gridsearch(X, y)

# Extract partial dependence for canyon distance
XX = gam.generate_X_grid(term=0, n=200)
pdep, confi = gam.partial_dependence(term=0, X=XX, width=0.95)
```

```python
# Option B (fallback): statsmodels with natural cubic splines
import patsy
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial

formula = 'y ~ cr(dist_to_canyon_km, df=6) + dist_to_military_z + pop_density_z + dist_to_coast_z + dist_to_nearest_port_z + log_port_count_25km_z'
y_endog, X_design = patsy.dmatrices(formula, data=df, return_type='dataframe')
model_gam = GLM(y_endog, X_design, family=Binomial()).fit()
```

```python
# Option C (last fallback): scipy spline on binned data
# Bin distance-to-canyon into 1 km bins (0–300 km)
# Compute UAP fraction per bin (adjusted for population control density)
# Fit smoothing spline to binned proportions
```

### Model comparison: cross-validated log-loss

Do NOT compare AIC between pyGAM and statsmodels — different likelihood implementations make ΔAIC unreliable.

Instead, compare GAM vs linear logistic via 5-fold CV:

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {'gam_logloss': [], 'linear_logloss': [], 'gam_auc': [], 'linear_auc': []}

for train_idx, test_idx in kf.split(X, y):
    # GAM
    gam_cv = LogisticGAM(s(0, n_splines=20) + l(1) + l(2) + l(3) + l(4) + l(5))
    gam_cv.fit(X[train_idx], y[train_idx])
    pred_gam = gam_cv.predict_proba(X[test_idx])
    cv_results['gam_logloss'].append(log_loss(y[test_idx], pred_gam))
    cv_results['gam_auc'].append(roc_auc_score(y[test_idx], pred_gam))

    # Linear logistic (statsmodels or sklearn)
    from sklearn.linear_model import LogisticRegression
    lr_cv = LogisticRegression(max_iter=1000)
    lr_cv.fit(X[train_idx], y[train_idx])
    pred_lr = lr_cv.predict_proba(X[test_idx])[:, 1]
    cv_results['linear_logloss'].append(log_loss(y[test_idx], pred_lr))
    cv_results['linear_auc'].append(roc_auc_score(y[test_idx], pred_lr))

print(f"GAM:    mean log-loss = {np.mean(cv_results['gam_logloss']):.4f}, AUC = {np.mean(cv_results['gam_auc']):.4f}")
print(f"Linear: mean log-loss = {np.mean(cv_results['linear_logloss']):.4f}, AUC = {np.mean(cv_results['linear_auc']):.4f}")
```

### Deliverable: Partial dependence plot

- X-axis: distance to nearest canyon in **kilometers** (0–300 km)
- Y-axis: partial effect (relative log-odds, centered to mean)
- **Caption must state:** "Relative log-odds centered to mean effect. Shaded band = 95% CI."
- Vertical reference lines at 10, 25, 50, 100 km
- Save as `figures/sprint2_gam_partial_dependence.png`

---

## Task 2: Bootstrap Confidence Intervals — Two Variants

### Variant A: Point bootstrap (standard)

Stratified resample of individual UAP and control points.

```python
n_boot = 2000

boot_results_point = {
    'canyon_beta': [], 'or_10km': [], 'or_25km': [], 'or_50km': [],
    'mean_dist_diff': [],
}

for i in range(n_boot):
    boot_uap = np.random.choice(uap_indices, size=len(uap_indices), replace=True)
    boot_ctrl = np.random.choice(ctrl_indices, size=len(ctrl_indices), replace=True)
    boot_idx = np.concatenate([boot_uap, boot_ctrl])

    X_boot = X_full[boot_idx]
    y_boot = y_full[boot_idx]

    try:
        model_boot = sm.Logit(y_boot, X_boot).fit(disp=0, maxiter=50)
        boot_results_point['canyon_beta'].append(model_boot.params['dist_to_canyon'])
    except:
        continue

    # ORs at thresholds
    dist_boot = dist_to_canyon_all[boot_idx]
    for threshold, key in [(10, 'or_10km'), (25, 'or_25km'), (50, 'or_50km')]:
        near_uap = np.mean(dist_boot[y_boot == 1] < threshold)
        near_ctrl = np.mean(dist_boot[y_boot == 0] < threshold)
        if near_ctrl > 0 and near_uap > 0 and (1-near_ctrl) > 0 and (1-near_uap) > 0:
            or_val = (near_uap / (1-near_uap)) / (near_ctrl / (1-near_ctrl))
            boot_results_point[key].append(or_val)

    diff = np.mean(dist_boot[y_boot == 1]) - np.mean(dist_boot[y_boot == 0])
    boot_results_point['mean_dist_diff'].append(diff)

    if (i+1) % 200 == 0:
        print(f"Point bootstrap: {i+1}/{n_boot}")
```

### Variant B: Cluster bootstrap (spatially honest)

Resample entire spatial clusters instead of individual points to account for spatial autocorrelation.

```python
# Step 1: Assign each point to a spatial cluster (25 km grid cells)
def assign_grid_cluster(coords, cell_size_km=25):
    """Assign points to grid cells. Returns cluster labels."""
    # Convert to approximate planar coordinates centered on CONUS
    ref_lat, ref_lon = 39.0, -98.0  # center of CONUS
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(ref_lat))

    x_km = (coords[:, 1] - ref_lon) * km_per_deg_lon
    y_km = (coords[:, 0] - ref_lat) * km_per_deg_lat

    grid_x = np.floor(x_km / cell_size_km).astype(int)
    grid_y = np.floor(y_km / cell_size_km).astype(int)

    # Create unique cluster ID from grid coordinates
    cluster_ids = grid_x * 10000 + grid_y
    return cluster_ids

all_coords = np.concatenate([uap_coords, control_coords])
cluster_ids = assign_grid_cluster(all_coords, cell_size_km=25)
unique_clusters = np.unique(cluster_ids)

print(f"Number of spatial clusters (25 km grid): {len(unique_clusters)}")

# Step 2: Cluster bootstrap
n_boot_cluster = 2000
boot_results_cluster = {
    'canyon_beta': [], 'or_10km': [], 'or_25km': [], 'or_50km': [],
    'mean_dist_diff': [],
}

for i in range(n_boot_cluster):
    # Resample clusters with replacement
    sampled_clusters = np.random.choice(unique_clusters, size=len(unique_clusters), replace=True)

    # Collect all points from sampled clusters
    boot_idx = []
    for c in sampled_clusters:
        boot_idx.extend(np.where(cluster_ids == c)[0])
    boot_idx = np.array(boot_idx)

    X_boot = X_full[boot_idx]
    y_boot = y_full[boot_idx]

    # Skip if too few of one class
    if y_boot.sum() < 100 or (len(y_boot) - y_boot.sum()) < 100:
        continue

    try:
        model_boot = sm.Logit(y_boot, X_boot).fit(disp=0, maxiter=50)
        boot_results_cluster['canyon_beta'].append(model_boot.params['dist_to_canyon'])
    except:
        continue

    # ORs and distance diff — same as point bootstrap
    dist_boot = dist_to_canyon_all[boot_idx]
    for threshold, key in [(10, 'or_10km'), (25, 'or_25km'), (50, 'or_50km')]:
        near_uap = np.mean(dist_boot[y_boot == 1] < threshold)
        near_ctrl = np.mean(dist_boot[y_boot == 0] < threshold)
        if near_ctrl > 0 and near_uap > 0 and (1-near_ctrl) > 0 and (1-near_uap) > 0:
            or_val = (near_uap / (1-near_uap)) / (near_ctrl / (1-near_ctrl))
            boot_results_cluster[key].append(or_val)

    diff = np.mean(dist_boot[y_boot == 1]) - np.mean(dist_boot[y_boot == 0])
    boot_results_cluster['mean_dist_diff'].append(diff)

    if (i+1) % 200 == 0:
        print(f"Cluster bootstrap: {i+1}/{n_boot_cluster}")
```

### Report both variants side by side:

```python
for metric in ['canyon_beta', 'or_10km', 'or_25km', 'or_50km', 'mean_dist_diff']:
    pt = np.array(boot_results_point[metric])
    cl = np.array(boot_results_cluster[metric])
    print(f"{metric}:")
    print(f"  Point:   median={np.median(pt):.4f}, 95% CI=[{np.percentile(pt,2.5):.4f}, {np.percentile(pt,97.5):.4f}]")
    print(f"  Cluster: median={np.median(cl):.4f}, 95% CI=[{np.percentile(cl,2.5):.4f}, {np.percentile(cl,97.5):.4f}]")
    print(f"  CI width ratio (cluster/point): {(np.percentile(cl,97.5)-np.percentile(cl,2.5)) / (np.percentile(pt,97.5)-np.percentile(pt,2.5)):.2f}x")
```

**Expected:** Cluster CIs will be wider (more honest). Key question: does the cluster CI for canyon_beta still exclude zero? If yes, the effect is robust to spatial autocorrelation.

### Deliverable: Bootstrap distribution plots

2×3 subplot grid: rows = point vs cluster, columns = canyon_beta, OR@25km, mean_dist_diff.
Each panel: histogram with 95% CI lines, point estimate marked, null (0 or 1) marked.
Save as `figures/sprint2_bootstrap_distributions.png`

---

## Task 3: Geocoding Jitter Test

### Spherical jitter (corrected method)

Do NOT convert km to degrees with simple division. Use proper spherical destination point:

```python
def jitter_coords_spherical(coords, sigma_km):
    """
    Add Gaussian noise on the sphere.
    For each point: sample random bearing (0-360) and random distance (Gaussian with sigma_km),
    then compute destination point using haversine forward formula.
    """
    R = 6371.0  # Earth radius in km
    n = len(coords)
    lat_rad = np.radians(coords[:, 0])
    lon_rad = np.radians(coords[:, 1])

    # Random bearing (uniform 0 to 2π) and distance (half-normal, folded to positive)
    bearings = np.random.uniform(0, 2 * np.pi, n)
    distances = np.abs(np.random.normal(0, sigma_km, n))  # km, always positive
    angular_dist = distances / R  # in radians

    # Destination point formula
    lat2 = np.arcsin(
        np.sin(lat_rad) * np.cos(angular_dist) +
        np.cos(lat_rad) * np.sin(angular_dist) * np.cos(bearings)
    )
    lon2 = lon_rad + np.arctan2(
        np.sin(bearings) * np.sin(angular_dist) * np.cos(lat_rad),
        np.cos(angular_dist) - np.sin(lat_rad) * np.sin(lat2)
    )

    jittered = np.column_stack([np.degrees(lat2), np.degrees(lon2)])
    return jittered
```

### Jitter test loop

```python
jitter_levels = [2, 5, 10, 15, 20]  # km
n_jitter_iterations = 50

jitter_results = {}
for sigma in jitter_levels:
    betas = []
    ors_10 = []
    ors_25 = []
    for i in range(n_jitter_iterations):
        # Jitter UAP coordinates
        jittered_uap = jitter_coords_spherical(uap_coords, sigma)

        # Recompute BOTH canyon distance AND port distance for jittered points
        dist_canyon_jittered = compute_nearest_dist(jittered_uap, canyon_tree)  # BallTree
        dist_port_jittered = compute_nearest_dist(jittered_uap, port_tree)      # BallTree

        # Rebuild feature matrix with jittered distances
        # Keep control points unchanged
        # Refit Model B
        # ...

        betas.append(model_jitter.params['dist_to_canyon'])

        # Compute ORs at thresholds using jittered canyon distances
        for threshold, container in [(10, ors_10), (25, ors_25)]:
            near_uap = np.mean(dist_canyon_jittered < threshold)
            near_ctrl = np.mean(dist_to_canyon_ctrl < threshold)
            if near_ctrl > 0 and near_uap > 0 and (1-near_ctrl) > 0 and (1-near_uap) > 0:
                or_val = (near_uap / (1-near_uap)) / (near_ctrl / (1-near_ctrl))
                container.append(or_val)

    jitter_results[sigma] = {
        'beta_mean': np.mean(betas),
        'beta_ci': np.percentile(betas, [2.5, 97.5]),
        'or10_mean': np.mean(ors_10),
        'or10_ci': np.percentile(ors_10, [2.5, 97.5]),
        'or25_mean': np.mean(ors_25),
        'or25_ci': np.percentile(ors_25, [2.5, 97.5]),
    }
    print(f"Jitter σ={sigma} km: β={np.mean(betas):.4f} [{np.percentile(betas,2.5):.4f}, {np.percentile(betas,97.5):.4f}]")
```

**IMPORTANT:** Both `dist_to_canyon` and `dist_to_nearest_port` must be recomputed for jittered points. If you only jitter canyon distance, the test is unfair (port covariate retains unjittered precision).

### Deliverable: Jitter stability plot

Two-panel figure:
- Left panel: canyon β vs jitter σ (0, 2, 5, 10, 15, 20 km) with error bars = 95% CI across iterations
- Right panel: OR@25km vs jitter σ, same format
- Horizontal dashed line at β=0 / OR=1
- σ=0 point = original (unjittered) result from Model B
- Label axes clearly with units

**Interpretation thresholds to print:**
- "Geocoding-robust at σ=X km" = last σ where 95% CI excludes zero/one
- If OR@10km collapses but OR@25km holds → recommend 25 km as "geocoding-robust threshold" in paper, with 10 km as "high-precision scenario"

Save as `figures/sprint2_jitter_stability.png`

---

## Task 4: Combined Summary

```
SPRINT 2 RESULTS
================

1. GAM PARTIAL DEPENDENCE
   - Effect onset: canyon signal separates from zero at ~X km
   - Peak effect: strongest at ~X km
   - Effect extinction: indistinguishable from zero beyond ~X km
   - Shape: [monotonic decay / threshold / plateau / non-monotonic]
   - 5-fold CV comparison:
     GAM:    log-loss = X.XXXX, AUC = X.XXXX
     Linear: log-loss = X.XXXX, AUC = X.XXXX
     → [GAM improves / comparable / worse]

2. BOOTSTRAP 95% CIs (n=2000 each)

   POINT BOOTSTRAP:
   - Canyon β: X [CI_lo, CI_hi]
   - OR@10km: X [CI_lo, CI_hi]
   - OR@25km: X [CI_lo, CI_hi]
   - OR@50km: X [CI_lo, CI_hi]
   - Mean dist difference: X km [CI_lo, CI_hi]

   CLUSTER BOOTSTRAP (25 km grid):
   - Canyon β: X [CI_lo, CI_hi]
   - OR@10km: X [CI_lo, CI_hi]
   - OR@25km: X [CI_lo, CI_hi]
   - OR@50km: X [CI_lo, CI_hi]
   - Mean dist difference: X km [CI_lo, CI_hi]

   CI width ratio (cluster/point): X.Xx
   Cluster CI excludes zero: [YES / NO] ← KEY RESULT

3. GEOCODING JITTER
   - Canyon β stable through σ = X km (last σ where CI excludes zero)
   - OR@10km stable through σ = X km
   - OR@25km stable through σ = X km
   - Recommended reporting threshold: X km

4. REVIEWER RESPONSES
   - R3§1 (CIs): ✓ Both point and cluster bootstrap CIs reported
   - R3§3 (continuous model): ✓ GAM shows [shape], CV-validated
   - R2§3 (geocoding): ✓ Effect stable through σ=X km, spherical jitter
   - Spatial autocorrelation: ✓ Cluster bootstrap CI [excludes/includes] zero
```

## Output files

- `sprint2_continuous_model.py` — full analysis script
- `sprint2_results.json` — all numerical results
- `figures/sprint2_gam_partial_dependence.png` — THE key figure
- `figures/sprint2_bootstrap_distributions.png` — 2×3 panel (point vs cluster × 3 metrics)
- `figures/sprint2_jitter_stability.png` — 2-panel jitter stability

## Notes

- Install pyGAM: `pip install pygam`. If it fails, use Option B or C for GAM.
- Bootstrap total: 2×2000 iterations = 4000 model fits on n≈62k. This will be slow (1-2 hours). Print progress every 200. If taking too long, reduce to 1000 per variant.
- For jitter: 5 levels × 50 iterations = 250 model fits + BallTree recomputations. Precompute canyon and port BallTrees once and reuse.
- Consistent figure style across all plots: clear axis labels with units, CI bands/bars, null reference lines. No specific style library required — just make it clean and readable.
- The GAM partial dependence plot is the single most important figure in the paper.
