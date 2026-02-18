# UAP-Canyon Spatial Association Analysis

Statistical analysis of the spatial relationship between UAP (Unidentified Anomalous Phenomena) reports from NUFORC and submarine canyon locations along the US coastline.

## Key Finding

UAP reports within 200 km of the US coastline show elevated odds near steep submarine canyon features (gradient >60 m/km), with a weighted OR of 5.18 [95% CI: 4.48, 6.03] relative to flat-shelf controls. The effect is binary (canyon vs. non-canyon), not a smooth dose-response gradient, and survives geocoding pileup correction, aggressive deduplication, and exclusion of known seasonal/astronomical confounds.

**Critical limitation**: The effect is regional. It replicates strongly on the West Coast (OR = 6.21) but **reverses** on the East/Gulf Coast (OR = 0.36). Holding out Puget Sound + Southern California collapses the full-sample effect to null (OR = 0.87). This geographic asymmetry is the primary threat to a general bathymetric interpretation and may reflect confounding with West Coast population/military/observer density patterns rather than a seafloor-driven signal.

### Summary statistics

- **Temporal clustering**: observed exceeds all 1,000 permutations (p < 0.001); survives within-month seasonal control (p = 0.015)
- **Binary canyon signal**: weighted OR = 5.18 [95% CI: 4.48, 6.03] at >60 m/km gradient (1990-2014 CONUS)
- **GAM**: continuous distance decay with 7 covariates, strongest in first ~50 km
- **Cluster bootstrap** (2,000 resamples): canyon beta median = -0.17, 95% CI [-0.26, -0.07]
- **Replication failure**: East/Gulf Coast OR = 0.36, opposite direction; holdout Puget + SoCal = null

## Project Structure

```
uap-canyon-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── figures/
│   ├── figure1_study_area.png
│   ├── figure2_main_panel.png
│   ├── figure3_flap_episodes.png
│   ├── sprint1_*.png                 (3 figures)
│   ├── sprint2_*.png                 (3 figures)
│   ├── sprint3_*.png                 (7 figures)
│   ├── d2_pileup_diagnostics.png     (Phase D)
│   ├── d4_missingness.png            (Phase D)
│   └── d5_seasonality.png            (Phase D)
├── results/
│   ├── sprint1_results.json
│   ├── sprint2_results.json
│   ├── sprint3_results.json
│   ├── sprint3_dose_robustness.json
│   ├── weighted_or_binned.json
│   └── phase_d_results.json          (Phase D — full robustness audit)
├── notebooks/
│   ├── 01_sprint1_observer_controls.py
│   ├── 02_sprint2_continuous_model.py
│   ├── 03_sprint3_temporal_doseresponse.py
│   ├── 04_sprint3_dose_robustness.py
│   ├── 05_weighted_or_for_post.py
│   ├── 06_generate_figures.py
│   └── 07_phase_d_robustness.py      (Phase D — robustness audit)
├── prompts/
│   ├── sprint1_prompt.md
│   ├── sprint2_prompt.md
│   └── sprint3_prompt.md
├── media/
│   └── UAP_Sightings_Cluster_Around_Submarine_Canyons.m4a
└── data/
    ├── sources.md
    ├── nuforc_reports.csv          (80,332 reports, 13 MB)
    ├── census_county_pop.json      (2020 county population)
    ├── county_centroids_pop.csv    (county centroids + pop)
    ├── military_bases_us.csv       (171 DoD installations)
    └── port_coords_cache.npz       (7,747 ports/marinas)
```

## Analysis Phases

### Sprints 1-3: Core Analysis (Scripts 01-06)

**Sprint 1: Observer Controls.** Logistic regression with population-matched controls. Tests whether canyon proximity effect survives addition of port/marina covariates, military base distance, population density, and coastline complexity.

**Sprint 2: Continuous Model & Uncertainty.** GAM partial dependence, cluster bootstrap (2,000 resamples), and geocoding jitter test (50 iterations per sigma level at 2/5/10/15/20 km). Confirms the canyon effect is not an artifact of binary discretization or spatial autocorrelation.

**Sprint 3: Temporal Clustering & Dose-Response.** Spatial permutation testing (1,000 shuffles) for temporal clustering near canyons. Importance-weighted odds ratios by bathymetric gradient bin. Key result: only the steepest gradient bin (>60 m/km) shows a reliable effect.

### Phase D: Robustness Audit (Script 07)

Systematic stress-tests of the headline result, addressing six categories of potential bias. Run on 38,256 coastal NUFORC reports (1990-2014 CONUS) with 1,000 bootstrap resamples per variant.

#### D1. Control construction sensitivity

Tests whether the OR depends on how population-matched controls are generated.

| Variant | 60+ m/km OR | 95% CI |
|---------|-------------|--------|
| Baseline (land=3.0, ocean=0.05) | 5.18 | [4.48, 6.03] |
| Land-only, no ocean knob | 6.66 | [5.74, 7.79] |
| Equal weights (1.0 / 1.0) | 1.68 | [1.53, 1.87] |
| Mild (2.0 / 0.2) | 3.09 | [2.73, 3.53] |
| Extreme (10.0 / 0.01) | 5.78 | [5.02, 6.79] |
| 5-seed variance | mean = 5.50, std = 0.24 | — |

The effect is always present (OR > 1) but magnitude is sensitive to the land/ocean weighting parameter. The reviewer-requested "land only, no tuning knobs" variant gives the strongest result (OR = 6.66).

#### D2. Geocoding pileup diagnostics

NUFORC geocodes city/state to coordinates, creating artificial spatial clusters at city centroids.

- Top pileup: Seattle (540 reports at one coordinate), NYC (442), Phoenix (435)
- 68.1% of reports share coordinates with 5+ other reports
- 16,279 unique coordinate locations out of 66,678 reports (1990-2014)
- **After collapsing to unique events (0.01 deg + date-day):** OR = 5.10 [4.41, 5.94] — virtually unchanged

Pileups are real (city centroid geocoding) but do not drive the spatial signal, because pileup locations are at major metros, not specifically near steep bathymetry.

#### D3. Deduplication sensitivity

| Variant | 60+ m/km OR | 95% CI | N coastal |
|---------|-------------|--------|-----------|
| Original (coords + datetime) | 5.18 | [4.48, 6.03] | 38,256 |
| Strict (coords + date-day) | 5.10 | [4.45, 5.97] | 37,332 |
| Text hash (first 100 chars) | 5.17 | [4.44, 6.04] | 38,721 |
| Hard cap (1 per coord per day) | 5.10 | [4.45, 5.97] | 37,332 |

Less than 2% variation. The dedupe definition does not matter.

#### D4. Missingness / selection bias

- 80,331 of 80,332 raw records have valid coordinates (99.999%)
- Only 3.3% have low-precision coordinates (1 decimal place or less)
- This dataset is pre-geocoded at source; missingness is not a concern here

Note: the reviewer's merged multi-source dataset shows only 44% geocoding rate. This analysis uses single-source NUFORC data where geocoding is near-complete.

#### D5. Temporal clustering & seasonality

- Monthly report counts peak in summer (July: 4,365) with a winter trough (February: 2,443)
- Excluding any single month changes the 60+ OR by less than 0.1 (range: 5.13-5.21)
- Excluding all high-activity windows (Perseids, Leonids, Geminids, July 4, Halloween): OR = 5.16 [4.49, 6.16]

No seasonal factor drives the spatial signal.

#### D6. Replication & out-of-sample

| Variant | 60+ m/km OR | 95% CI | N coastal |
|---------|-------------|--------|-----------|
| Full sample | 5.18 | [4.48, 6.03] | 38,256 |
| **West Coast only** (lon < -115) | **6.21** | [5.39, 7.29] | 14,701 |
| **East/Gulf only** (lon >= -90) | **0.36** | [0.23, 0.68] | 21,258 |
| Hold out Puget Sound + SoCal | 0.87 | [0.71, 1.11] | 31,597 |
| Narrow coastal band (0-20 km) | 2.52 | [2.15, 2.91] | 18,234 |

**This is the most important result in Phase D.** The canyon-bathymetry association is entirely driven by the West Coast. The East/Gulf Coast shows a statistically significant effect in the *opposite direction*. Holding out just two regions (Puget Sound + Southern California) collapses the full-sample effect to null.

Possible interpretations:
- The West Coast has both the steepest submarine canyons (Monterey, Juan de Fuca) and the highest coastal UAP reporting density — these may be independently caused by population/military patterns rather than connected
- West Coast narrow continental shelf means steep canyons are closer to shore and to observers; East Coast wide shelf places canyons far offshore
- The result may reflect a genuine but geographically restricted phenomenon

**Gradient sanity check:** ETOPO resolution is 1 arc-minute (~1.85 km lat, ~1.48 km lon at 37N). Gradient computed as m/km with proper cos(lat) correction. 3x3 smoothing changes canyon cell count by only -4.8%.

## Robustness Summary

| Test | Result | Verdict |
|------|--------|---------|
| Primary permutation | 0/1,000 exceeded, p < 0.001 | Pass |
| Within-month null | p = 0.015 | Pass (seasonal control) |
| Cluster bootstrap (2,000) | beta = -0.17, CI [-0.26, -0.07] | Pass |
| Geocoding jitter (2-20 km) | stable | Pass |
| Pileup collapse (D2) | OR 5.18 -> 5.10 | Pass |
| Dedupe variants (D3) | all ~5.1 | Pass |
| Missingness (D4) | 99.999% geocoded | Pass (non-issue) |
| Month exclusion (D5) | max delta < 0.1 | Pass |
| High-activity exclusion (D5) | OR = 5.16 | Pass |
| Land-only controls (D1) | OR = 6.66 | Pass (stronger) |
| Equal-weight controls (D1) | OR = 1.68 | Weakened but >1 |
| **East/Gulf replication (D6)** | **OR = 0.36** | **Fail** |
| **Holdout Puget+SoCal (D6)** | **OR = 0.87** | **Fail** |
| Narrow coastal 0-20 km (D6) | OR = 2.52 | Pass (reduced) |

## Gradient-Canyon Overlap

| Gradient | n | Mean dist to canyon | % within 25 km | % within 50 km |
|----------|---|---------------------|-----------------|----------------|
| 60+ (very steep) | 5,514 | 12.3 km | 85.3% | 100% |
| 10-60 (mid) | 5,533 | 48.0 km | 49.2% | 80.0% |
| 0-10 (flat) | 27,209 | 148.9 km | 0.0% | 7.4% |

## Data Sources

All data files are included in this repository except ETOPO1 bathymetry (52 MB netCDF). See `data/sources.md` for full provenance and ETOPO1 download instructions.

- **NUFORC**: 80,332 sighting reports (38,256 coastal CONUS 1990-2014) — `data/nuforc_reports.csv`
- **ETOPO1**: NOAA 1-arc-minute bathymetry — download separately, see `data/sources.md`
- **Census**: 2020 decennial county population — `data/census_county_pop.json`
- **Military**: 171 DoD installations — `data/military_bases_us.csv`
- **Ports**: 7,747 OSM port/marina locations — `data/port_coords_cache.npz`

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.9+

## Methodology

1. Canyon cells identified from ETOPO1 bathymetry (gradient > 20 m/km on 0-500 m shelf)
2. Population-matched control points generated on coastal grid with county-centroid weighting
3. Logistic regression with iterative covariate addition (Sprint 1)
4. GAM, cluster bootstrap, geocoding jitter (Sprint 2)
5. Temporal permutation testing with within-month null (Sprint 3)
6. Importance-weighted binned OR with bootstrap CI (Scripts 04-05)
7. Gradient-canyon overlap validation (binary signal confirmation)
8. Robustness audit: 6-category stress-test of headline result (Phase D, Script 07)

## Audio walkthrough (optional)

A non-technical overview generated with NotebookLM:

[`media/UAP_Sightings_Cluster_Around_Submarine_Canyons.m4a`](media/UAP_Sightings_Cluster_Around_Submarine_Canyons.m4a)

## License

MIT
