# UAP-Canyon Spatial Association Analysis

Statistical analysis of the spatial relationship between UAP (Unidentified Anomalous Phenomena) reports from NUFORC and submarine canyon locations along the US coastline.

## Key Finding

UAP reports show a non-random spatial-temporal pattern concentrated at steep submarine canyon features (gradient >60 m/km), which overlap 85% with mapped canyon locations within 25 km.

- **Temporal clustering**: observed exceeds all 1,000 permutations (p < 0.001); survives within-month seasonal control (p = 0.015)
- **Binary canyon signal**: weighted OR = 3.90 [95% CI: 1.42, 10.83] at >60 m/km gradient; lower gradients show no reliable effect
- **GAM**: continuous distance decay with 7 covariates, strongest in first ~50 km
- **Cluster bootstrap** (2,000 resamples): canyon beta median = -0.17, 95% CI [-0.26, -0.07], excludes zero
- **Logistic regression** (Sprint 1): canyon distance beta = -0.166, p < 10^-56 (n = 61,985)

The effect is binary (canyon vs. non-canyon), not a smooth dose-response gradient.

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
│   ├── sprint1_model_comparison.png
│   ├── sprint1_sensitivity.png
│   ├── sprint1_stratified_or.png
│   ├── sprint2_gam_partial_dependence.png
│   ├── sprint2_bootstrap_distributions.png
│   ├── sprint2_jitter_stability.png
│   ├── sprint3_dose_response_bins.png
│   ├── sprint3_dose_response_gam.png
│   ├── sprint3_dose_response_stratified.png
│   ├── sprint3_dose_robustness_coastal_trend.png
│   ├── sprint3_dose_robustness_weighted_or.png
│   ├── sprint3_flap_map.png
│   ├── sprint3_temporal_permutation.png
│   └── sprint3_temporal_sensitivity.png
├── results/
│   ├── sprint1_results.json
│   ├── sprint2_results.json
│   ├── sprint3_results.json
│   ├── sprint3_dose_robustness.json
│   └── weighted_or_binned.json
├── notebooks/
│   ├── 01_sprint1_observer_controls.py
│   ├── 02_sprint2_continuous_model.py
│   ├── 03_sprint3_temporal_doseresponse.py
│   ├── 04_sprint3_dose_robustness.py
│   ├── 05_weighted_or_for_post.py
│   └── 06_generate_figures.py
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

## Sprint Overview

### Sprint 1: Observer Controls
Logistic regression with population-matched controls. Tests whether canyon proximity effect survives addition of port/marina covariates, military base distance, population density, and coastline complexity. Includes stratified odds ratios and sensitivity analysis. Note: Sprint 1 uses an additional `coast_complexity` feature that increases NaN-dropping, resulting in a smaller effective sample (n ≈ 23K) compared to Sprint 2+ (n ≈ 42K).

### Sprint 2: Continuous Model & Uncertainty
GAM partial dependence, cluster bootstrap (2,000 resamples), and geocoding jitter test (50 iterations per sigma level at 2/5/10/15/20 km). Confirms the canyon effect is not an artifact of binary discretization or spatial autocorrelation.

### Sprint 3: Temporal Clustering & Dose-Response
Spatial permutation testing (1,000 shuffles) for temporal clustering near canyons. Primary metric exceeds all permutations (p < 0.001). Within-month null controls for seasonality (p = 0.015). Robustness tests: trimmed mean, heavy-tail excess, ECDF quantile comparison.

### Dose-Response & Weighted OR (Scripts 04-05)
Importance-weighted odds ratios by bathymetric gradient bin, correcting for 60x land_weight asymmetry in control generation. Key result: only the steepest gradient bin (>60 m/km) shows a reliable effect (weighted OR = 3.90 [1.42, 10.83]). Post-hoc overlap analysis confirms 85% of >60 m/km locations are within 25 km of mapped canyons. Lower gradient bins (10-30, 30-60) show no reliable weighted effect — the signal is binary canyon/non-canyon.

## Robustness Summary

| Test | Result |
|------|--------|
| Primary permutation | 0/1,000 exceeded observed, p < 0.001 |
| Within-month null | p = 0.015, z = 4.0 (seasonal control) |
| Trimmed mean (5-95%) | p = 1.0, z = -5.3 (episodic, not diffuse) |
| Heavy-tail excess (ratio >= 2) | p = 1.0, z = -5.3 (confirms episodic) |
| Cluster bootstrap (2,000) | median beta = -0.17, CI [-0.26, -0.07] |
| Geocoding jitter (2–20 km) | effect stable across 50 iterations per sigma |
| GAM (7 covariates) | continuous distance decay confirmed |
| Importance-weighted OR | 3.90 [1.42, 10.83] at >60 m/km only |

## Gradient-Canyon Overlap

| Gradient | n | Mean dist to canyon | % within 25 km | % within 50 km |
|----------|---|---------------------|-----------------|----------------|
| 60+ (very steep) | 5,906 | 12.3 km | 85.3% | 100% |
| 10-60 (mid) | 6,063 | 48.0 km | 49.2% | 80.0% |
| 0-10 (flat) | 29,659 | 148.9 km | 0.0% | 7.4% |

## Data Sources

All data files are included in this repository except ETOPO1 bathymetry (52 MB netCDF). See `data/sources.md` for full provenance and ETOPO1 download instructions.

- **NUFORC**: 80,332 sighting reports (42,008 after coastal CONUS filter) — `data/nuforc_reports.csv`
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

1. Canyon cells identified from ETOPO1 bathymetry using gradient threshold
2. Population-matched control points generated per coastal county
3. Logistic regression with iterative covariate addition (Sprint 1)
4. GAM, cluster bootstrap, geocoding jitter (Sprint 2)
5. Temporal permutation testing with within-month null (Sprint 3)
6. Importance-weighted binned OR with bootstrap CI (Scripts 04-05)
7. Gradient-canyon overlap validation (binary signal confirmation)

## Audio walkthrough (optional)

A non-technical overview generated with NotebookLM:

🎧 [`media/UAP_Sightings_Cluster_Around_Submarine_Canyons.m4a`](media/UAP_Sightings_Cluster_Around_Submarine_Canyons.m4a)

## License

MIT
