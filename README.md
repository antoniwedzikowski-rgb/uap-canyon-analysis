# UAP-Canyon Spatial Association Analysis

Statistical analysis of the spatial relationship between UAP (Unidentified Anomalous Phenomena) reports from NUFORC and submarine canyon locations along the US coastline.

## Key Finding

UAP reports cluster significantly closer to submarine canyons than population-matched control points, even after controlling for coastline proximity, military bases, population density, ocean depth, and port/marina infrastructure.

- **Logistic regression**: canyon distance beta = -0.166, p < 10^-56 (n = 61,985)
- **GAM confirmation**: nonlinear effect consistent with logistic model
- **Cluster bootstrap** (200 iterations): beta 95% CI [-0.208, -0.127], p < 0.001
- **Geocoding jitter** (100 iterations): effect stable across +/-5 km perturbation

## Project Structure

```
uap-canyon-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── figures/
│   ├── sprint1_model_comparison.png
│   ├── sprint1_sensitivity.png
│   ├── sprint1_stratified_or.png
│   ├── sprint2_gam_partial_dependence.png
│   ├── sprint2_bootstrap_distributions.png
│   └── sprint2_jitter_stability.png
├── results/
│   ├── sprint1_results.json
│   └── sprint2_results.json
├── notebooks/
│   ├── 01_sprint1_observer_controls.py
│   ├── 02_sprint2_continuous_model.py
│   └── 03_sprint3_temporal_doseresponse.py
├── prompts/
│   ├── sprint1_prompt.md
│   ├── sprint2_prompt.md
│   └── sprint3_prompt.md
└── data/
    └── sources.md
```

## Sprint Overview

### Sprint 1: Observer Controls
Logistic regression with population-matched controls. Tests whether canyon proximity effect survives addition of port/marina covariates, military base distance, and population density. Includes multiple imputation, stratified odds ratios, and sensitivity analysis.

### Sprint 2: Continuous Model & Uncertainty
GAM partial dependence, cluster bootstrap (200 iterations), and geocoding jitter test (100 iterations at +/-5 km). Confirms the canyon effect is not an artifact of binary discretization or spatial autocorrelation.

### Sprint 3: Temporal Clustering & Dose-Response
*(In progress)* Temporal event clustering near canyons and dose-response relationship between canyon proximity and report density.

## Data Sources

Raw data files are not included in this repository due to size. See `data/sources.md` for download links and instructions.

- **NUFORC**: National UFO Reporting Center sighting reports
- **ETOPO1**: NOAA bathymetry data for canyon identification
- **Census**: US county population data for density controls
- **OSM**: Port and marina locations via Overpass API

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.9+

## Methodology

1. Canyon cells identified from ETOPO1 bathymetry using gradient threshold (>20 degrees)
2. Population-matched control points generated per coastal county
3. Logistic regression with iterative covariate addition
4. Robustness: GAM, cluster bootstrap, geocoding jitter, stratified analysis

## License

MIT
