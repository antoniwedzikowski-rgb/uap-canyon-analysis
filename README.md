# UAP-Canyon Spatial Association Analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19340933.svg)](https://doi.org/10.5281/zenodo.19340933)

Statistical analysis of the spatial relationship between UAP (Unidentified Anomalous Phenomena) reports from NUFORC and submarine canyon locations along the US coastline.

**Preprint:** Wędzikowski, A. (2026). Submarine Canyon Gradient Predicts UAP Report Density Along the US West Coast: A Geospatial Correlational Study. Zenodo. [doi:10.5281/zenodo.19340933](https://doi.org/10.5281/zenodo.19340933)

## Media Coverage

- **Daily Mail** (2025): [UFO hotspots and underwater canyons on the West Coast](https://www.dailymail.co.uk/sciencetech/article-15573389/ufo-hotspots-underwater-canyons-west-coast.html)

## Key Finding

UAP reports within 200 km of the US West Coast show elevated density near steep submarine canyon features (gradient >60 m/km), with a population-adjusted Spearman rho = 0.37 (p = 0.0001, n = 102 testable cells). The effect is primarily spatial (canyon cells have higher population-adjusted report rates). It replicates in temporally independent post-2014 data (rho = 0.35, p = 0.0001), though this replication shares the same city-centroid geocoding structure as the primary dataset (see Limitation #8). An out-of-country replication on Norwegian fjords was attempted but is inconclusive after population control (see Supplementary: Norway below).

**Critical limitations**: The effect is regional. It is not detected on the East/Gulf Coast (ρ = 0.055, p = 0.459, n_hot = 2 of 185 cells). The wider continental shelf places canyons far from shore, leaving insufficient contrast to test the hypothesis with land-based observer data (NUFORC). This is a testability limitation, not a falsification. The ESI shore type confound test is inconclusive (n = 18, California only, no Puget coverage). No independent out-of-country replication exists. This geographic asymmetry is the primary threat to a general bathymetric interpretation.

## Repository Structure

This release contains 19 Python scripts in `notebooks/` covering the full Phase E evaluation pipeline, confound tests, replication, and figure generation. Older exploratory and superseded scripts (Sprints 1-3, Phase D, earlier Phase E iterations) are retained in `archive/legacy_scripts/` for transparency.

```
uap-canyon-analysis/
├── data/                                    # Input datasets (see data/sources.md)
│   ├── nuforc_reports.csv                   # 80,332 NUFORC reports (included, 13 MB)
│   ├── census_county_pop.json               # 2020 county population (included)
│   ├── county_centroids_pop.csv             # County centroids + pop (included)
│   ├── military_bases_us.csv                # 171 DoD installations (included)
│   ├── port_coords_cache.npz               # 7,747 ports/marinas (included)
│   └── sources.md                           # Full data provenance documentation
│
├── notebooks/                               # Active analysis scripts (19 files)
│   ├── 10_phase_ev2_scoring.py              # Canyon scoring function (geometry only)
│   ├── 13_phase_e_red_v2.py                 # PRIMARY EVALUATION (haversine-corrected)
│   ├── 14-17b: Regional + confound tests    # Puget, band sweep, East Coast
│   ├── phase_e_*_confound.py                # Nested F-test confound analyses
│   ├── phase_e_replication_suite.py         # Temporal splits, LOO, post-2014
│   ├── phase_e_norway_*.py                  # Norway replication (Spearman + logistic)
│   ├── phase_e_threshold_sensitivity.py     # Threshold sweep (20-100 m/km)
│   └── generate_*.py                        # Figure generation scripts
│
├── results/                                 # JSON results + narrative summaries
│   ├── phase_ev2/                           # All Phase E v2 result JSONs (18 files)
│   │   ├── phase_e_red_v2_evaluation.json   # PRIMARY RESULT (rho=0.374, p=0.0001)
│   │   ├── phase_e_replication_suite.json   # Temporal + spatial replication
│   │   ├── phase_e_*_confound.json          # All confound test results
│   │   └── ...
│   ├── phase_d_results.json                 # Phase D robustness results
│   └── PHASE_E_SUMMARY.md                   # Phase E narrative (contains pre-audit note)
│
├── figures/                                 # Publication figures (PNG)
│
├── archive/                                 # Non-canonical materials (retained for transparency)
│   ├── legacy_scripts/                      # Sprints 1-3, Phase D, superseded Phase E scripts
│   ├── exploratory_results/                 # Sprint results, Phase E v1 outputs
│   ├── exploratory_figures/                 # Sprint-era and diagnostic figures
│   ├── communication/                       # MEDIA_STATEMENT.md, SUMMARY.md, docs/ landing page
│   ├── drafts/                              # Earlier manuscript drafts
│   ├── prompts/                             # LLM prompts used during development
│   └── qa/                                  # QA task logs and revision memos
│
├── preprint.md                              # Manuscript (Markdown source)
├── preprint.pdf                             # Manuscript (PDF)
├── STATISTICAL_AUDIT_REPORT.md              # 47-finding statistical audit
└── requirements.txt                         # Python dependencies
```

**External data required:** ETOPO1 bathymetry (~52 MB netCDF) must be downloaded separately — see `data/sources.md` for instructions. Some confound scripts also fetch data at runtime (MODIS chl-a, OPAREA polygons). See `data/sources.md` for the full list of included vs. external datasets.

## Analysis Phases

### Phase B: Initial Hypothesis Test

Tests whether UAP reports cluster near submarine canyons.

- **B v1**: Full-ocean analysis. Result: UAP are *farther* from canyons (OR = 0.48). Canyon defined as 95th-percentile ocean gradient.
- **B v2**: Coastal shelf refinement (200 km band, county-level population, 20 m/km gradient). Result: UAP are *closer* (OR = 5.30 at 10 km). Direction reversal reflects the change from deep-ocean to shelf-canyon definition.
- **B v2 robustness**: Leave-one-out metro, 100 placebo runs, distance-matched bands. Effect survives but is confined to the 0-25 km coastal band.

### Phase C: Data Calibration & Discriminating Tests

NUFORC data quality assessment and hypothesis discrimination.

- **C1**: Night fraction (80.5%), meteor shower amplification (1.10x), media event detection (Phoenix Lights 7.5x), 65 NPPs, 72k earthquakes
- **C2**: Discriminating tests between Natural, Technology, and Agency hypotheses. Verdict: **Agency**
- **C3**: Adaptation tests (temporal evolution of sighting characteristics). Verdict: **WEAK_AGENCY** (1/6 adaptation signals)

### Sprints 1-3: Core Statistical Analysis

- **Sprint 1**: Logistic regression with population-matched controls. Canyon proximity effect survives port/military/population covariates.
- **Sprint 2**: GAM partial dependence, cluster bootstrap (2,000 resamples), geocoding jitter test. Effect is not an artifact of binary discretization.
- **Sprint 3**: Temporal permutation testing (1,000 shuffles), importance-weighted OR by gradient bin. Only the 60+ m/km bin shows reliable effect. **Note**: Sprint 3 temporal clustering result (p = 0.001) was CONUS-wide; does not replicate when restricted to West Coast (p = 1.0 for 1990-2014, p = 0.18 for post-2014). Not included in paper.

### Phase D: Robustness Audit

Six-category stress-test of the headline result on 38,256 coastal reports (1990-2014 CONUS):

| Test | Result | Verdict |
|------|--------|---------|
| D1: Control sensitivity | OR range 1.68-6.66 depending on weights | Sensitive |
| D2: Geocoding pileup | OR 5.18 -> 5.10 after collapse | Pass |
| D3: Deduplication | All variants ~5.1 | Pass |
| D4: Missingness | 99.999% geocoded | Non-issue |
| D5: Seasonality | No month drives effect | Pass |
| D6: West Coast only | OR = 6.21 | Pass |
| **D6: East/Gulf Coast** | **ρ = 0.055, p = 0.459 (not detected; n_hot = 2/185)** | **Fail — testability** |
| **D6: Holdout Puget+SoCal** | **OR = 0.87 (null)** | **Fail** |

**Phase D extensions:**
- **D7**: Canyon anatomy tests (named canyon verification, canyon size effects)
- **D7d**: Within-West-Coast variation analysis
- **D8**: Puget Sound deep-dive (population density, military proximity, reporting patterns)

### Phase E: Out-of-Sample Prediction

Pre-registered geometric scoring function frozen before evaluation.

**Design evolution:**
1. **E v1** (20 m/km threshold): Top-20 hotspots fell outside NUFORC footprint. Non-measurable.
2. **E v2** (60 m/km, aligned with Phase C/D estimand): Primary result.
3. **E-RED v2** (haversine-corrected, population-adjusted): Definitive evaluation.

**Primary result (E-RED v2, West Coast 200 km):**
- Spearman rho = 0.374, p = 0.0001, n = 102 testable cells
- 26 hot cells, 76 cold cells
- Bootstrap 95% CI: [0.190, 0.531]
- exp(beta) = 1.93 (OLS log-linear proxy)

**Regional breakdown (O/E rate by canyon presence):**

| Region | S=0 rate | S>0 rate | Uplift | n (S>0 / S=0) |
|--------|----------|----------|--------|----------------|
| Puget Sound (46-50N) | 0.74 | 5.04 | 6.8x | 11 / 11 |
| San Diego (32-33.5N) | 0.60 | 5.85 | 9.8x | 3 / 2 |
| Monterey Bay (36-37N) | 1.21 | 3.18 | 2.6x | 2 / 8 |
| Rest of West Coast | 1.06 | 1.46 | 1.4x | 10 / 55 |

The effect concentrates in regions with extreme near-shore submarine topography, with an identical pattern: S=0 suppression below baseline + S>0 uplift. San Diego has only n=5 cells, too few for independent statistical test, but the LOO SoCal fold (rho = 0.49, p = 0.024, n = 21) is significant.

Monterey Bay provides one of the cleanest local examples in the present dataset. Its two canyon cells have intermediate steepness scores (S = 1.35–1.61), elevated report ratios (R = 2.75 and 4.80), and lie 127–141 km from the nearest mapped Navy operating area. This makes Monterey a strong local case for the canyon association without turning it into a stand-alone causal test.

**Non-linearity:** The relationship is non-linear. Quintile analysis shows Q1–Q4 of S have similar mean logR (0.20–0.46, overlapping CIs), while Q5 jumps to 1.40. The effect concentrates in the highest quintile (S > ~1.3), corresponding to cells with extreme near-shore canyon topography. The Spearman rho = 0.37 captures a real monotonic trend but understates the threshold-like character of the association.

**Coastal band dependence:** The effect peaks at 50 km (rho = 0.43, p = 0.0005), is non-significant at 10 km (rho = 0.15, p = 0.37), and stabilizes at 100–200 km (rho = 0.37). This argues against a direct line-of-sight coastal observation mechanism and suggests a broader coastal-zone phenomenon.

**Confound tests:**

| Confound | Method | Result | Verdict |
|----------|--------|--------|---------|
| Ocean depth | Nested F-test | S dominant over depth | S_DOMINANT |
| Magnetic anomaly | Nested F-test | S dominates; canyon cells have *lower* anomalies | S_DOMINANT |
| ESI shore type | Nested F-test (n=18, CA only) | All F-tests non-sig (p > 0.39); no Puget coverage | INCONCLUSIVE (underpowered) |
| Shore type proxy (ETOPO cliff) | Nested F-test (n=102) | S survives cliff control (p=0.004); cliff also independent (p=0.019); R²=0.21 combined | S_SURVIVES (both contribute) |
| Puget Sound | 2x2 rate interaction | S=0 Puget rate (0.53) not elevated vs elsewhere (1.08) | NO_CONFOUND (canyon-specific) |
| Coastal upwelling (chl-a) | Nested F-test (n=99) | S adds to chl-a: F=18.5, p<0.0001; chl-a uncorrelated with S (rho=-0.02); S_DOM at all radii (50-200km) | S_DOMINANT |
| Military OPAREAs (polygons) | Regional nested F-test (35 OPAREA polygons, NOAA MarineCadastre) | S dominant in Puget (p=0.018 vs p=0.10); Central CA marginal (p=0.056); Monterey Bay remains informative because its two canyon cells lie 127-141 km from the nearest mapped OPAREA and still show elevated report ratios (R = 2.75 and 4.80); SoCal uninformative (OPAREA boundary = coastline) | S_SURVIVES_REGIONALLY |

**Replication (corrected — population-adjusted E_i, per-fold normalization):**

| Test | rho | p | Status |
|------|-----|---|--------|
| Full dataset baseline | 0.338 | 0.0005 | Pass |
| Temporal split (1990-2002) | 0.319 | 0.002 | Pass |
| Temporal split (2003-2014) | 0.361 | 0.0003 | Pass |
| Post-2014 replication | 0.350 | 0.0001 | Pass |
| LOO spatial CV | mean 0.315 | 4/5 positive, 2/5 sig | Partial |
| Spatial forward: Puget | 0.604 | 0.017 | Pass |
| Spatial forward: SoCal | 0.470 | 0.049 | Pass |
| 5-year rolling windows | 21/21 positive | 18/21 sig | Pass |
| Threshold sweep (20-100 m/km) | 0.374-0.416 | all < 0.0002 | Pass |
| Norway fjord replication | — | — | Inconclusive (see Supplementary below) |

See `results/PHASE_E_SUMMARY.md` for the complete 600-line Phase E narrative.

See `STATISTICAL_AUDIT_REPORT.md` for the full 47-finding audit and resolution status.

## Known Issues (from Statistical Audit)

**Resolved (post-audit fixes):**

| # | Issue | Resolution |
|---|-------|------------|
| 1 | Replication suite used uniform E_i (pop defaults to 1) | FIXED — full IDW population model imported; rho improved from 0.28 to 0.35 (post-2014) |
| 2 | LOO CV used logR from full-data normalization (leakage) | FIXED — per-fold E_i normalization; LOO mean rho dropped from 0.38 to 0.315 (still positive) |
| 3 | Spatial forward prediction used leaked logR | FIXED — per-fold recomputation; Puget rho=0.604, SoCal rho=0.470 (both sig) |
| 4 | 60 m/km threshold selected from data (double-dipping) | RESOLVED — threshold sweep (20-100 m/km) shows all 8 thresholds significant (rho 0.37-0.42) |

**Remaining (documented as limitations):**

| # | Issue | Severity | Location |
|---|-------|----------|----------|
| 5 | cKDTree degree-space distance (up to 40% error at 48N) | MEDIUM | phase_ev2_scoring.py |
| 6 | Land/ocean weighting 60:1 ratio arbitrary, no sensitivity | MEDIUM | phase_e_red_v2.py |
| 7 | Phase D still uses deg*111 approximation (not haversine) | MEDIUM | phase_d_robustness.py |
| 8 | Post-2014 geocoding assigns identical coords per city | MEDIUM | phase_e_replication_suite.py |
| 9 | min_reports threshold varies (20, 10, or 5) across tests | MEDIUM | phase_e_replication_suite.py |
| 10 | Oregon S=0 regional excess (O/E = 2.09 vs 0.91 elsewhere, p ≈ 0.03) | MEDIUM | E_i population model |

Oregon S=0 cells report at 2.1× expected rate, indicating a Pacific Northwest cultural factor not captured by the population model. That Puget Sound S=0 cells within this same region show suppressed rates (0.74×) strengthens the canyon-specific interpretation: the regional excess does not extend to non-canyon cells in Puget Sound.

## Data Sources

Core datasets for the primary pipeline are included in `data/`. Confound tests and international replication require external data downloaded separately or fetched at runtime. See `data/sources.md` for full provenance and download instructions.

**Included in this release:**

| Dataset | File | Records | Source |
|---------|------|---------|--------|
| NUFORC reports | `nuforc_reports.csv` | 80,332 | NUFORC (1990-2014) |
| Population | `census_county_pop.json` | 3,108 counties | US Census 2020 |
| County centroids | `county_centroids_pop.csv` | 3,108 | US Census |
| Military bases | `military_bases_us.csv` | 171 | DoD |
| Ports | `port_coords_cache.npz` | 7,747 | OpenStreetMap |

**External (not included — see `data/sources.md` for download instructions):**

| Dataset | Required by | Source |
|---------|-------------|--------|
| ETOPO1 bathymetry (`etopo_subset.nc`, 52 MB) | Core pipeline (scoring + evaluation) | NOAA |
| Post-2014 NUFORC (`nuforc_post2014.csv`) | `phase_e_replication_suite.py` | HuggingFace kcimc/NUFORC |
| EMAG2v3 magnetic anomaly | `phase_e_magnetic_confound.py` | NOAA |
| MODIS chlorophyll-a | `phase_e_chla_confound.py` (fetched at runtime) | NASA ERDDAP |
| ESI shoreline classification | `phase_e_esi_shoretype.py` | NOAA ESI |
| Navy OPAREA polygons | `phase_e_oparea_confound.py` (fetched at runtime) | NOAA MarineCadastre |
| SRTM30 Norway bathymetry | `phase_e_norway_*.py` | SRTM30 |

## Methodology

1. **Canyon detection**: ETOPO1 gradient magnitude (m/km with cos(lat) correction), threshold 60 m/km on continental shelf (0 to -500 m), minimum component size 3 cells
2. **Population control**: 0.5-degree grid, IDW weighting from k=10 nearest county centroids, land/ocean weighting (3.0/0.05), haversine-corrected coastal filtering
3. **Scoring function**: S = mean(rank_G + rank_P + rank_C) for steep cells within 50 km. G = p95 gradient, P = shore proximity (exponential decay), C = coastal complexity
4. **Primary test**: Spearman correlation between S and log(O_i/E_i) across 0.5-degree cells with 20+ reports
5. **Confound testing**: Nested F-tests comparing full model (S + confound) vs reduced models
6. **Replication**: Temporal splits, leave-one-region-out CV, post-2014 temporally independent data (shared geocoding), Norway out-of-sample

## Supplementary: Norway Fjord Replication

Attempted out-of-country replication on Norwegian fjords (SRTM30 bathymetry, WorldPop population, 40 NUFORC reports in 463 coastal cells). Included for transparency.

**Original test** (Spearman on 17 cells with reports): rho = 0.49, p = 0.047. However, all 17 testable cells have S > 0 — Norway's entire coastline is fjords, leaving no flat-shelf control group.

**Population-controlled retest** (logistic regression on 227 cells with pop > 0):

| Test | Statistic | p | Interpretation |
|------|-----------|---|----------------|
| Logistic: S + log(pop) | S coef = 0.03, OR = 1.03 | 0.76 | S adds nothing after pop control |
| LR test (S added to pop-only) | chi2 = 0.09 | 0.76 | Pop-only model sufficient |
| Population alone | log(pop) coef = 0.61 | 0.0001 | Reports track population, not canyons |
| Full-sample Spearman (n=227) | rho = 0.07 | 0.32 | No correlation including zero-report cells |
| Binary Fisher (S>0 vs S=0) | 17/220 vs 0/7 | 0.58 | Only 7 S=0 cells — test underpowered |

**Verdict**: Inconclusive. The original rho = 0.49 is explained by population (larger fjord cities sit near steeper gradients). With only 7 flat-shelf cells, Norway cannot provide a meaningful canyon-vs-flat contrast. Scripts: `phase_e_norway_replication.py` (original), `phase_e_norway_logistic.py` (population-controlled).

## Quick Start

Reproduce the headline result in under 5 minutes:

```bash
# 1. Clone and install
git clone https://github.com/antoniwedzikowski-rgb/uap-canyon-analysis.git
cd uap-canyon-analysis
pip install -r requirements.txt

# 2. Download ETOPO1 bathymetry (~52 MB) — the only file not included
#    Get ETOPO1_Bed_g_gmt4.grd.gz from:
#    https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/bedrock/grid_registered/netcdf/
#    Then subset it:
python -c "
import xarray as xr
ds = xr.open_dataset('ETOPO1_Bed_g_gmt4.grd')
subset = ds.sel(lat=slice(20, 55), lon=slice(-135, -55))
subset.to_netcdf('data/etopo_subset.nc')
"

# 3. Run the scoring function (geometry only — no UAP data touched)
python notebooks/10_phase_ev2_scoring.py

# 4. Run the primary evaluation (produces Spearman rho = 0.374, p = 0.0001)
python notebooks/13_phase_e_red_v2.py
```

The included data files (`data/nuforc_reports.csv`, `data/census_county_pop.json`, `data/military_bases_us.csv`, `data/port_coords_cache.npz`) are all you need beyond ETOPO1. See `data/sources.md` for full provenance.

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.9+. Additional packages for specific analyses: `rasterio` (magnetic confound), `ephem` (night detection in Phase C).

## Audio Walkthrough

A non-technical overview of the full analysis (Phases B–E), generated with NotebookLM:

[`media/UAPs_Cluster_Over_Steep_Underwater_Canyons.m4a`](media/UAPs_Cluster_Over_Steep_Underwater_Canyons.m4a)

## Author

**Antoni Wędzikowski**, independent researcher, lawyer and legaltech founder, Warsaw, Poland
[LinkedIn](https://www.linkedin.com/in/antekwedzikowski/) · [GitHub](https://github.com/antoniwedzikowski-rgb/uap-canyon-analysis)

## License

MIT
