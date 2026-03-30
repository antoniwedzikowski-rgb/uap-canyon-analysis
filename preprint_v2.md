# Submarine Canyon Gradient Predicts UAP Report Density Along the US West Coast: A Geospatial Correlational Study

**Antoni Wędzikowski**
Independent researcher, Warsaw, Poland

**Preprint v2** — revised 2026-03-30

---

## Abstract

This study tests whether Unidentified Anomalous Phenomena (UAP) report density correlates with submarine canyon steepness along the US coastline. A pre-specified canyon scoring function was evaluated against population-adjusted report rates (80,332 geocoded National UFO Reporting Center sighting reports, 1990–2014; ETOPO bathymetry; inverse-distance-weighted population controls) across 102 testable 0.5° grid cells on the West Coast. Canyon cells receive roughly twice the reports expected from population alone (Spearman ρ = 0.374, p = 0.0001, 95% CI [0.190, 0.531]; aggregate observed/expected 2.69 vs. 1.02 for flat-shelf cells). The association concentrates where canyons approach the shore — Puget Sound (6.8× uplift), La Jolla/San Diego (9.8×), Monterey Bay (2.75–4.80×, 2 cells) — and exhibits threshold-like non-linearity. Nested confound tests against ocean depth, magnetic anomalies, chlorophyll-a, and military operating area proximity did not eliminate the association in regions where those tests are informative. A post-2014 temporal holdout (ρ = 0.350, n = 119) confirms persistence, though shared city-centroid geocoding limits its independence. The association is absent on the East/Gulf Coast (ρ = 0.055, p = 0.459) and inconclusive in Norway. The 60 m/km canyon gradient threshold was selected during exploratory analysis; a post-hoc sweep (20–100 m/km) shows all thresholds significant. The result is a regional spatial correlation, not a causal mechanism. Code, data, figures, and a 47-finding statistical audit are available at the project repository.

**Keywords:** UAP, submarine canyons, geospatial analysis, NUFORC, bathymetry, spatial correlation

---

## 1. Introduction

The spatial distribution of Unidentified Anomalous Phenomena (UAP) reports across the United States has received limited quantitative attention. Medina et al. (2023) showed that reported sightings concentrate in the western US, with light pollution, cloud cover, and military proximity partially explaining the pattern. Their analysis left unexplained why the West Coast generates disproportionately more reports than the East Coast, even after controlling for population density and observational conditions.

Several high-profile incidents — notably the 2004 Nimitz encounter off Southern California — and broader calls for ocean–UAP investigation (Gallaudet, 2024) point to a marine component in the pattern. Submarine canyons are a testable candidate: steep shelf incisions that sometimes reach within meters of shore (Allen & Durrieu de Madron, 2009), they underlie several well-known reporting hotspots — Puget Sound, La Jolla, Monterey Bay. This observation motivated the hypothesis but does not constitute evidence for it; no published analysis has tested the association.

This study tests whether population-adjusted UAP report density correlates with submarine canyon steepness along the US coastline, using nested confound tests, temporal replication, and a transparent audit trail. The answer is conditionally affirmative for the West Coast, absent on the East Coast, and subject to caveats detailed throughout.


## 2. Data

### 2.1 UAP Reports

The primary dataset consists of 80,332 geocoded UAP sighting reports from the National UFO Reporting Center (NUFORC), spanning 1990–2014. NUFORC is a volunteer-operated database accepting self-reported sightings via web form and telephone. Reports include date, time, location (city/state, with latitude and longitude derived via geocoding), duration, shape description, and a free-text narrative. The database is publicly available and has been used in prior geospatial analyses (Medina et al., 2023).

For temporal holdout testing, a post-2014 dataset (~50,000 reports, 2015–2023) was obtained from a HuggingFace mirror of NUFORC data (kcimc/NUFORC). This dataset was not used during model development or threshold selection. However, the post-2014 data is geocoded via the same city/state lookup as the primary dataset, meaning reports from a given city receive identical coordinates regardless of era. This shared geocoding structure means the temporal holdout inherits the spatial distribution of the primary dataset by construction, limiting its value as an independent spatial replication (see Section 6).

**Filtering.** Reports were restricted to the contiguous United States (CONUS), within a 200 km coastal buffer measured from the ETOPO shoreline. This yielded approximately 42,000 reports for the primary analysis (38,256 in the 1990–2014 training window). Inland reports were excluded as the hypothesis concerns submarine terrain.

**Known limitations.** NUFORC data is self-reported, unverified, and subject to reporting biases including population density, media influence, and cultural attitudes toward UAP reporting. It is an opportunistic observational dataset, not a systematic survey. Geocoding to city centroids introduces spatial imprecision, particularly in large counties where the centroid may be far from the actual observation site. The geocoding pileup effect was tested and found not to materially affect the primary result (OR changed from 5.18 to 5.10 after collapsing duplicate coordinates).

### 2.2 Bathymetry

Submarine topography was derived from the NOAA ETOPO 2022 dataset at 1 arc-minute resolution (~1.85 km). Gradient magnitude was computed for each ocean cell using central finite differences with cos(latitude) correction:

    G = sqrt((dz/dx)² + (dz/dy)²)   [m/km]

where z is ocean depth and x, y are east–west and north–south distances respectively.

Canyon features were identified as connected components of cells where G > 60 m/km on the continental shelf (depth 0 to −500 m), with a minimum component size of 3 cells. The 60 m/km threshold was selected during exploratory analysis (Sprint 3 dose-response binning) and carried forward to the evaluation phase. A post-hoc sensitivity sweep across 20–100 m/km confirmed that all tested thresholds produce significant associations (Section 4.9), but the threshold selection process had an exploratory component that should be noted when interpreting the results.

### 2.3 Population Data

Population density was estimated at each 0.5° grid cell using inverse distance weighting (IDW) from the k = 10 nearest county centroids (US Census 2020, 3,108 counties). A land/ocean weighting ratio of 3.0:0.05 was applied to prevent inland population centers from dominating coastal cell estimates. Expected report counts (E_i) for each cell were computed by distributing total coastal reports proportionally to population weight, enabling computation of observed/expected ratios. The county-centroid population proxy is coarse; it does not capture within-county population distribution along the coast (e.g., San Diego County's centroid is inland).

### 2.4 Confound Datasets

The following additional datasets were used for confound testing:

| Dataset | Source | Records/Coverage |
|---------|--------|------------------|
| Military installations | Department of Defense | 171 bases |
| Ports and marinas | OpenStreetMap | 7,747 locations |
| Navy Operational Areas | NOAA MarineCadastre | 35 polygons |
| Ocean depth | ETOPO 2022 | Per-cell mean |
| Magnetic anomaly | NOAA EMAG2v3 | Global grid |
| Chlorophyll-a | MODIS Aqua | Annual mean |
| ESI shoreline type | NOAA ESI | California only (n = 18) |


## 3. Methods

### 3.1 Spatial Framework

The US coastline was divided into a 0.5° latitude × 0.5° longitude grid. Each cell was classified as testable if it (a) contained at least 20 UAP reports and (b) fell within the 200 km coastal buffer. Each cell covers roughly 55 km north–south by 37–46 km east–west on the US West Coast (approximately 2,000–2,500 km² depending on latitude). This yielded 102 testable cells on the West Coast and a comparable number on the East/Gulf Coast.

### 3.2 Canyon Scoring Function

For each grid cell, a continuous canyon score S was computed based on the submarine terrain within a 50 km radius:

    S_i = mean(rank(G_p95) + rank(P) + rank(C))

where G_p95 is the 95th-percentile gradient magnitude among steep shelf cells within 50 km, P is shore proximity (exponential decay from nearest canyon cell to coast), and C is coastal complexity (number of distinct canyon components). All components were rank-transformed before averaging.

The scoring function was frozen before evaluation against UAP data. Only geometric bathymetric inputs were used; no UAP information entered the scoring function. The spatial distance computations used a cKDTree in degree-space, which introduces up to ~40% error in east-west distances at 48°N latitude (Puget Sound). This is a known approximation that biases against finding the effect at high latitudes.

### 3.3 Primary Test Statistic

The primary test is a Spearman rank correlation between S_i and log(O_i / E_i) across testable cells, where O_i is the observed report count and E_i is the population-expected count. The log ratio (logR) was used to normalize the heavily right-skewed distribution of report counts. Of the 102 testable cells, 76 have S = 0 and 26 have S > 0, meaning the Spearman correlation is primarily driven by the contrast between canyon and flat-shelf cells rather than by fine-grained dose-response within the S > 0 group.

### 3.4 Confound Testing Framework

Each candidate confound variable V was evaluated using nested F-tests:

1. Does S add beyond V? Compare model logR ~ V versus logR ~ V + S.
2. Does V add beyond S? Compare model logR ~ S versus logR ~ S + V.

If S adds significantly (p < 0.05) beyond V but V does not add beyond S, the verdict is S_DOMINANT. If both add independently, the verdict is INDEPENDENT. If V explains away S, the verdict is CONFOUND. This framework was applied to ocean depth, magnetic anomaly, chlorophyll-a, ESI shore type, and Navy OPAREA distance.

### 3.5 Temporal Replication

Three temporal replication strategies were employed:

1. **Split-half**: The dataset was split temporally (1990–2002 vs. 2003–2014, and 1990–2006 vs. 2007–2014). Even/odd year splits provided an additional non-temporal control. Each half was tested independently with recomputed E_i.
2. **Rolling windows**: 5-year sliding windows from 1990–1994 through 2010–2014 (21 windows total).
3. **Temporal holdout**: Post-2014 data (2015–2023) was evaluated against the scoring function developed entirely on pre-2014 data. Population expectations (E_i) were recomputed within the holdout window. This holdout shares the same city-centroid geocoding as the primary dataset, limiting its independence to the temporal dimension only.

### 3.6 Spatial Replication

Leave-one-region-out cross-validation was performed across 5 geographic folds (Pacific Northwest, Northern California, Central California, Southern California, and a residual fold). Each fold recomputed E_i within the training set to prevent normalization leakage — an error identified and corrected during the statistical audit (Appendix A).

### 3.7 Regional Decomposition

Given the finding that the East Coast shows no effect, all results are reported separately for the West Coast and East/Gulf Coast. Within the West Coast, four sub-regions were analyzed:

| Region | Latitude range | n cells | Key features |
|--------|---------------|---------|--------------|
| Puget Sound | ≥46°N | 22 | Deep fjord-like inlets, Admiralty Inlet |
| Oregon/Washington | 42–46°N | 16 | Moderate shelf, limited canyon exposure |
| Central California | 35–42°N | 43 | Monterey Canyon, Sur Canyon |
| Southern California | 32–35°N | 21 | La Jolla/Scripps Canyon, Mugu Canyon |

### 3.8 Navy OPAREA Analysis

Proximity to 35 publicly available Navy Operational Area polygons (NOAA MarineCadastre) was tested as a confound using minimum haversine distance from cell centroid to polygon boundary. Regional decomposition was necessary because the SOCAL Range Complex boundary tracks the San Diego coastline (118 vertices along the shore from Point Loma to Camp Pendleton), rendering the distance metric functionally equivalent to distance-from-shore in Southern California (Spearman correlation between OPAREA distance and longitude = +0.58, p = 0.006 in SoCal cells). In Puget Sound and Central California, where OPAREA boundaries do not track the coastline, S and OPAREA distance can be separated.

### 3.9 Software and Reproducibility

All analyses were conducted in Python 3.9+ using NumPy, SciPy, scikit-learn, and custom scripts. The AI assistant Claude (Anthropic) was used for code generation and statistical design. All analytical decisions and domain interpretations are the author's. The project repository contains the analysis scripts, figures, results JSON files, and a 47-finding statistical audit report. See Section 7 for data and code availability details.


## 4. Results

### 4.1 Primary Association

The pre-specified canyon scoring function S showed a significant positive association with population-adjusted UAP report density across 102 testable West Coast cells (Spearman ρ = 0.374, p = 0.0001, bootstrap 95% CI [0.190, 0.531], 2,000 cluster resamples). Cells with higher canyon scores — reflecting steeper near-shore submarine gradients, greater proximity to shore, and higher coastal complexity — received more UAP reports than expected from their population alone. An OLS log-linear proxy yielded exp(β) = 1.93, indicating that a one-unit increase in S approximately doubles the expected report rate.

Of the 102 testable cells, 26 were classified as canyon-proximate (S > 0) and 76 as flat-shelf (S = 0). Canyon cells showed an aggregate observed/expected ratio of 2.69 (ΣO_i / ΣE_i), compared to 1.02 for flat-shelf cells.

**[Figure 2 about here]**

### 4.2 Non-Linearity and Threshold Character

The association is not smoothly linear. A quintile decomposition of S among cells with S > 0 reveals that the four lower quintiles (Q1–Q4) show similar mean log(O/E) values (0.20–0.46, with overlapping confidence intervals), while the top quintile (Q5, corresponding to S > ~1.3) jumps to a mean log(O/E) of 1.40. The Spearman ρ = 0.37 captures a real monotonic trend but understates the threshold-like character of the association: the effect concentrates among cells with extreme near-shore canyon topography.

### 4.3 Regional Decomposition

The association is not uniformly distributed along the West Coast. It concentrates in areas with extreme near-shore bathymetry:

| Region | Lat. range | S=0 rate (O/E) | S>0 rate (O/E) | Uplift | n (S>0 / S=0) |
|---|---|---|---|---|---|
| Puget Sound | 46–50°N | 0.74 | 5.04 | 6.8× | 11 / 11 |
| San Diego | 32–33.5°N | 0.60 | 5.85 | 9.8× | 3 / 2 |
| Monterey Bay | 36–37°N | 1.21 | 3.18 | 2.6× | 2 / 8 |
| Rest of West Coast | — | 1.06 | 1.46 | 1.4× | 10 / 55 |

The pattern is consistent across the three hotspot regions: flat-shelf cells show suppressed or baseline reporting rates, while adjacent canyon cells show pronounced elevation. San Diego shows the largest per-cell uplift but with only 5 cells total, precluding an independent statistical test at the sub-regional level.

Monterey Bay (36–37°N) contains two canyon cells, both elevated above baseline (R = 2.75 and 4.80). The bay's eight adjacent flat-shelf cells report near baseline in aggregate (ΣO_i / ΣE_i = 1.21). Both canyon cells lie 127–141 km from the nearest mapped Navy operating area, making Monterey one of the clearest local examples of the canyon–density association in a region with minimal military overlap.

**Sensitivity to Puget Sound.** Excluding all Puget Sound cells (46–50°N) from the primary analysis yields ρ = 0.243 (p = 0.021), which remains positive but is reduced in magnitude. Puget Sound is a major driver of the overall result, though the residual association persists in the remaining West Coast data.

**Oregon regional excess.** Oregon flat-shelf cells (S = 0) report at 2.09× expected, significantly above baseline (p ≈ 0.03). This suggests a Pacific Northwest cultural or geographic reporting factor not captured by the population model. Puget Sound S = 0 cells within the same broader region show suppressed rates (0.74×), indicating the regional excess does not extend uniformly.

**[Figure 1 about here]**

### 4.4 Coastal Band Dependence

The strength of the association varies with the width of the coastal analysis buffer.

| Band (km) | n cells | Spearman ρ | p-value |
|-----------|---------|------------|---------|
| 10 | 39 | 0.149 | 0.36 (NS) |
| 25 | 50 | 0.348 | 0.013 |
| 50 | 62 | 0.430 | 0.0005 |
| 100 | 81 | 0.372 | 0.0006 |
| 200 | 102 | 0.374 | 0.0001 |

The effect peaks at 50 km, is non-significant at 10 km, and stabilizes in the 100–200 km range. The non-significance at 10 km argues against a direct line-of-sight coastal observation mechanism. The stability at broader bandwidths suggests the association operates at a coastal-zone scale.

**[Figure 4 about here]**

### 4.5 Confound Tests

Six potential confounding variables were tested using nested F-tests:

| Confound | n | S adds to confound (F, p) | Confound adds to S (F, p) | Verdict |
|---|---|---|---|---|
| Ocean depth | 102 | F = 10.3, p = 0.002 | F = 2.6, p = 0.113 | S dominant |
| Magnetic anomaly (EMAG2) | 94 | F = 12.1, p = 0.001 | F = 2.3, p = 0.129 | S dominant |
| Chlorophyll-a (MODIS) | 99 | F = 18.5, p < 0.0001 | F = 1.5, p = 0.219 | S dominant |
| Puget Sound (2×2 interaction) | 22 | S=0 Puget rate = 0.53, not elevated | — | No confound |
| Shore type proxy (ETOPO cliff) | 102 | F = 8.6, p = 0.004 | F = 5.7, p = 0.019 | Both contribute (R² = 0.21) |
| ESI shore classification | 18 | All F-tests non-sig (p > 0.39) | — | Inconclusive (underpowered) |

Among the 94 cells with valid EMAG2v3 coverage, canyon cells exhibit lower mean magnetic anomaly values (47 nT) than flat-shelf cells (91 nT; Mann-Whitney p = 0.0002), inconsistent with a magnetic-attraction explanation. Chlorophyll-a concentration is orthogonal to S (ρ = −0.02), confirming that the canyon signal is not a proxy for productive upwelling zones.

The shore type proxy test is the most informative confound result. Steep coastal cliffs (measured via ETOPO land-side gradient) independently predict UAP density (p = 0.019), but S retains independent explanatory power after controlling for cliff steepness (p = 0.004). The combined model explains 21% of variance in log(O/E). This suggests that while dramatic coastal topography contributes, submarine canyon steepness carries additional information beyond what surface terrain alone predicts. However, the two variables are moderately correlated (Spearman ρ = 0.61), limiting the precision of their separation.

The ESI shore classification test — a direct measure of coastal geomorphology using NOAA shoreline data — was limited to 18 cells in California and lacks coverage of Puget Sound. Its non-significant results (p > 0.39) likely reflect low statistical power rather than absence of effect.

### 4.6 Navy OPAREA Confound

Military proximity was tested regionally because of the geometric confound described in Section 3.8.

In Puget Sound (n = 22), where OPAREA boundaries are geographically separable from canyon features, S retains significance after OPAREA control (F = 6.8, p = 0.018), while OPAREA distance does not reach significance after S control (F = 2.9, p = 0.10). In Central California (n = 43), S is marginal after OPAREA control (F = 3.9, p = 0.056), and OPAREA distance is not significant after S control (p = 0.14). At Monterey Bay, two canyon cells 127–141 km from any OPAREA show mean log(O/E) = 1.29, while adjacent flat-shelf cells report near baseline — consistent with the canyon interpretation, though not a formal statistical test. Southern California results are uninformative due to the coastline-boundary collinearity (Section 3.8).

A definitive resolution of the military confound would require classified operational tempo data. The publicly available polygon analysis narrows the confound in Puget Sound and is consistent with the canyon interpretation at Monterey, but cannot resolve it in Southern California.

**Note on Puget Sound definitions.** The regional decomposition (Section 4.3) defines Puget Sound as lat ≥ 46°N (n = 22 cells, including 11 S = 0 cells), yielding an S = 0 aggregate rate of 0.74 and canyon uplift of 6.8×. The confound test uses a stricter coastal definition (lat ≥ 46°N and lon ≤ −121°, n = 18, including 7 S = 0 cells), which excludes four inland cells, yielding an S = 0 aggregate rate of 0.53 and uplift of 9.5×. Both values are correct within their respective definitions. Section 4.6 uses the 22-cell definition.

### 4.7 Temporal Replication

The association was tested across multiple time windows. The replication suite recomputes population expectations (E_i) within each time window independently, producing slightly different baseline statistics (ρ = 0.338, n = 103) from the primary evaluation (ρ = 0.374, n = 102) due to cell-count edge effects at the minimum-reports threshold and independent E_i normalization:

| Test | ρ | p | n cells |
|---|---|---|---|
| Replication baseline (1990–2014) | 0.338 | 0.0005 | 103 |
| Early split (1990–2002) | 0.337 | 0.003 | 75 |
| Late split (2003–2014) | 0.361 | 0.00004 | 122 |
| Mid split (1990–2006 → 2007–2014) | 0.334 | 0.0007 | 100 |
| Even → Odd years | 0.319 | 0.0009 | 106 |
| Post-2014 holdout (2015–2023) | 0.350 | 0.0001 | 119 |
| 5-year rolling windows (21 total) | 21/21 positive | 18/21 sig | 14–116 |

The association is present in both temporal halves. Of 21 rolling five-year windows spanning 1990–2014, all produce positive ρ values and 18 reach significance at α = 0.05. The temporal stability argues against an event-driven or media-cycle explanation.

**Geocoding caveat.** The post-2014 holdout (ρ = 0.350, p = 0.0001, n = 119) uses the same city-centroid geocoding as the primary dataset. Reports from "Bellevue, WA" in 2020 receive the same coordinates as in 2005. This means the spatial structure of the holdout is not independent of the training data. The temporal holdout demonstrates that the association persists in newer reports, but it does not constitute a spatially independent replication. The minimum-reports threshold for the post-2014 analysis (n = 5 per cell) is lower than the primary analysis (n = 20), which may affect the comparison.

### 4.8 Spatial Replication

Leave-one-region-out cross-validation (5 geographic folds) yielded a mean ρ of 0.315, with 4 of 5 folds positive and 2 of 5 reaching significance:

| Fold held out | n (held out) | ρ | p |
|---|---|---|---|
| WA North (Puget) | 13 | 0.630 | 0.021 |
| WA South / OR North | 20 | −0.126 | 0.597 |
| OR South / NorCal | 8 | 0.514 | 0.192 |
| Central California | 41 | 0.068 | 0.673 |
| Southern California | 21 | 0.490 | 0.024 |

The two significant folds (WA North, SoCal) correspond to the regions with the strongest canyon signals. Central California shows near-zero correlation when held out. The negative fold (WA South/OR North) contains the Oregon S = 0 anomaly discussed in Section 4.3.

Spatial forward prediction — training on all cells outside a region and predicting within — yielded significant results for Puget (ρ = 0.604, p = 0.017, n = 15) and SoCal (ρ = 0.470, p = 0.049, n = 18). Population-adjusted expected counts were recomputed within each fold to prevent information leakage, following a correction during the statistical audit (Appendix A). Pre-correction LOO mean ρ was 0.38; post-correction it dropped to 0.315.

### 4.9 Threshold Sensitivity

The 60 m/km canyon gradient threshold was selected during exploratory analysis (Sprint 3 dose-response binning across four gradient bins), raising a double-dipping concern. A post-hoc sensitivity sweep across eight thresholds (20, 30, 40, 50, 60, 70, 80, 100 m/km) showed that all thresholds produced significant associations (ρ range: 0.374–0.416, all p < 0.0002). This mitigates the concern that the result depends on a particular threshold choice, but the sweep itself is post-hoc and does not fully resolve the selection issue. A pre-registered threshold in an independent dataset would provide stronger evidence.

### 4.10 East Coast and Gulf Coast

The same analysis applied to the East and Gulf Coast yields ρ = 0.055 (p = 0.459), with only 2 of 185 testable cells classified as canyon-proximate. The wider continental shelf on the East Coast places submarine canyons far from shore (typically >100 km), leaving insufficient near-shore bathymetric contrast to test the canyon hypothesis using land-based observer data. This is a testability limitation, not a falsification: the analysis cannot distinguish "no effect" from "no measurable contrast" with the available data.

**[Figure 4 also shows East Coast null across all bandwidths]**

### 4.11 Norway Replication

An out-of-country replication was attempted using Norwegian fjords (SRTM30 bathymetry, WorldPop population, 40 NUFORC reports across 463 coastal cells). Of the 17 cells with sufficient reports, an initial Spearman test yielded ρ = 0.49 (p = 0.047). However, population-controlled logistic regression reduced the canyon score to non-significance (S coefficient = 0.03, OR = 1.03, p = 0.76). **Provenance note:** these logistic regression values are reported from the repository summary and README; no final result JSON for the logistic retest is shipped in `results/phase_ev2/`. The initial Spearman result is confirmed in the phase summary, but the population-controlled retest should be verified against the analysis scripts before citation. Norway's coastline is almost entirely fjords, leaving only 7 cells with S = 0 — insufficient to construct a meaningful canyon-versus-flat contrast.

The Norwegian test is inconclusive. The simplified scoring formula used for Norway (lacking the shore proximity and coastal complexity components of the US scoring function) and the extremely small sample (40 reports, 17 testable cells) provide insufficient power to detect or rule out an effect. A fully powered international replication would require a site with adequate flat-shelf control cells, sufficient UAP reporting data, and the complete scoring methodology.


## 5. Discussion

### 5.1 Summary of Findings

Submarine canyon steepness predicts UAP report density along the US West Coast (ρ = 0.374, p = 0.0001, n = 102). The association survived nested confound tests (ocean depth, magnetic anomalies, chlorophyll-a, military proximity), persists across temporal splits including a post-2014 holdout, and remains positive in 4 of 5 spatial cross-validation folds. It concentrates in Puget Sound, La Jolla/San Diego, and Monterey Bay; excluding Puget Sound weakens but does not eliminate it (ρ = 0.243, p = 0.021). The association is absent on the East Coast and inconclusive in Norway.

### 5.2 What This Study Does Not Show

The correlation does not identify a mechanism. Three explanations remain viable.

Canyon-proximate coastlines feature dramatic topography — cliffs, deep bays, headlands — that may drive sky-watching or maritime activity beyond what population density captures. The ESI shore type test shows S retains predictive power after controlling for cliff steepness (p = 0.004), but is underpowered (n = 18). Unmeasured behavioral correlates of dramatic coastline remain plausible.

The West Coast's narrow continental shelf places submarine features closer to population centers. The canyon signal may be inseparable from a narrow-shelf signal using land-based observer data.

A genuine environmental phenomenon — atmospheric, electromagnetic, or acoustic — linked to canyon dynamics is also consistent with the data. Chlorophyll-a orthogonality argues against simple upwelling, but finer-grained oceanographic measurements were unavailable.

The mechanism is unresolved. Near-shore bathymetry predicts an observational anomaly; whether it reflects environment, behavior, or an unmeasured confound remains open.

### 5.3 The Military Question

Military proximity is the most consequential confound. The analysis yields a regionally differentiated answer. In Puget Sound, the canyon signal retains significance after OPAREA adjustment (p = 0.018). At Monterey Bay, two canyon cells lie 127–141 km from any OPAREA yet show elevated reporting, while adjacent flat-shelf cells report near baseline. In Southern California, the SOCAL Range Complex tracks the coastline so closely that the two variables cannot be separated with public data.

The military confound is narrowed but not resolved. A definitive test would require classified operational tempo data. The MarineCadastre polygons may also miss facilities — Dabob Bay, a Navy underwater testing site in Puget Sound, is absent from the data, making the Puget result conservative with respect to military proximity.

### 5.4 Monterey Bay

Monterey Bay is one of the clearest local examples of the canyon–density association. Both canyon cells show elevated reporting (R = 2.75, 4.80) and lie 127–141 km from the nearest OPAREA, while adjacent flat-shelf cells report near baseline. The contrast favors a bathymetric rather than military explanation, though two cells are too few for independent statistical conclusions.

### 5.5 Relationship to Prior Work

Medina et al. (2023) identified the West Coast excess and partially attributed it to light pollution, cloud cover, and military proximity. Canyon steepness predicts additional variance (R² = 0.16) beyond those factors. Population, military, observational, and bathymetric variables likely all contribute; the contribution here is evidence that a specific submarine terrain feature carries independent predictive power.

The cryptoterrestrial hypothesis (Lomas, Case & Masters, 2024) predicts ocean-floor associations, but the observed correlation is equally consistent with atmospheric, electromagnetic, behavioral, or other explanations tied to submarine topography. The data do not distinguish among these frameworks.

### 5.6 Effect Size in Context

Methodologically, the present design is best understood as an ecological spatial study (Lawson, 2006; Elliott & Wartenberg, 2004): area-level exposure (canyon steepness) is correlated with area-level outcome (report density), with no individual-level data linking specific observers to specific terrain features. In that tradition, moderate area-level associations are best treated as hypothesis-generating signals that warrant triangulation across alternative data sources, exposure definitions, and negative controls, rather than as stand-alone evidence of causation (Lawlor, Tilling & Davey Smith, 2016).

The Spearman ρ of 0.374 is moderate (Cohen, 1988), explaining approximately 16% of variance in log(O/E). Measurement noise in self-reported data attenuates correlations; residual confounding can inflate them. The direction of net bias is unknown.

An effect of this magnitude is hypothesis-generating: sufficient to justify targeted replication, insufficient on its own for causal inference.

### 5.7 Implications for Future Research

**Independent replication.** The most informative next step is replication with non-NUFORC data — military sensor records, AARO case files, or FAA pilot reports — using the same frozen canyon score, grid resolution, and primary test statistic but with independent geolocation and standardized reporting criteria. International sites with steep near-shore canyons and adequate flat-shelf controls (e.g., Nazaré, Kaikoura, Tokyo Canyon) would test geographic generalizability.

**Instrumented canyon sites.** Hydrophones, infrasound detectors, and electromagnetic sensors at canyon locations could test specific physical mechanisms. MBARI's Monterey Canyon sensor networks and archival hydroacoustic data from NOAA PMEL arrays and declassified SOSUS segments offer existing infrastructure for retrospective temporal co-occurrence analysis.

**Atmospheric and oceanographic follow-up.** Canyon upwelling produces localized meteorological effects — SST anomalies, coastal fog, atmospheric refraction, temperature inversions — that could contribute to the observed association. Finer-grained oceanographic and atmospheric data would help distinguish a genuine environmental signal from a reporting artifact tied to dramatic coastline.


## 6. Limitations

**Data quality.** NUFORC is self-reported, unverified, and subject to population, cultural, media, and temporal biases. Geocoding to city centroids introduces spatial imprecision varying by county size. Both random noise and systematic bias are present.

**Non-causal interpretation.** The study establishes a spatial correlation, not a mechanism. Unmeasured confounds correlated with canyon-proximate coastlines — behavioral patterns, maritime activity, coastal recreation, cultural factors — remain viable alternative explanations.

**Regional concentration.** The signal is driven primarily by Puget Sound and to a lesser extent by La Jolla/San Diego. Excluding Puget Sound weakens the association substantially (ρ = 0.374 → 0.243). The finding may reflect region-specific rather than general bathymetric effects.

**Temporal holdout is not spatially independent.** The post-2014 holdout shares city-centroid geocoding with the primary dataset. It demonstrates temporal stability but does not constitute an independent spatial replication. The minimum-reports threshold also differs between the primary analysis (n ≥ 20) and the holdout (n ≥ 5), which may affect comparability.

**Threshold selection.** The 60 m/km threshold was identified during exploratory analysis. A post-hoc sweep (20–100 m/km) confirms robustness but does not fully resolve the data-dipping concern.

**Confound resolution is incomplete.** The ESI shore type confound test is underpowered (n = 18, California only, no Puget Sound coverage). The military OPAREA test is uninformative in Southern California. Unmeasured behavioral correlates of dramatic coastline (recreation, maritime activity, sky-watching culture) have not been tested. Proximity to major coastal airports was not tested as a confound because at 0.5° grid resolution (~55 × 40 km cells), airports and canyon features co-locate in the same cells (e.g., SAN/La Jolla, SeaTac/Puget Sound), precluding meaningful separation.

**Possible residual confounding.** The county-centroid population proxy is coarse. Canyon-proximate areas may have systematically different population distributions within counties (e.g., coastal concentration) that the model does not capture. The land/ocean IDW weighting ratio (3.0:0.05) was not subjected to sensitivity analysis.

**Geographic scope.** The association is detected only on the US West Coast. Without replication in an independent geography with near-shore canyons and adequate flat-shelf controls, the finding remains regional. The East Coast null result may reflect absence of effect or absence of testable contrast.

**Grid resolution and co-location.** Because the canyon score is computed at 0.5° grid resolution and from bathymetry within a 50 km radius, the analysis cannot cleanly separate sub-cell co-located features such as airports, ports, cliffs, and canyon heads. Any feature systematically proximate to near-shore canyons at this spatial scale could confound the result.

**Statistical approximations.** The cKDTree used for spatial distance operates in degree-space, introducing up to ~40% distance error at 48°N. The OLS regression used as a secondary metric assumes log-normal residuals without formal diagnostic testing.

**Oregon anomaly.** Oregon flat-shelf cells report at 2.09× expected rate, indicating the population model is incomplete for the Pacific Northwest.

**No independent replication.** No non-NUFORC dataset has tested this association. Military sensor data, AARO case files, or FAA reports would substantially strengthen or weaken the result.

**Author credentials.** The author has no institutional affiliation in geoscience, oceanography, or spatial statistics. AI assistance (Claude, Anthropic) was used for code generation and statistical design. All code and audit trail are publicly available.


## 7. Data and Code Availability

The project repository is available at:
https://github.com/antoniwedzikowski-rgb/uap-canyon-analysis

The repository contains:
- Analysis scripts (Python, in `notebooks/`)
- Results JSON files (in `results/phase_ev2/`)
- Publication figures (in `figures/`)
- A 47-finding statistical audit report (`STATISTICAL_AUDIT_REPORT.md`)
- Phase E evaluation summary (`results/PHASE_E_SUMMARY.md`)

The primary data sources — NUFORC reports, ETOPO bathymetry, US Census population, EMAG2v3 magnetic anomaly, and NOAA MarineCadastre OPAREA polygons — are publicly available from their respective agencies. The NUFORC post-2014 data was obtained from the HuggingFace mirror (kcimc/NUFORC); its continued availability cannot be guaranteed. ESI shoreline data is available from NOAA for individual US states.

The analysis was developed iteratively across seven phases. Some intermediate scripts use different parameters or contain known issues documented in the audit report. The `STATISTICAL_AUDIT_REPORT.md` catalogs 47 findings (5 critical, 14 high, 28 medium/minor) and their resolution status. Four of the five critical findings were corrected (the fifth concerns a null result not cited in this paper); the main E-RED v2 result (ρ = 0.374, p = 0.0001) was unaffected by any bug.


## 8. Conclusion

Submarine canyon steepness predicts UAP report density along the US West Coast after population control (ρ = 0.374, ~16% of variance). The association is temporally stable, partially replicates in spatial cross-validation, and was not eliminated by the confounds tested here. It is regional, concentrated in a few locations with extreme near-shore topography, and absent on the East Coast.

Unresolved confounds remain: unmeasured behavioral correlates of dramatic coastline, military activity in Southern California, and the shared geocoding structure of the temporal holdout. No independent dataset has tested the association. The mechanism — environmental, behavioral, or artifactual — is unknown.

Resolution requires replication with independently geolocated, systematically collected data and finer spatial controls than NUFORC provides.


## Appendix A: Statistical Audit Summary

A formal statistical audit was conducted after the primary analysis was complete, identifying 47 findings across the codebase (5 critical, 14 high/major, 17 medium, 7 low, 4 informational).

**Critical findings (the four affecting the headline result were corrected; the fifth — an unimplemented seismicity permutation test — concerns a null result not cited in this paper):**

1. **Uniform E_i in replication suite (CRIT-1).** The replication suite used E_i = 1 for all cells instead of the IDW population model. Fixed: full population model imported. Post-fix: post-2014 ρ improved from 0.28 to 0.35, indicating the population model was suppressing noise rather than inflating signal.

2. **LOO CV normalization leakage (CRIT-2).** Leave-one-region-out cross-validation computed log(O/E) using full-dataset normalization, leaking information from held-out cells. Fixed: per-fold E_i normalization. LOO mean ρ dropped from 0.38 to 0.315.

3. **Spatial forward prediction leakage (CRIT-3).** Same normalization issue as CRIT-2 applied to spatial forward predictions. Fixed: per-fold recomputation. Post-fix: Puget ρ = 0.604, SoCal ρ = 0.470 (both significant).

4. **Threshold double-dipping (CRIT-4).** The 60 m/km threshold was selected from the same data as tested. Resolved: threshold sweep (20–100 m/km, 8 thresholds) shows all thresholds significant (ρ range 0.374–0.416). The effect is not an artifact of threshold choice, though the sweep is post-hoc.

5. **C3 permutation test abandoned (CRIT-5).** Seismicity permutation test was not properly implemented. Resolution: not needed — seismicity showed no correlation with UAP reports (r = 0.019, p = 0.74). This is a null result not cited in the paper.

**Medium-severity findings (documented as limitations):**

The cKDTree degree-space distance approximation (up to 40% error at 48°N), arbitrary land/ocean weighting ratio (3.0:0.05), Phase D degree×111 km approximation, post-2014 shared geocoding, and varying minimum-report thresholds across tests are documented in Section 6.

The main E-RED v2 result (ρ = 0.374, p = 0.0001) was unaffected by any bug — it uses its own population model and haversine distances, independent of the replication suite code.

The full audit report is available at:
https://github.com/antoniwedzikowski-rgb/uap-canyon-analysis/blob/main/STATISTICAL_AUDIT_REPORT.md


## Figure Captions

**Figure 1.** Study area: CONUS coastline and submarine canyon features. Red points indicate canyon cells (gradient > 60 m/km, continental shelf 0 to −500 m depth). Labeled hotspots: Puget Sound, Monterey, Mugu, La Jolla. Subtitle indicates 42,008 coastal UAP reports and 19,977 population-matched controls from NOAA ETOPO 2022. Source: `figures/figure1_study_area.png`.

**Figure 2.** Primary result: quintile decomposition of canyon score versus mean population-adjusted report rate. Bar chart showing mean log(R) = log(O_i / E_i) by S quintile for 102 testable West Coast cells (200 km band). Q1–Q3 (S = 0.00) and Q4 (median S = 0.19) show moderate mean log(R) of 0.20–0.46 with overlapping confidence intervals. Q5 (median S = 1.72) shows a pronounced jump to mean log(R) = 1.40. Error bars: 95% bootstrap CI. Haversine-corrected E_i. Spearman ρ = 0.374 (p = 0.0001, n = 102). Dashed red line: log(R) = 0 (expected rate). Source: `figures/e_red_v2_primary_200km.png`.

**Figure 3.** West Coast bathymetric context and UAP report distribution. ETOPO 2022 ocean depth (blue color scale, meters) with canyon cells (orange dots, gradient > 60 m/km) and UAP report locations (small red dots, n = 15,278 West Coast). Labeled canyon systems: Puget Sound & Juan de Fuca, Astoria Canyon, Monterey Canyon, Mugu Canyon, La Jolla / Scripps Canyon. Source: `figures/figure_hero_bathymetry.png`.

**Figure 4.** Coastal band sensitivity: West Coast versus East Coast. Two-panel bar chart of Spearman ρ(S, log R) at multiple coastal buffer widths. Left panel (West Coast, 10–200 km): green bars indicate p < 0.05; gray bar (10 km, n = 39) is non-significant. Peak at 50 km (ρ = 0.43, p = 0.0005, n = 62). Right panel (East Coast, 25–200 km; 10 km omitted due to only 1 canyon cell): all bars gray (non-significant), n = 87–185, all ρ < 0.06. Error bars: bootstrap 95% CI. Dashed red line: ρ = 0. Source: `figures/e_red_band_sweep.png`.

**Supplementary Figure S1.** Robustness forest plot. Point estimates and 95% confidence intervals for S–logR association across confound tests, temporal splits, spatial cross-validation folds, and sensitivity analyses. Source: `figures/figure_forest_robustness.png`.

**Supplementary Figure S2.** Canyon head proximity to coastline. Three-panel figure. Left: histogram of canyon-head-to-shore distance for West Coast (n = 191, red) and East/Gulf Coast (n = 580, blue), showing West Coast canyon heads concentrated within 50 km of shore. Center: head depth versus shore distance, showing West Coast canyons originate in shallower water closer to shore. Right: canyon component size versus shore distance. Source: `figures/d7a_canyon_head_distances.png`.

**Supplementary Figure S3.** Blind prediction versus reality. Side-by-side spatial maps of the US West Coast. Left panel: canyon score S (from ocean-floor geometry alone, yellow-to-red color scale). Right panel: population-adjusted UAP report excess log(R) (blue-to-red color scale, 1990–2014). Subtitle: "The model was built from seafloor geometry without seeing any UAP data. ρ = 0.37, p = 0.0001." Source: `figures/figure_prediction_vs_reality.png`.


## References

Allen, S.E. & Durrieu de Madron, X. (2009). A review of the role of submarine canyons in deep-ocean exchange with the shelf. *Ocean Science*, 5, 607–620. doi:10.5194/os-5-607-2009

Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Hillsdale, NJ: Lawrence Erlbaum Associates.

Elliott, P. & Wartenberg, D. (2004). Spatial epidemiology: current approaches and future challenges. *Environmental Health Perspectives*, 112(9), 998–1006. doi:10.1289/ehp.6735

Lawlor, D.A., Tilling, K. & Davey Smith, G. (2016). Triangulation in aetiological epidemiology. *International Journal of Epidemiology*, 45(6), 1866–1886. doi:10.1093/ije/dyw314

Lawson, A.B. (2006). *Statistical Methods in Spatial Epidemiology* (2nd ed.). Chichester: John Wiley & Sons.

Gallaudet, T. (2024). Beneath the Surface: We May Learn More about UAP by Looking in the Ocean. *Sol Foundation White Paper*, Vol. 1, No. 1. https://thesolfoundation.org/publication/beneath-the-surface/

Lomas, T., Case, B. & Masters, M.P. (2024). The cryptoterrestrial hypothesis: A case for scientific openness to a concealed earthly explanation for Unidentified Anomalous Phenomena. *Philosophy and Cosmology*, 33, 67–122.

Medina, R.M., Brewer, S.C. & Kirkpatrick, S.M. (2023). An environmental analysis of public UAP sightings and sky view potential. *Scientific Reports*, 13, 22213. doi:10.1038/s41598-023-49527-x

