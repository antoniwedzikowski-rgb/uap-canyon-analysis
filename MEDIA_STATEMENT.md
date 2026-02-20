# Media Statement — UAP-Canyon Spatial Association Analysis

**Author:** Antoni Wedzikowski — independent researcher, lawyer and legaltech founder based in Warsaw, Poland
**Contact:** [LinkedIn](https://www.linkedin.com/in/antekwedzikowski/) · via GitHub Issues or through referring publication
**Repository:** https://github.com/antoniwedzikowski-rgb/uap-canyon-analysis

---

## What is this?

A statistical analysis testing whether UAP (Unidentified Anomalous Phenomena) reports from the NUFORC database cluster near submarine canyon features along the US coastline. The full code, data, and methodology are open-source under MIT license.

## Key finding

UAP reports within 200 km of the US West Coast show a statistically significant spatial association with steep submarine canyon features — Spearman ρ = 0.374 (p = 0.0001, n = 102 grid cells, 200 km coastal band) after controlling for population density, military installations, port infrastructure, and proximity to 35 Navy offshore operating area polygons (NOAA MarineCadastre). The association replicated in independent post-2014 data not used to develop the model (ρ = 0.35, p = 0.0001), is stable across temporal splits (1990-2002 and 2003-2014 both significant), survives all tested gradient thresholds (20-100 m/km), and correctly predicted 4 out of 5 top hotspot cells in held-out spatial validation.

The effect concentrates in regions with extreme near-shore topography: Puget Sound (6.8× uplift, ρ = 0.77, p = 0.005, n = 11 cells) and San Diego (9.8× uplift). Coastal upwelling (satellite chlorophyll-a) was orthogonal to canyon steepness and did not explain the association. Monterey Bay canyon cells — located 127–192 km from the nearest Navy operating area — still show elevated report rates (logR = 0.75 vs −0.02 for non-canyon cells), ruling out military proximity as an explanation at that distance.

### Key figures

**Figure 1 — Study area: US coastline with submarine canyon features (>60 m/km gradient)**
![Study Area](figures/figure1_study_area.png)
*Red dots mark steep submarine canyon cells detected from NOAA ETOPO1 bathymetry. Named canyons labeled. Note the concentration of near-shore canyons on the West Coast versus far-offshore canyons on the East Coast.*

**Figure 2 — Headline result: UAP report excess by canyon score quintile**
![Quintile Result](figures/e_red_v2_primary_200km.png)
*102 West Coast grid cells split into quintiles by canyon proximity score (S). Q1–Q4 (no/weak canyon) show modest excess. Q5 (strongest canyon signal) jumps to ~25× expected rate. Population-adjusted, haversine-corrected.*

**Figure 3 — West Coast vs East Coast: the geographic asymmetry**
![Band Sweep](figures/e_red_band_sweep.png)
*Green = statistically significant (p < 0.05). The canyon–UAP association is robust across all coastal band widths on the West Coast but absent on the East Coast, where canyons sit far from shore.*

**Figure 4 — Why the asymmetry exists: canyon proximity to shore**
![Canyon Distances](figures/d7a_canyon_head_distances.png)
*West Coast canyons (blue) cluster within 50 km of shore. East/Gulf Coast canyons (red) are 100–400 km offshore — beyond the reach of land-based observer data.*

## What the data does not show

The effect is not detected on the East and Gulf Coasts (ρ = 0.055, p = 0.459). The wider continental shelf there places canyons far from shore, leaving insufficient contrast to test the hypothesis with land-based observer data. This is a testability limitation, not a falsification. The effect concentrates in specific West Coast regions — primarily Puget Sound and the San Diego canyon system — and does not generalise across the full US coastline. An out-of-country replication on Norwegian fjords was attempted but is inconclusive after population control — Norway's coastline is entirely fjords, leaving no flat-shelf contrast group.

## Methodology

The analysis uses 24 Python scripts and 14 public datasets including NUFORC sighting reports (80,332 reports, 1990-2014; primary analysis restricted to West Coast 200 km coastal band), NOAA ETOPO1 bathymetry, US Census population data, military base locations, Navy operating area boundaries, and port infrastructure. Canyon features are detected via gradient magnitude on the continental shelf (threshold 60 m/km). Report rates are population-adjusted using inverse distance weighting from county-level census data.

## Next steps

The priority is to prepare a preprint for arXiv with full methodology available for independent scrutiny. The goal is peer review, independent replication with non-US data, and engagement from researchers working on spatial analysis or the systematic study of anomalous aerial phenomena.

## For accurate reporting

This is a preliminary, single-author analysis showing a robust regional signal whose mechanism and scope remain genuinely uncertain. Any coverage should note that the effect is not detected on the East Coast and is concentrated in specific West Coast regions. The goal is to contribute to a body of evidence, not to make claims that outrun it.

---

*Analysis conducted with Claude Code (Anthropic) as a research partner. Analytical decisions and domain knowledge are the author's. AI served as a force multiplier for rigorous testing.*
