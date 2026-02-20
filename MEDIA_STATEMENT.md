# Media Statement — UAP-Canyon Spatial Association Analysis

**Author:** Antoni Wędzikowski — independent researcher, lawyer and legaltech founder based in Warsaw, Poland
**Contact:** [LinkedIn](https://www.linkedin.com/in/antekwedzikowski/) · via GitHub Issues or through referring publication
**Repository:** https://github.com/antoniwedzikowski-rgb/uap-canyon-analysis

---

## What is this?

A statistical analysis testing whether UAP (Unidentified Anomalous Phenomena) reports from the NUFORC database cluster near submarine canyon features along the US coastline. The full code, data, and methodology are open-source under MIT license.

## Key finding

UAP reports within 200 km of the US West Coast show a statistically significant spatial association with steep submarine canyon features — Spearman ρ = 0.374 (p = 0.0001, n = 102 grid cells, 200 km coastal band) after controlling for population density, military installations, port infrastructure, and proximity to 35 Navy offshore operating area polygons (NOAA MarineCadastre). The association replicated in independent post-2014 data not used to develop the model (ρ = 0.35, p = 0.0001), is stable across temporal splits (1990-2002 and 2003-2014 both significant), survives all tested gradient thresholds (20-100 m/km), and correctly predicted 4 out of 5 top hotspot cells in held-out spatial validation.

The effect concentrates in regions with extreme near-shore topography: Puget Sound (6.8× uplift, ρ = 0.77, p = 0.005, n = 11 cells), San Diego (9.8× uplift), and Monterey Bay (2.75–4.80× uplift). Coastal upwelling (satellite chlorophyll-a) was orthogonal to canyon steepness and did not explain the association. Monterey Bay canyon cells — located 127–192 km from the nearest Navy operating area — still show elevated report rates (logR = 0.75 vs −0.02 for non-canyon cells), ruling out military proximity as an explanation at that distance.

### Key figures

**Figure 1 — Study area: US coastline with submarine canyon features (>60 m/km gradient)**
![Study Area](figures/figure1_study_area.png)
*Red dots mark steep submarine canyon cells detected from NOAA ETOPO1 bathymetry. Named canyons labeled. Note the concentration of near-shore canyons on the West Coast versus far-offshore canyons on the East Coast.*

**Figure 2 — Headline result: UAP report excess by canyon score quintile**
![Quintile Result](figures/e_red_v2_primary_200km.png)
*102 West Coast grid cells split into quintiles by canyon proximity score (S). Q1–Q4 (no/weak canyon) show modest excess. Q5 (strongest canyon signal) jumps to ~25× expected rate. Population-adjusted, haversine-corrected.*

**Figure 3 — West Coast zoom: bathymetry, canyon cells, and UAP report density**
![Hero Bathymetry](figures/figure_hero_bathymetry.png)
*Color bathymetry from NOAA ETOPO1. Orange/red dots = shelf cells with gradient >60 m/km (submarine canyon features). Faint red scatter = individual UAP reports (NUFORC, 1990–2014). Five major canyon systems labeled. Note how canyon cells and UAP report clusters overlap in Puget Sound, Monterey Bay, and the San Diego corridor.*

**Figure 4 — West Coast vs East Coast: the geographic asymmetry**
![Band Sweep](figures/e_red_band_sweep.png)
*Green = statistically significant (p < 0.05). The canyon–UAP association is robust across all coastal band widths on the West Coast but absent on the East Coast, where canyons sit far from shore.*

**Figure 5 — Why the asymmetry exists: canyon proximity to shore**
![Canyon Distances](figures/d7a_canyon_head_distances.png)
*West Coast canyons (blue) cluster within 50 km of shore. East/Gulf Coast canyons (red) are 100–400 km offshore — beyond the reach of land-based observer data.*

### Strength of evidence

**Regional breakdown** — the effect is not uniform across the West Coast:

| Region | Reports per expected (no canyon) | Reports per expected (canyon cells) | Uplift | Grid cells |
|--------|--------------------------------|-------------------------------------|--------|------------|
| Puget Sound (46–50°N) | 0.74× | 5.04× | 6.8× | 22 |
| San Diego (32–33.5°N) | 0.60× | 5.85× | 9.8× | 5 |
| Rest of West Coast | 1.08× | 1.53× | 1.4× | 75 |

Non-canyon cells in Puget Sound and San Diego report *below* baseline — the uplift is canyon-specific, not regional.

**Dose-response** — Monterey Canyon, one of the world's largest submarine canyons, shows intermediate canyon scores (S = 1.3–1.6) and intermediate uplift (2.75–4.80×). The effect scales with canyon proximity to shore, not canyon depth alone.

**Non-linearity** — Quintile analysis shows Q1–Q4 (no/weak canyon) have similar report excess (logR 0.20–0.46, overlapping confidence intervals). Q5 (strongest canyon signal) jumps to logR = 1.40 (~25× expected). The effect concentrates in the extreme tail.

**Temporal stability** — The association holds in both halves of the dataset (1990–2002: ρ = 0.32, p = 0.002; 2003–2014: ρ = 0.36, p = 0.0003) and in 21 out of 21 rolling 5-year windows (18 of 21 statistically significant). It is not driven by a single event or time period.

**Coastal band dependence** — The effect peaks at 50 km from shore (ρ = 0.43), is non-significant at 10 km (ρ = 0.15), and stabilizes at 100–200 km (ρ = 0.37). This argues against a direct line-of-sight observation mechanism and suggests a broader coastal-zone phenomenon.

**Bootstrap confidence interval** — 2,000-resample cluster bootstrap yields 95% CI: [0.190, 0.531]. The lower bound is far from zero.

**Magnetic anomaly** — Canyon score S dominates magnetic anomaly in nested F-tests. Canyon cells have *lower* magnetic anomalies than non-canyon cells — the opposite of what a magnetic-attraction hypothesis would predict.

### Anticipated objections

**"It's just population — more people live near canyons."**
Report rates are population-adjusted using inverse distance weighting from 3,108 US county centroids. The key test: in Puget Sound, cells *without* canyon features report at 0.74× the expected rate (below baseline), while canyon cells report at 5.04× — a 6.8× differential within the same metropolitan region. If population were driving the effect, non-canyon Puget cells would also be elevated. They are not.

**"It's military activity — Navy operates near those canyons."**
We controlled for proximity to 171 DoD installations and 35 Navy offshore operating area (OPAREA) polygons from NOAA MarineCadastre. Canyon score (S) remains the dominant predictor in Puget Sound (p = 0.018 vs OPAREA p = 0.10). Monterey Bay canyon cells sit 127–192 km from the nearest OPAREA boundary and still show elevated report rates (logR = 0.75 vs −0.02 for non-canyon cells).

**"It's coastal geography — you're just measuring coastline complexity."**
We tested a shore-type proxy (ETOPO coastal cliff metric) as an alternative predictor. Canyon score S survives cliff control (p = 0.004). Both contribute independently (combined R² = 0.21), but canyon steepness is not reducible to coastline shape.

**"It's fishing ports or marine traffic."**
Port infrastructure (7,747 ports/marinas from OpenStreetMap) is included as a covariate in the model. The canyon association survives this control.

**"It's ocean currents or upwelling attracting marine activity."**
We tested coastal upwelling directly using satellite chlorophyll-a data (NASA MODIS Aqua, 2003–2020 climatology). Upwelling (chl-a) is uncorrelated with canyon steepness (rho = −0.02) and canyon score S remains the dominant predictor when chl-a is added to the model (F = 18.5, p < 0.0001). The two variables are orthogonal — upwelling does not explain the canyon association.

**"You cherry-picked the threshold."**
The 60 m/km gradient threshold was selected during Phase C/D, but a sensitivity sweep across all thresholds from 20 to 100 m/km shows the association is significant at every threshold tested (rho 0.37–0.42, all p < 0.0002).

**"How do we know the model isn't overfit?"**
The scoring function was frozen before evaluation — it uses only bathymetric geometry (gradient, proximity, coastal complexity) and touches no UAP data. In held-out spatial validation (leave-one-region-out cross-validation), it correctly predicted 4 out of 5 top hotspot cells. It also replicated in fully independent post-2014 data (rho = 0.35, p = 0.0001) that was not used at any stage of model development.

## What the data does not show

The effect is not detected on the East and Gulf Coasts (ρ = 0.055, p = 0.459). The wider continental shelf there places canyons far from shore, leaving insufficient contrast to test the hypothesis with land-based observer data. This is a testability limitation, not a falsification. The effect concentrates in specific West Coast regions — primarily Puget Sound, Monterey Bay, and the San Diego canyon system — and does not generalise across the full US coastline. An out-of-country replication on Norwegian fjords was attempted but is inconclusive after population control — Norway's coastline is entirely fjords, leaving no flat-shelf contrast group.

## Methodology

The analysis uses 24 Python scripts and 14 public datasets including NUFORC sighting reports (80,332 reports, 1990-2014; primary analysis restricted to West Coast 200 km coastal band), NOAA ETOPO1 bathymetry, US Census population data, military base locations, Navy operating area boundaries, and port infrastructure. Canyon features are detected via gradient magnitude on the continental shelf (threshold 60 m/km). Report rates are population-adjusted using inverse distance weighting from county-level census data.

## Next steps

The priority is to prepare a preprint for arXiv with full methodology available for independent scrutiny. The goal is peer review, independent replication with non-US data, and engagement from researchers working on spatial analysis or the systematic study of anomalous aerial phenomena.

Beyond that, I would welcome engagement from researchers who work on spatial analysis, geophysical data, or the systematic study of anomalous aerial phenomena. Projects like VASCO, led by Beatriz Villarroel, or the Galileo Project at Harvard have established frameworks for rigorous work in this space, and their perspective on the methodology and the geographic asymmetry would be invaluable.

## Honest uncertainties

- The effect is regional (West Coast only) and strongest in three canyon systems — Puget Sound (6.8×), San Diego (9.8×), and Monterey Bay (2.75–4.80×). The remaining West Coast cells average 1.4× uplift. All three strong regions have canyons unusually close to shore.
- NUFORC is a self-reported database. We cannot rule out unknown reporting biases correlated with coastal geography.
- This is a single-author analysis that has not been independently replicated or published in a peer-reviewed journal.
- An attempted replication on Norwegian fjords is inconclusive — Norway's coastline is entirely fjords, leaving no flat-shelf contrast group for comparison.
- We show a spatial correlation, not a causal mechanism. We do not claim to know *why* this pattern exists.

## What would strengthen or weaken this finding

- **Independent replication** in countries with near-shore canyons and independent UAP/UFO databases (Japan, Chile, Portugal, Mediterranean) would be the strongest test.
- **Hydrophone or sonar data** from NOAA or navy acoustic monitoring networks near canyon systems could test whether anomalous underwater acoustic activity correlates with surface reports.
- **Improved population models** using nighttime lights (VIIRS) or mobile phone density instead of county-level census would reduce uncertainty in the population adjustment.
- **Temporal correlation** with oceanographic events (internal waves, upwelling episodes) could test physical mechanisms.
- If independent replications consistently fail, or if a confound is identified that explains the Puget/Monterey/San Diego concentration, the finding should be downgraded or retracted.

## For accurate reporting

This is a preliminary, single-author analysis showing a robust regional signal whose mechanism and scope remain genuinely uncertain. Any coverage should note that the effect is not detected on the East Coast and is concentrated in specific West Coast regions. The goal is to contribute to a body of evidence, not to make claims that outrun it.

---

*Analysis conducted with Claude Code (Anthropic) as a research partner. Analytical decisions and domain knowledge are the author's. AI served as a force multiplier for rigorous testing.*

**Audio walkthrough:** [`UAPs_Cluster_Over_Steep_Underwater_Canyons.m4a`](media/UAPs_Cluster_Over_Steep_Underwater_Canyons.m4a) (NotebookLM overview, ~20 min)
