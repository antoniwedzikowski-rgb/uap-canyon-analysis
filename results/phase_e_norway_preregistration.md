# Phase E: Norwegian Fjord Replication — Pre-Registration

**Date**: 2026-02-18
**Written BEFORE seeing any Norwegian UAP data**

## Hypothesis

If the canyon-UAP correlation observed in Puget Sound reflects a genuine
geophysical mechanism (CTH or atmospheric optics from extreme submarine
topography), then Norwegian fjords — which have even more extreme underwater
topography close to shore — should show the same pattern.

## Design

1. **Scoring**: Apply the same frozen canyon scoring function S (60 m/km
   gradient threshold, 25 km radius, same component aggregation) to Norwegian
   coastal bathymetry (GEBCO or EMODnet).

2. **Reports**: Collect Norwegian UAP/UFO reports with coordinates from
   available sources (NUFORC international, FOTOCAT catalogue, scraped
   national databases).

3. **Population model**: Use Norwegian population data (SSB/Kartverket) to
   compute E_i per coastal cell, same rate-ratio approach as US analysis.

4. **Metric**: Spearman(S, log R) across 0.5° coastal cells with N ≥ threshold.

## Pre-Registered Decision Criteria

| Outcome | Criterion | Interpretation |
|---------|-----------|----------------|
| **POSITIVE** | ρ(S, logR) > 0.3 AND p < 0.05 | Fjord replication succeeds |
| **NULL** | ρ < 0.15 OR p > 0.10 | Fjord replication fails |
| **INCONCLUSIVE** | 0.15 ≤ ρ ≤ 0.3 or 0.05 ≤ p ≤ 0.10 | Insufficient to decide |
| **UNDERPOWERED** | n < 15 testable cells | Cannot evaluate |

## Key Confounds Addressed

If positive:
- **Kills US-specific cultural confound** (Navy bases, American reporting culture)
- **Kills demographic confound** (Norway has different population distribution)
- **Supports extreme topography mechanism** (fjords = canyons × 3)

If null:
- **Weakens CTH** — effect doesn't generalize beyond US West Coast
- **Supports US-specific confound** (culture, demographics, reporting bias)
- BUT: low N may prevent conclusive null (Norway has far fewer UFO reports)

## Power Concern

Norwegian UFO databases are small (~4000 total for all of UFO-Norge, vs
64,000 NUFORC in US). Even if the effect exists, we may not have enough
reports per cell for stable R_i estimates. If n < 15 testable cells,
the test is declared UNDERPOWERED regardless of ρ.

## Data Sources (to be used)

- Bathymetry: GEBCO 2023 (15 arc-second global grid)
- UAP reports: best available (NUFORC international + any obtainable Norwegian data)
- Population: Norwegian census or WorldPop gridded data
- Analysis restricted to coastal Norway (58-71°N, 4-31°E)
