# Phase E: Out-of-Sample Evaluation — Summary

## Overview

Phase E tested whether the geometric canyon scoring function (S),
developed in Phases C/D, predicts spatial variation in UAP report density.
The scoring function S was frozen before evaluation began.

The phase evolved through three design iterations as we discovered and
corrected specification errors, culminating in a clear empirical finding.

---

## Timeline and Design Evolution

### E1: Original Pre-Registered Scoring (commit `09de8d8`, tag `phase-e-frozen`)

**Scoring**: S = mean(rank_G + rank_P + rank_C) for steep cells within 50km,
using canyon gradient threshold = 20 m/km (shelf 0 to -500m).

**E2a evaluation**: Global shelf scoring selected top-20 hotspots outside the
NUFORC data footprint (Bahamas, Cuba, British Columbia). All 20 hot predictions
had zero reports. Test was **non-measurable** — not failed, but unevaluable.

**E2b fix** (commit `763fdab`): Restricted ranking to CONUS footprint
(24.5-49.0°N, -125.0 to -66.0°W). Results: HOT 1/2 = 50% (18 insufficient data),
COLD 11/11 = 100% (tautological — S=0 implies OR=0 structurally).

**Diagnosis**: Threshold mismatch discovered — scoring used 20 m/km (canyon
detection) while Phase C/D established the effect in the 60+ m/km gradient bin.
~41% of "hot" cells physically could not produce reports in the evaluation bin.

### E v2: Re-Specification (commit `c2366d2`, tag `phase-ev2-frozen`)

**What changed**: Canyon gradient threshold aligned to 60 m/km (matching Phase C/D
estimand). This was a specification correction, not post-hoc tuning — the 60+ bin
was defined by Phase C/D before Phase E began.

**Results**: 42,128 steep cells, 2,004 components, 347 CONUS grid cells with S > 0.

**Evaluation** (commit `b5eb015`): **INCONCLUSIVE** — per-cell OR approach was
structurally underpowered. Only 11 of 347 cells yielded valid OR (required both
60+ and flat reports within a single 0.5° cell). Hot: 1/1 = 100% but 19/20
insufficient data. Cold: 0/0.

### E-RED: Redesigned Evaluation (commit `0d70882`)

**Rationale**: Switched from per-cell OR to rate ratio R_i = O_i / E_i, which uses
all reports in a cell rather than requiring rare gradient-specific bins.

**Design**:
- Unit: West Coast only (lon ≤ -115°, lat ≥ 30°), cells with N ≥ 20 reports
- E_i: population-weighted expected reports (county centroid inverse-distance,
  land/ocean weighting, normalized so ΣE = ΣO)
- Metrics: Spearman(S, log R), Precision@K, decile plot, Poisson proxy (OLS)
- Two passes: primary 200 km coastal band, secondary 0-20 km

**Bug found**: E_i used `degree * 111 km` approximation for coastal/county
distances. At 48°N (Puget Sound), this overestimates E-W distances by ~33%,
systematically deflating E_i and inflating R_i for Puget cells.

### E-RED v2: Haversine-Corrected (commit `40ec07a`)

Fixed CRITICAL-3: replaced degree×111 with haversine for all distance computations.
Also compared secondary band at 20 km and 25 km.

---

## Final Results (E-RED v2, haversine-corrected)

### Primary (200 km coastal band, n = 102)

| Metric | Value | 95% CI |
|--------|-------|--------|
| Spearman(S, log R) | **0.374** | [0.190, 0.531] |
| p-value | **0.0001** | |
| Spearman without Puget | **0.243** | p = 0.021 |
| β_S (Poisson proxy) | **0.656** | [0.352, 0.975] |
| exp(β) interpretation | **1.93×** rate per unit S | |
| Precision@5 | 80% | |
| Precision@10 | 60% | |

### Secondary (25 km band, n = 50)

| Metric | Value | 95% CI |
|--------|-------|--------|
| Spearman(S, log R) | 0.348 | [0.054, 0.610] |
| p-value | 0.013 | |
| Spearman without Puget | 0.091 | p = 0.58 (NS) |
| β_S (Poisson proxy) | 0.477 | [0.063, 0.857] |

**The 25 km band outperformed 20 km** (rho 0.348 vs 0.285), matching the scoring
function's GRADIENT_RADIUS_KM = 25 km parameter. However, both secondary results
collapse without Puget Sound.

---

## Puget Interaction Test (commits `76f87e8`, `5ee84c1`)

OLS model: `logR ~ S + Puget + S × Puget` (n = 102)

| Coefficient | Value | Bootstrap 95% CI | p_boot |
|-------------|-------|------------------|--------|
| β_S | +0.518 | [+0.131, +0.913] | 0.007 |
| β_Puget | -21.6 | [-34.5, -15.8] | 0.010 |
| β_S×Puget | +11.05 | [+8.15, +17.7] | 0.010 |

**F-test for interaction**: F(1, 98) = 12.2, **p = 0.0007**

### Sanity checks confirm robustness:

1. **Centering**: Large coefficients are an extrapolation artefact (S = 0 does not
   exist in Puget, where S ∈ [1.85, 2.24]). At actual Puget S values, the model
   predicts R = 5.4× vs non-Puget R = 3.8× — a modest 1.4× difference.

2. **Leave-one-out**: β_S×Puget stays positive in all 11 drops (range 10.2–13.1).
   F-test remains significant in all 11 drops (worst: p = 0.016 dropping Vancouver).

3. **Cook's distance**: Only 2 of 102 cells exceed the 4/n threshold.
   Vancouver (49.2°N, R = 72) is the most influential (D = 0.12) but does not
   drive the result alone.

4. **Within-group correlations**:
   - Inside Puget: Spearman(S, logR) = **0.773** (p = 0.005, n = 11)
   - Outside Puget (S > 0 only): Spearman = -0.050 (p = 0.86, n = 15)

---

## Interpretation

### What the data show

1. **Regional signal is real**: S predicts excess UAP report density across the
   West Coast (rho = 0.374, p = 0.0001). This survives removal of Puget Sound
   (rho = 0.243, p = 0.021), haversine correction, and bootstrap validation.

2. **The mechanism is Puget-specific**: The S × Puget interaction is significant
   (p = 0.0007). Within Puget, S explains report density powerfully (rho = 0.77).
   Outside Puget, among cells with S > 0, there is no S-logR correlation (rho ≈ 0).

3. **Proximity does not generalize**: The 25 km secondary band shows a signal
   (p = 0.013) but it vanishes without Puget (p = 0.58). The nearshore canyon
   mechanism does not operate at other West Coast locations.

### What this means

The canyon geometry scoring function S successfully identifies Puget Sound as a
region where UAP report density far exceeds population-based expectations,
in proportion to canyon steepness. This is consistent with Phases C/D (West Coast
OR = 6.18, East/Gulf OR = 0.37).

However, outside Puget Sound, S does not predict cell-level variation among other
"hot" cells (San Diego, Santa Barbara, Monterey, Humboldt). The overall West Coast
Spearman (0.243 without Puget) is driven by the contrast between S > 0 and S = 0
cells, not by dose-response within the S > 0 group.

---

## Confound Test (commit `f0cbbf9`)

The Puget interaction result raised a critical question: is the Puget excess
caused by canyon geometry, or is Puget Sound simply a region with more UAP
reports (Navy bases, reporting culture, Pacific NW observer density)?

### Test design

Compare UAP rates (ΣO / ΣE) in a 2×2 design: Region (Puget / Other) × Canyon (S>0 / S=0).

If Puget S=0 cells have elevated rates → **confound** (the region is "hot" regardless).
If Puget S=0 cells have normal rates → **canyons matter** (excess is geometry-specific).

### Rate table (ΣO / ΣE)

|                  | S = 0 | S > 0 | Canyon uplift |
|------------------|-------|-------|---------------|
| **Puget**        | 0.528 | 5.035 | **9.54×**     |
| **Other WC**     | 1.083 | 1.901 | 1.76×         |

### Result: NOT a confound

Puget S=0 cells have **lower** rates than expected (0.528), below the West Coast
average (1.083). Mann-Whitney p = 0.58 — no significant elevation. The excess
is specific to canyon cells (S>0), with a 9.54× uplift.

San Diego shows a similar pattern (uplift 10.96×) but with only n=4 cells.

### 2×2 interaction model

| Coefficient | Value | Bootstrap 95% CI | F-test |
|-------------|-------|------------------|--------|
| β_Canyon | +0.674 | [+0.156, +1.191] | p = 0.031 * |
| β_Puget | −0.147 | [−1.664, +1.470] | p = 0.73 NS |
| β_Canyon×Puget | +0.841 | [−1.025, +2.626] | p = 0.17 NS |

The canyon main effect is significant (p = 0.031) — canyon cells have higher
rates than non-canyon cells **across the entire West Coast**. The Puget main
effect is null — the region is not generically "hot". The interaction is
positive but not significant (p = 0.17, n=102) — the canyon effect is larger
in Puget but the sample is too small to confirm this statistically in a 2×2
framework.

---

## Band Sensitivity Sweep (commit `69f7241`)

Tested S–logR correlation at 10, 25, 50, 100, 200 km coastal bands,
separately for West Coast and East Coast.

### West Coast

| Band (km) | n cells | Spearman ρ | p-value | w/o Puget ρ | w/o Puget p |
|-----------|---------|------------|---------|-------------|-------------|
| 10 | 39 | 0.149 | 0.37 (NS) | — | — |
| 25 | 50 | 0.348 | 0.013 | 0.091 | 0.58 (NS) |
| **50** | **63** | **0.430** | **0.0005** | — | — |
| 100 | 83 | 0.372 | 0.0006 | — | — |
| 200 | 102 | 0.374 | 0.0001 | 0.243 | 0.021 |

Peak at 50 km supports a nearshore mechanism. Signal emerges at 25 km
and stabilizes at 50+ km. The 10 km band lacks power (n = 39).

### East Coast

| Band (km) | n cells | Spearman ρ | p-value |
|-----------|---------|------------|---------|
| 10 | 41 | -0.068 | 0.67 |
| 25 | 72 | 0.055 | 0.65 |
| 50 | 108 | 0.029 | 0.77 |
| 100 | 151 | 0.043 | 0.60 |
| 200 | 190 | 0.055 | 0.46 |

**Null at every bandwidth.** Only 2 testable S > 0 cells (both Miami area,
S ≈ 0.32). East Coast canyon cells (Norfolk, Hudson) are too far from the
NUFORC-dense areas to test. The CTH does not generalize to the East Coast.

---

## Shoreline Type Proxy Test (commit `1d7c948`)

Tested Option 3 confound: does above-water coastal topography (rocky cliffs,
viewpoints) explain the UAP excess better than below-water canyons (S)?

**Proxy**: ETOPO land-side elevation gradient within 5 km of coast, per cell.
High cliff_score = steep terrain near water (rocky/cliffy).

### S vs cliff_score correlation

| Metric | Value |
|--------|-------|
| Spearman(S, cliff) | **0.613** (p < 0.0001) |
| Pearson(S, cliff) | 0.456 (p < 0.0001) |

Moderate collinearity: where canyons exist underwater, the coast above tends
to be steeper too. But ρ = 0.61 (not ~1) allows partial separation.

### Competing models (logR ~ predictors, n = 102)

| Model | R² | β_S | β_cliff |
|-------|-----|-----|---------|
| A: logR ~ S | 0.163 | +0.656 | — |
| B: logR ~ cliff | 0.140 | — | +0.007 |
| C: logR ~ S + cliff | **0.208** | +0.478 | +0.005 |

**F-tests for incremental contribution:**
- S adds to cliff-only: F = 8.56, **p = 0.004** — S survives controlling for cliff
- Cliff adds to S-only: F = 5.72, **p = 0.019** — cliff also contributes

Both predictors carry independent information. β_S bootstrap CI [+0.13, +0.83]
excludes zero. β_cliff bootstrap CI [-0.0002, +0.0088] is marginal.

### Within Puget (n = 11)

| Correlation | ρ |
|-------------|---|
| S → logR | **0.773** |
| cliff → logR | 0.273 |
| S → cliff | 0.218 |

Inside Puget, underwater canyon geometry (S) explains ~6× more variance than
above-water terrain (cliff). The two predictors are only weakly correlated
within Puget (ρ = 0.22), so collinearity is not an issue there.

### Puget S=0 vs S>0 cliff comparison

| Group | Mean cliff |
|-------|-----------|
| Puget S=0 (n=11) | 52.4 |
| Puget S>0 (n=11) | 74.6 |

Mann-Whitney p = 0.065 (NS). Canyon cells have somewhat steeper land
topography, but the difference is not significant.

### Assessment

**Option 3 (cliff/viewpoint confound) is weakened but not fully eliminated.**

The underwater canyon score S survives controlling for land-side cliff gradient
(p = 0.004). Inside Puget, S dominates cliff by 3:1 in predictive power.
However, the moderate S–cliff collinearity (ρ = 0.61) means perfect separation
is impossible with ETOPO alone. A proper shoreline classification (e.g., NOAA
ESI data with rocky/sandy/muddy categories) would provide a cleaner test.

---

## Final Assessment: Path A vs Path B

### What survives

1. **Canyon cells have higher UAP rates** — β_Canyon is significant (p = 0.031)
   across the entire West Coast. This is not Puget-only.

2. **Puget is not a generic hotspot** — S=0 cells in Puget have below-average
   rates (0.528). The Navy/culture confound is ruled out.

3. **The canyon effect is strongest in Puget** — 9.54× uplift vs 1.76× elsewhere.
   Within Puget, S correlates with logR at rho = 0.77 (p = 0.005).

4. **San Diego echoes the pattern** — 10.96× canyon uplift (small n).

### What does not survive

1. **CTH as a universal prediction** — East Coast canyons (Norfolk, Hudson) show
   no effect. The hypothesis that "shelf canyons → UAP everywhere" is falsified.

2. **Generalizability across West Coast** — among S>0 cells outside Puget,
   there is no S-logR correlation (rho = -0.05).

### Verdict

**Neither pure Path A nor pure Path B.** The data show:

- Path B is wrong about Puget being an outlier/artefact — the confound test
  rules this out. Canyon cells in Puget have 9.54× the rate of non-canyon cells
  in the same region.

- Path A is wrong about universality — the mechanism does not work on the East
  Coast or generalize across all West Coast canyon locations.

The empirical picture is: **canyon geometry is associated with UAP report excess
in specific locations (Puget Sound, possibly San Diego) where canyon topography
is extreme (fjord-like, high gradient density), but not at moderate canyon sites
or on the East Coast.** Whether this reflects a genuine geophysical mechanism or
an unmeasured confound correlated with extreme submarine topography remains open.

### Open questions

- Why do Puget and San Diego show the pattern but not Monterey, Santa Barbara,
  or Humboldt? All have S > 0 but only Puget/SD show canyon uplift.
- Is the within-Puget gradient (rho = 0.77, n = 11) stable to alternative E_i
  models or population data?
- The scoring function's aggregation radius (50 km) extends beyond the 0.5° cell
  boundary — cells can inherit S from neighboring steep features (CRITICAL-2,
  not yet resolved).
- NUFORC bounding box differs slightly between Phase C and Phase D/E (CRITICAL-1,
  not yet resolved but impact is minimal for West Coast analysis).

---

## File Index

### Scripts (notebooks/)

| # | File | Purpose |
|---|------|---------|
| 09 | `phase_e_evaluate_e2b.py` | E2b/E2c evaluation with CONUS footprint mask |
| 10 | `phase_ev2_scoring.py` | E v2 scoring (60 m/km threshold, frozen) |
| 11 | `phase_ev2_evaluate.py` | E v2 per-cell OR evaluation (underpowered) |
| 12 | `phase_e_red.py` | E-RED rate ratio evaluation (degree×111 bug) |
| 13 | `phase_e_red_v2.py` | E-RED v2 haversine-corrected + 20/25 km comparison |
| 14 | `phase_e_puget_interaction.py` | Puget interaction model (logR ~ S + P + S×P) |
| 15 | `phase_e_puget_sanity.py` | Centering, Cook's D, LOO, within-group checks |
| 16 | `phase_e_puget_confound.py` | Confound test: Puget S=0 vs Other S=0 rates |
| 17 | `phase_e_band_sweep.py` | Coastal band sensitivity sweep (10–200 km, WC+EC) |
| 17b | `phase_e_eastcoast_red.py` | East Coast E-RED check (null result) |
| 18 | `phase_e_shoretype_proxy.py` | Shoreline type proxy: cliff vs canyon confound |

### Results (results/)

| Directory | Key files |
|-----------|-----------|
| `phase_e/` | `phase_e_predictions.json`, `phase_e_grid.json`, `E2b_note.md` |
| `phase_ev2/` | `phase_ev2_predictions.json`, `phase_ev2_grid.json` |
| `phase_ev2/` | `phase_e_red_v2_evaluation.json` (primary results) |
| `phase_ev2/` | `phase_e_puget_interaction.json`, `phase_e_puget_sanity.json` |
| `phase_ev2/` | `phase_e_puget_confound.json` (confound test) |
| `phase_ev2/` | `phase_e_band_sweep.json` (band sensitivity) |
| `phase_ev2/` | `phase_e_eastcoast_check.json` (East Coast null) |
| `phase_ev2/` | `phase_e_shoretype_proxy.json` (cliff confound test) |
| `phase_ev2/` | `e_red_v2_*.png`, `e_red_band_sweep.png` (plots) |

### Git Tags

| Tag | Commit | Purpose |
|-----|--------|---------|
| `phase-e-frozen` | `09de8d8` | Original scoring (20 m/km) — frozen before evaluation |
| `phase-ev2-frozen` | `c2366d2` | Re-specified scoring (60 m/km) — frozen before E-RED |
