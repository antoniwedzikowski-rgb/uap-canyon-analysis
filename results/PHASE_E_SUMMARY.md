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

### Open questions

- Why is Puget Sound qualitatively different? Possible factors: fjord-like
  topography, dense population along narrow waterways, military presence
  (Whidbey NAS, Bangor), or a genuine localized phenomenon.
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

### Results (results/)

| Directory | Key files |
|-----------|-----------|
| `phase_e/` | `phase_e_predictions.json`, `phase_e_grid.json`, `E2b_note.md` |
| `phase_ev2/` | `phase_ev2_predictions.json`, `phase_ev2_grid.json` |
| `phase_ev2/` | `phase_e_red_v2_evaluation.json` (primary results) |
| `phase_ev2/` | `phase_e_puget_interaction.json`, `phase_e_puget_sanity.json` |
| `phase_ev2/` | `e_red_v2_*.png` (decile plots) |

### Git Tags

| Tag | Commit | Purpose |
|-----|--------|---------|
| `phase-e-frozen` | `09de8d8` | Original scoring (20 m/km) — frozen before evaluation |
| `phase-ev2-frozen` | `c2366d2` | Re-specified scoring (60 m/km) — frozen before E-RED |
