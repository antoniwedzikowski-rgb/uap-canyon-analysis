# Phase E Evaluation: Design Fix Log

## E2a (original evaluation)

Global shelf scoring selected top-20 hotspots outside the NUFORC data footprint
(Bahamas, Cuba, British Columbia, Isla Guadalupe). All 20 hot predictions had
zero NUFORC reports. Test was non-measurable — not failed, but unevaluable.

Cold predictions (S=0 coastal cells) scored 9/9 where evaluable, but trivially:
S=0 means no steep bathymetry, so OR in the 60+ gradient bin is structurally zero.

Spearman(S, OR) = 0.342 (p=0.08, n=27) across the few evaluable cells — positive
direction but underpowered.

## E2b (design fix: CONUS footprint mask)

Same scoring function, same ranks, same evaluation thresholds. Only change:
ranking restricted to the NUFORC data footprint (CONUS bounding box:
24.5–49.0°N, −125.0 to −66.0°W, within 200km coastal band).

**E2b changes only the geographic evaluation mask to match the dataset footprint.
The scoring function, ranks, and evaluation thresholds are unchanged from the
Phase E freeze tag (`phase-e-frozen`, commit `09de8d8`).**

## E2c (measurability mask)

Same as E2b, plus: grid cells with <10 total NUFORC reports are excluded from
both hot and cold candidate pools before ranking. This is a measurability
constraint (OR estimates are undefined/unstable below ~10 reports), not a
model-tuning step.

E2c is reported as a robustness check alongside E2b. If results differ
substantially between E2b and E2c, the signal is likely driven by reporting
density rather than geometry.

## What is frozen (unchanged from `phase-e-frozen`)

- Scoring function: S = mean(rank_G + rank_P + rank_C) for steep cells within 50km
- Global ranking across all CONUS shelf steep cells
- Spatial dedup: greedy ≥200km
- Hot hit definition: lower CI > 1.0
- Cold hit definition: OR < 1.0 AND upper CI < 1.5
- All geometric parameters (gradient threshold 20 m/km, shelf 0 to −500m,
  proximity e-fold 50km, coast complexity radius 25km, aggregation radius 50km)
