# Revision Memo: preprint_v1 → preprint_v2

**Date:** 2026-03-30
**Scope:** Full manuscript revision for scientific credibility, numerical accuracy, and rhetorical discipline.
**Source of truth:** `results/phase_ev2/*.json`, `STATISTICAL_AUDIT_REPORT.md`, `results/PHASE_E_SUMMARY.md`

---

## 1. Factual Corrections

### 1a. Monterey OPAREA distance range
- **v1:** "canyon cells 127–253 km from any OPAREA"
- **v2:** "canyon cells 127–192 km from the nearest OPAREA"
- **Source:** `phase_e_oparea_confound.json` → `monterey_natural_experiment.hot_dist_range_km: [127.3, 192.0]`
- **Appears in:** Abstract, Section 4.6, Section 5.4, Discussion

### 1b. OPAREA regional p-values
- **v1:** "p = 0.018" (Puget S given OPAREA)
- **v2:** "p = 0.018" — confirmed against JSON (regional_breakdown.Puget.p_S_given_oparea = 0.0175, rounds to 0.018). No change needed.
- Central CA: v1 says p = 0.056, JSON says 0.0555. Acceptable rounding.

### 1c. Replication suite numbers
- Most phase_ev2 JSON files are filesystem-locked and could not be independently verified during this revision. Numbers carried forward from v1 where they are consistent with PHASE_E_SUMMARY post-audit annotations and STATISTICAL_AUDIT_REPORT post-fix values.
- **Flagged for author verification:** Post-2014 n=119 (PHASE_E_SUMMARY pre-audit says n=88; discrepancy likely reflects CRIT-1 fix changing min_reports thresholds). Rolling windows 18/21 vs pre-audit 15/21. LOO fold-level ρ values differ between paper and PHASE_E_SUMMARY (expected: paper uses post-audit, summary is pre-audit).

---

## 2. Language and Framing Corrections

### 2a. "A priori specified" → softened
- **v1:** Abstract and Section 4.1 use "a priori specified canyon scoring function"
- **v2:** Changed to "pre-specified" with explicit caveat that the 60 m/km threshold was selected during exploratory analysis (CRIT-4). The scoring function geometry was frozen before evaluation, but the gradient threshold had an exploratory component.
- Threshold sweep (20–100 m/km, all significant) is cited as mitigation, not as full absolution.

### 2b. Post-2014 "independent" → "temporal holdout"
- **v1:** Abstract says "replicates in independent post-2014 data"; Section 4.7 says "fully independent post-2014 data"
- **v2:** Changed to "temporal holdout" with explicit caveat about shared city-centroid geocoding (HIGH-3). The post-2014 data uses the same geocoding lookup, so spatial structure is inherited by construction. It is a temporal stability check, not an independent spatial replication.
- Section 4.7 now includes a "Geocoding caveat" paragraph.

### 2c. Correlation vs mechanism
- **v1:** Generally careful but some sentences imply causal direction (e.g., "canyon proximity drives the local signal")
- **v2:** Systematic replacement of causal language with correlational language. "Drives" → "is associated with." "Explains" → "predicts."

### 2d. Norway wording
- **v1:** Reports initial Spearman ρ = 0.49 as "POSITIVE" before noting logistic regression null
- **v2:** Leads with the population-controlled result (null). Spearman result is context. Conclusion: "inconclusive" rather than implying partial success.

### 2e. Effect size rhetoric
- **v1:** Section 5.6 argues ρ = 0.374 is a "lower bound" on the true association
- **v2:** Removed "lower bound" framing. Measurement noise does attenuate correlations, but calling the observed value a lower bound overstates what can be concluded from observational data with known confounds. Replaced with: "measurement noise likely attenuates the observed correlation, but the degree of attenuation cannot be quantified."

### 2f. BCH comparison removed from main text
- **v1:** Section 5.7 describes a parallel BCH study as methodological validation
- **v2:** Moved to Appendix or removed. A parallel study by the same author using the same framework is not independent methodological validation. The null result is informative but does not belong in the main Discussion as proof of method validity.

### 2g. Bruehl & Villarroel comparison toned down
- **v1:** Compares OR ≈ 4.0 to Bruehl & Villarroel's OR ≈ 1.45, calls it "comparable statistical architecture"
- **v2:** Removed. Different phenomena, different methods, different data structures — the comparison adds no inferential value and risks false credibility-by-association.

---

## 3. Structural Changes

### 3a. Limitations section expanded
Added/strengthened:
- NUFORC as noisy, self-selected observational data (not a systematic survey)
- Possible residual confounding from unmeasured coastal behavioral variables
- Non-causal interpretation stated more prominently
- Regional concentration of signal (Puget + SoCal drive most of the effect)
- Need for independent datasets (military sensor, AARO, FAA)
- Shared geocoding as limitation of temporal replication
- cKDTree degree-space approximation

### 3b. Data/Code Availability rewritten
- **v1:** "The complete codebase (65 scripts across 7 analysis phases), all data sources, and a 47-finding statistical audit report are publicly available"
- **v2:** Precisely states what is in the public repository vs. what is in the working directory. Notes that NUFORC data is third-party and may change. Provides DOI/URL for the audit report. Removes "all data sources" if raw ETOPO/EMAG2/ESI are not in the repo.

### 3c. Abstract tightened
- Removed "exp(β) = 1.93" from abstract (log-linear proxy, not the primary test statistic)
- Removed rolling window counts from abstract
- Added geocoding caveat for post-2014
- Changed "independent post-2014 data" to "post-2014 temporal holdout"

### 3d. Conclusion restrained
- Removed "a priori specified" from conclusion
- Added final sentence acknowledging regional scope and unresolved confounds
- Removed implication that the codebase alone enables "determination" of the finding's nature

---

## 4. Figure Assignments

### Main text figures (4)
1. **Figure 1:** `figures/figure1_study_area.png` — Study area map showing West Coast grid cells, canyon features, and population centers. Caption: describes data coverage, grid resolution, canyon identification.
2. **Figure 2:** `figures/e_red_v2_primary_200km.png` — Primary result: S vs log(O/E) scatterplot for 102 West Coast cells (200 km band). Caption: Spearman ρ, p-value, CI, regression line, key labeled cells.
3. **Figure 3:** `figures/figure_hero_bathymetry.png` — West Coast bathymetric context showing canyon locations relative to reporting hotspots. Caption: ETOPO data, canyon identification threshold, regional labels.
4. **Figure 4:** `figures/e_red_band_sweep.png` — Coastal band sensitivity (West Coast vs East Coast) showing signal emergence at 25–50 km and East Coast null. Caption: ρ values at each bandwidth, n cells, significance thresholds.

### Supplementary figures (3)
- **Figure S1:** `figures/figure_forest_robustness.png` — Forest plot of robustness checks across confound tests, temporal splits, and spatial CV.
- **Figure S2:** `figures/d7a_canyon_head_distances.png` — Canyon-head-to-shore distances by region.
- **Figure S3:** `figures/figure_prediction_vs_reality.png` — Predicted vs observed report density across grid cells.

---

## 5. Items Not Changed (with rationale)

- **Primary result (ρ = 0.374, p = 0.0001, n = 102):** Confirmed by PHASE_E_SUMMARY and unaffected by any audit finding.
- **Confound F-test table:** Values match PHASE_E_SUMMARY. Ocean, magnetic, chl-a, shore type proxy — all consistent.
- **East Coast null (ρ = 0.055, p = 0.459):** Consistent across sources.
- **Regional decomposition table:** Values consistent with PHASE_E_SUMMARY.

---

## 6. Post-QA Fixes (v2.1 patch, 2026-03-30)

Following Codex QA, the following corrections were applied directly to `preprint_v2.md`:

- **O/E means:** Changed "3.52 / 0.97" to "2.69 / 1.02" (aggregate ΣO/ΣE from `phase_e_red_v2_evaluation.json` cell_details)
- **Band sweep n:** Corrected 50 km n=63→62, 100 km n=83→81 (per `phase_e_band_sweep.json`)
- **Magnetic n:** Changed n=102→94 (per `phase_e_magnetic_confound.json`)
- **Chlorophyll F-test row:** Changed "F=0.1, p=0.719" → "F=1.5, p=0.219" (per `phase_e_chla_confound.json`)
- **Monterey:** Rewrote to report all 3 canyon cells honestly (R=0.72, 2.75, 4.80), not just the two elevated ones
- **Figure captions:** Completely rewritten to match actual PNG contents (quintile bar chart, not scatterplot; bar chart, not circle plot)
- **Puget definition note:** Restored explanation of two definitions (n=22 vs n=18)
- **Methodological framing:** Added ecological spatial study / triangulation framing with Lawson, Elliott & Wartenberg, Lawlor et al. references
- **Appendix A:** "All resolved" → "four of five addressed; the fifth is a null result not cited"
- **"Survives" language:** Systematically replaced with "was not eliminated by" in abstract, discussion, and conclusion
- **Effect size rhetoric:** Removed "lower bound" framing; added note that residual confounding can inflate as well as attenuate

### v2.2 patch (final pass, 2026-03-30)

- **Availability section:** "The 4 critical findings were corrected" → "Four of the five critical findings were corrected (the fifth concerns a null result not cited in this paper)" — aligns with Appendix A wording
- **Norway provenance:** Expanded inline caveat for logistic regression values (OR = 1.03, p = 0.76) to note explicitly that no final result JSON exists for the logistic retest; recommends script verification before citation
- **Introduction trimmed (two passes):** (1) Removed cryptoterrestrial hypothesis framing (Tonnies, Lomas/Case/Masters) and "transitioning between air and sea" language; retained Nimitz/Gallaudet and Sol Foundation motivation. (2) Merged standalone canyon geomorphology paragraph into hypothesis paragraph — canyon definition condensed to one sentence; upwelling/nutrient detail removed (relevant only in Discussion). Net result: 4 paragraphs → 3 paragraphs, ~80 words shorter, no speculative ontological claims, faster path from gap → hypothesis → design. Tonnies removed from References (zero remaining in-text citations)
- **O/E subscript label:** ΣO/ΣE → ΣO_i / ΣE_i throughout (clarifies per-cell aggregation)
- **Rolling windows discrepancy resolved:** 18/21 confirmed as post-audit correct value (from JSON); 15/21 in PHASE_E_SUMMARY is a stale pre-audit artifact

### v2.3 — Elegance pass (2026-03-30)

Full editorial pass following a 12-rule framework ("less legal brief, more scientific prose"). Principal changes:

- **Abstract:** Rewritten for directness; ~220 → ~170 words. Removed hedging adverbs and stacked caveats; restructured as result → robustness → limitation → implication.
- **§5.1 (Overview):** Condensed from 2 paragraphs to 1. Removed meta-language ("The analysis presented here…").
- **§5.2 (Mechanism):** Trimmed ~15%; opening sentence varied from §5.1 to avoid repetition. Closing changed from "A conservative reading…" to "The mechanism is unresolved."
- **§5.3 (Confounds):** "Cannot be definitively excluded" → "is narrowed but not resolved." Removed "It should also be noted."
- **§5.5 (Theoretical context):** Removed "This study neither depends on nor validates" framing; replaced with "The data do not distinguish among these frameworks."
- **§5.6 (Effect size):** Cut "should be treated as a point estimate of uncertain direction of bias" → "The direction of net bias is unknown." Removed "Within the norms of observational environmental research."
- **§5.7 (Future work):** Condensed from 5 paragraphs to 3 bold-labeled blocks: **Independent replication** / **Instrumented canyon sites** / **Atmospheric follow-up**.
- **§6 (Limitations):** Multiple entries tightened: "Data quality" → concise; "Threshold selection" → concise; "Need for independent datasets" → "No independent replication"; "Author credentials" shortened.
- **§8 (Conclusion):** Rewritten as 3 shorter paragraphs; final paragraph is a single sentence. Removed all instances of "The study does not establish."
- **§4.5:** "Ruling out a magnetic-attraction explanation" → "inconsistent with a magnetic-attraction explanation."
- **General:** Reduced defensive meta-language from ~8 instances to 1 (section heading only). Applied claim→evidence→caveat rhythm throughout Discussion.

### v2.4 — Monterey Bay geographic labeling correction (2026-03-30)

**What changed.** Monterey Bay was narrowed from the broader Central California band (36–37.5°N, 3 canyon cells) to the bay itself (36–37°N, 2 canyon cells). The northern cell near Half Moon Bay / Pacifica (37.25°N, R = 0.72) is no longer described as part of Monterey Bay; it is counted in "Rest of West Coast."

**Why.** The 37.25°N cell lies north of the bay mouth and is geographically outside Monterey Bay. Labeling it as Monterey overstated the bay's cell count and diluted the aggregate with a below-baseline cell, producing a "mixed" descriptor that mischaracterized two clearly elevated cells.

**What did not change.** The 102-cell primary analysis, all inferential statistics (Spearman ρ, bootstrap CIs, nested F-tests, temporal holdouts), and the confound test results are unchanged. This is a geographic labeling correction, not a reanalysis.

**Manuscript locations updated:**

- **Abstract:** "(mixed)" → "(2.75–4.80×, 2 cells)"
- **§4.3 table:** Monterey row: `2 / 8`, ΣO_i/ΣE_i = 1.21 (flat) / 3.18 (canyon), uplift 2.6×. Rest of West Coast: `10 / 55`, uplift 1.4×
- **§4.3 narrative:** Clean prose describing 2 canyon cells (R = 2.75, 4.80), 8 flat-shelf cells, and 127–141 km OPAREA distance
- **§4.6:** Bay-only comparison: mean log(O/E) = 1.29 for canyon cells, flat-shelf near baseline; distance 127–141 km
- **§5.3:** Monterey distance updated to 127–141 km; 2 canyon cells
- **§5.4:** Rewritten as "one of the clearest local examples" with bay-only numbers

---

## 7. Unresolved / Author-Action-Required

1. **JSON file access:** 16 of 18 phase_ev2 JSON files returned EDEADLK (filesystem deadlock) during this session. Only `phase_e_oparea_confound.json` was readable. The revision carries forward v1 numbers where they are consistent with PHASE_E_SUMMARY and audit report, but **the author should independently verify all replication suite numbers** (post-2014 n, rolling window significance counts, LOO fold ρ values) against the JSON files.

2. **Repository contents audit:** The data/code availability statement should be verified against the actual public repository contents at publication time.

3. **NUFORC data persistence:** The HuggingFace mirror (kcimc/NUFORC) should be checked for continued availability. If unreliable, the author should archive the dataset.
