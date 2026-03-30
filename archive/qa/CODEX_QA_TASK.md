# Pre-Publication QA Task: preprint_v2 vs preprint_v1

> **Status (2026-03-30):** First QA pass completed. All BLOCKERs from that pass have been fixed in `preprint_v2.md` (see REVISION_MEMO.md §6). This task file is preserved for re-run or independent verification.

## Your role

You are a pre-publication reviewer for an observational geospatial study. Your job is two-fold:

1. **QA the manuscript** (`preprint_v2.md`) for internal consistency, numerical accuracy, and rhetorical discipline.
2. **Compare v2 against v1** (`paper_draft_v1.md`) and assess whether the revision improved research credibility.

---

## Files in this repo

| File | Role |
|------|------|
| `preprint_v2.md` | **Target manuscript** — the document under review |
| `paper_draft_v1.md` | **Baseline** — the previous draft, for comparison |
| `REVISION_MEMO.md` | Author's own revision log — what changed and why |
| `STATISTICAL_AUDIT_REPORT.md` | Independent 47-finding audit of the codebase — **primary source of truth for methodological issues** |
| `results/PHASE_E_SUMMARY.md` | Detailed Phase E results with pre- and post-audit annotations |
| `results/phase_ev2/phase_e_oparea_confound.json` | OPAREA confound test results (readable JSON) |
| `results/phase_ev2/*.json` | Other result files (may have filesystem lock issues; use PHASE_E_SUMMARY as fallback) |
| `README.md` | Repo overview with post-audit numbers |
| `figures/` | All figure PNGs referenced in the manuscript |

---

## Task 1: Numerical QA

For every quantitative claim in `preprint_v2.md`, verify against the source-of-truth hierarchy:

1. `results/phase_ev2/*.json` (if readable)
2. `STATISTICAL_AUDIT_REPORT.md`
3. `results/PHASE_E_SUMMARY.md` (note: some values are pre-audit; check inline annotations)
4. `README.md`

### Specific checks

| Claim location | Claim | Expected source |
|---|---|---|
| Abstract, §4.1 | ρ = 0.374, p = 0.0001, n = 102 | PHASE_E_SUMMARY §Final Results |
| Abstract, §4.1 | CI [0.190, 0.531] | PHASE_E_SUMMARY §Final Results |
| Abstract, §4.1 | exp(β) = 1.93 | PHASE_E_SUMMARY §Final Results |
| §4.1 | 26 canyon cells, 76 flat-shelf | PHASE_E_SUMMARY |
| §4.1 | Canyon O/E = 3.52, flat = 0.97 | PHASE_E_SUMMARY / README |
| §4.3 | Puget uplift 6.8×, SD 9.8×, Monterey 2.4–4.3× | PHASE_E_SUMMARY §Confound Test |
| §4.3 | ρ = 0.243 without Puget, p = 0.021 | PHASE_E_SUMMARY §Final Results |
| §4.4 | Band sweep table (5 rows) | PHASE_E_SUMMARY §Band Sensitivity Sweep |
| §4.5 | Confound F-test table (6 rows) | PHASE_E_SUMMARY §Shoretype/Ocean/Magnetic sections |
| §4.5 | Magnetic: S>0 mean 47 nT, S=0 mean 91 nT, MW p = 0.0002 | PHASE_E_SUMMARY §Magnetic |
| §4.6 | Puget: F = 6.8, p = 0.018; OPAREA: F = 2.9, p = 0.10 | phase_e_oparea_confound.json → regional_breakdown.Puget |
| §4.6 | Central CA: F = 3.9, p = 0.056 | phase_e_oparea_confound.json → regional_breakdown.Central_CA |
| §4.6 | Monterey: 127–192 km, logR = 0.75 vs −0.02 | phase_e_oparea_confound.json → monterey_natural_experiment |
| §4.7 | Temporal replication table (7 rows) | PHASE_E_SUMMARY §Replication Suite (post-audit annotation) |
| §4.7 | Post-2014: ρ = 0.350, p = 0.0001, n = 119 | PHASE_E_SUMMARY §6e (post-audit note) |
| §4.7 | Rolling windows: 21/21 positive, 18/21 sig | README §Replication (post-audit); cf. PHASE_E_SUMMARY pre-audit = 15/21 |
| §4.8 | LOO CV table (5 folds), mean ρ = 0.315 | STATISTICAL_AUDIT_REPORT CRIT-2 resolution |
| §4.8 | Spatial forward: Puget ρ = 0.604, SoCal ρ = 0.470 | STATISTICAL_AUDIT_REPORT CRIT-3 resolution |
| §4.9 | Threshold sweep: ρ 0.374–0.416, all p < 0.0002 | STATISTICAL_AUDIT_REPORT CRIT-4 resolution |
| §4.10 | East Coast: ρ = 0.055, p = 0.459, 2/185 cells | PHASE_E_SUMMARY §East Coast |
| §4.11 | Norway: ρ = 0.49, p = 0.047, then logistic OR = 1.03, p = 0.76 | PHASE_E_SUMMARY §Norway |
| Appendix A | LOO pre-correction = 0.38, post = 0.315 | STATISTICAL_AUDIT_REPORT CRIT-2 |
| Appendix A | Post-2014 pre-fix ρ = 0.28, post-fix = 0.35 | STATISTICAL_AUDIT_REPORT CRIT-1 |

**Output format for each claim:**
```
CLAIM: [exact claim text]
SOURCE: [file, line/section]
STATUS: CONFIRMED | UNVERIFIABLE | DISCREPANCY
NOTE: [if discrepancy, what the source says]
```

---

## Task 2: Rhetorical Audit

Check every section of `preprint_v2.md` against these criteria:

### 2a. Causal language leakage
Search for any remaining instances of:
- "drives" / "driven by" (when referring to canyon → UAP, not statistical variance)
- "causes" / "caused by"
- "explains" (when it should be "predicts" or "is associated with")
- "independent replication" or "independent data" referring to post-2014 holdout
- "a priori" without qualification
- "fully reproducible" / "all data sources" without specifics

### 2b. Overclaiming
Flag any sentence where the claim exceeds what the cited test supports. Common patterns:
- Claiming replication when it's temporal stability with shared geocoding
- Claiming confound elimination when the test is regional or underpowered
- Claiming the effect "survives" a test that was actually marginal (p near 0.05)
- Using "robust" without specifying to what

### 2c. Underclaiming / excessive hedging
Flag any sentence where the hedging undermines a result that is well-supported. The goal is scientific sobriety, not self-sabotage.

### 2d. Missing caveats
For each major result, check whether the relevant limitation from `STATISTICAL_AUDIT_REPORT.md` is acknowledged nearby (within the same section or with a cross-reference to Limitations).

---

## Task 3: v1 → v2 Comparison

Read both manuscripts and produce a structured comparison:

### 3a. Deletion inventory
List every substantive claim, section, or argument present in v1 but removed in v2. For each, assess:
- Was the deletion justified? (Y/N/PARTIAL)
- Was the deleted content replaced by something better, or just removed?

### 3b. Addition inventory
List every substantive claim, section, or caveat added in v2 that was absent in v1. For each:
- Does the addition improve accuracy? (Y/N)
- Does the addition improve credibility? (Y/N)

### 3c. Softening inventory
List every claim that was softened (weaker language, added caveats). For each:
- Was the softening warranted by the data? (Y/N)
- Did the softening go too far? (Y/N)

### 3d. Figure changes
Compare figure assignments. Were the right figures promoted to main text?

### 3e. Overall assessment

Score each manuscript on these dimensions (1–5 scale):

| Dimension | v1 | v2 | Comment |
|---|---|---|---|
| Numerical accuracy | | | Numbers match source-of-truth files |
| Rhetorical discipline | | | No overclaiming, no causal language leakage |
| Limitation honesty | | | Known weaknesses acknowledged, proportionate |
| Structural clarity | | | Professional manuscript structure, clear sections |
| Reproducibility claims | | | Data/code availability matches reality |
| Overall credibility | | | Would a skeptical reviewer trust this document? |

**Final verdict:** Does v2 have higher research value than v1? Specifically:
- Would v2 survive a more hostile peer review than v1?
- Are there remaining issues that should be fixed before submission?
- Is there anything v1 did better that v2 lost?

---

## Task 4: Remaining Issues

List any problems found during QA that should be fixed before the manuscript is submitted. Classify each as:

- **BLOCKER** — must fix before submission (factual error, unsupported claim)
- **IMPORTANT** — should fix (ambiguous wording, missing caveat)
- **MINOR** — nice to fix (style, formatting, redundancy)

---

## Output structure

Return your findings as a single Markdown document with these sections:
1. `## Numerical QA Results` (table of all claims checked)
2. `## Rhetorical Audit` (flagged sentences with rationale)
3. `## v1 → v2 Comparison` (structured comparison per §3a–3e)
4. `## Remaining Issues` (prioritized fix list)
5. `## Verdict` (one paragraph)
