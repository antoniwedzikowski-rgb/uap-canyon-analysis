#!/usr/bin/env python3
"""
Phase E: Puget Confound Test
==============================
Question: Is Puget Sound's elevated UAP rate driven by canyon geometry,
or is it a regional confound (Navy bases, reporting culture, etc.)?

Test logic:
  If Puget has elevated UAP/capita EVEN in cells WITHOUT canyons (S=0),
  then the S×Puget interaction is a confound — Puget is just "hot" overall.

  If Puget's rate is normal where S=0 and elevated only where S>0,
  then canyon geometry is genuinely associated with the excess.

Design:
  1. Define comparable coastal regions (similar pop density, coastal character)
  2. Compare raw rate (O_i / E_i) between:
     a) Puget S>0 cells
     b) Puget S=0 cells
     c) Other West Coast S>0 cells
     d) Other West Coast S=0 cells
  3. Formal 2×2 interaction: Region(Puget/Other) × Canyon(S>0/S=0) → logR

Uses haversine-corrected cell_details from E-RED v2 primary 200km.
"""

import json, os
import numpy as np
from numpy.linalg import lstsq
from scipy.stats import spearmanr, mannwhitneyu, ttest_ind
from scipy.stats import f as f_dist

BASE_DIR = os.environ.get("UAP_BASE_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ============================================================
# LOAD
# ============================================================
with open(os.path.join(BASE_DIR, "phase_ev2", "phase_e_red_v2_evaluation.json")) as f:
    data = json.load(f)

cells = data["primary_200km"]["cell_details"]
n = len(cells)

S_arr    = np.array([c['S'] for c in cells])
logR_arr = np.array([c['logR'] for c in cells])
R_arr    = np.array([c['R_i'] for c in cells])
O_arr    = np.array([c['O_i'] for c in cells])
E_arr    = np.array([c['E_i'] for c in cells])
lat_arr  = np.array([c['lat'] for c in cells])
lon_arr  = np.array([c['lon'] for c in cells])

print(f"Loaded {n} testable cells (primary 200km, haversine-corrected)\n")

# ============================================================
# DEFINE REGIONS
# ============================================================
# Puget Sound region: broad definition (entire Salish Sea / WA coast)
# Using lat 46-50, lon < -121 (covers Puget, Whidbey, San Juans, Olympia)
puget_region = (lat_arr >= 46) & (lat_arr <= 50) & (lon_arr <= -121)

# For comparison: other coastal regions with significant Navy/military
# San Diego (32-33.5N) — Naval Base San Diego, massive Navy presence
sandiego_region = (lat_arr >= 32) & (lat_arr < 33.5) & (lon_arr <= -116)

# SF Bay Area (37-38.5N) — no major canyon score, large coastal metro
sfbay_region = (lat_arr >= 37) & (lat_arr < 38.5) & (lon_arr <= -121)

# Oregon coast (43-46N) — similar Pacific NW character, less Navy
oregon_region = (lat_arr >= 43) & (lat_arr < 46) & (lon_arr <= -122)

# SoCal (33.5-35.5N) — mixed Navy, large population
socal_region = (lat_arr >= 33.5) & (lat_arr < 35.5) & (lon_arr <= -117)

# Everything else on West Coast
other_wc = ~puget_region

# Canyon presence
has_canyon = S_arr > 0

print("=" * 70)
print("REGION DEFINITIONS")
print("=" * 70)

regions = {
    'Puget (46-50N)': puget_region,
    'San Diego (32-33.5N)': sandiego_region,
    'SF Bay (37-38.5N)': sfbay_region,
    'Oregon (43-46N)': oregon_region,
    'SoCal (33.5-35.5N)': socal_region,
    'Other West Coast': other_wc,
}

for name, mask in regions.items():
    n_cells = mask.sum()
    n_s_pos = (mask & has_canyon).sum()
    n_s_zero = (mask & ~has_canyon).sum()
    print(f"  {name:25s}: {n_cells:3d} cells (S>0: {n_s_pos:2d}, S=0: {n_s_zero:2d})")


# ============================================================
# TEST 1: Four-group comparison (Puget vs Other × Canyon vs No Canyon)
# ============================================================
print(f"\n{'='*70}")
print("TEST 1: FOUR-GROUP RATE COMPARISON")
print(f"{'='*70}")

groups = {
    'Puget + Canyon (S>0)':    puget_region & has_canyon,
    'Puget + No Canyon (S=0)': puget_region & ~has_canyon,
    'Other + Canyon (S>0)':    other_wc & has_canyon,
    'Other + No Canyon (S=0)': other_wc & ~has_canyon,
}

print(f"\n  {'Group':30s} {'n':>4} {'mean_R':>8} {'med_R':>8} {'mean_logR':>10} {'sum_O':>7} {'sum_E':>8}")
print(f"  {'-'*80}")

group_stats = {}
for name, mask in groups.items():
    if mask.sum() == 0:
        print(f"  {name:30s} {'(empty)':>4}")
        continue
    mr = R_arr[mask].mean()
    medr = np.median(R_arr[mask])
    mlr = logR_arr[mask].mean()
    so = O_arr[mask].sum()
    se = E_arr[mask].sum()
    agg_rate = so / se if se > 0 else 0
    group_stats[name] = {
        'n': int(mask.sum()),
        'mean_R': round(float(mr), 3),
        'median_R': round(float(medr), 3),
        'mean_logR': round(float(mlr), 3),
        'sum_O': int(so),
        'sum_E': round(float(se), 1),
        'aggregate_rate': round(agg_rate, 3),
    }
    print(f"  {name:30s} {mask.sum():4d} {mr:8.2f} {medr:8.2f} {mlr:10.3f} {so:7d} {se:8.1f}")

# Aggregate rate (sum O / sum E) for each group
print(f"\n  Aggregate rate (ΣO/ΣE) — the cleanest measure:")
for name, stats in group_stats.items():
    bar = '█' * int(stats['aggregate_rate'] * 10)
    print(f"    {name:30s}: {stats['aggregate_rate']:.3f}  {bar}")


# ============================================================
# TEST 2: THE CRITICAL COMPARISON
# ============================================================
print(f"\n{'='*70}")
print("TEST 2: CRITICAL CONFOUND CHECK")
print(f"{'='*70}")

# If Puget S=0 cells have elevated R compared to Other S=0 cells → confound
puget_s0 = puget_region & ~has_canyon
other_s0 = other_wc & ~has_canyon

if puget_s0.sum() >= 3 and other_s0.sum() >= 3:
    R_puget_s0 = R_arr[puget_s0]
    R_other_s0 = R_arr[other_s0]

    print(f"\n  Puget S=0 cells (n={puget_s0.sum()}):")
    print(f"    mean R = {R_puget_s0.mean():.3f}, median R = {np.median(R_puget_s0):.3f}")
    print(f"    ΣO/ΣE = {O_arr[puget_s0].sum() / E_arr[puget_s0].sum():.3f}")
    for i in np.where(puget_s0)[0]:
        print(f"      ({lat_arr[i]:.1f}, {lon_arr[i]:.1f}) O={O_arr[i]:4d} E={E_arr[i]:6.1f} R={R_arr[i]:.2f}")

    print(f"\n  Other WC S=0 cells (n={other_s0.sum()}):")
    print(f"    mean R = {R_other_s0.mean():.3f}, median R = {np.median(R_other_s0):.3f}")
    print(f"    ΣO/ΣE = {O_arr[other_s0].sum() / E_arr[other_s0].sum():.3f}")

    # Mann-Whitney U test: Puget S=0 vs Other S=0
    U, p_mw = mannwhitneyu(R_puget_s0, R_other_s0, alternative='greater')
    print(f"\n  Mann-Whitney U (Puget S=0 > Other S=0): U={U:.0f}, p={p_mw:.4f}")

    # Welch t-test on logR
    t_stat, p_t = ttest_ind(logR_arr[puget_s0], logR_arr[other_s0], equal_var=False)
    print(f"  Welch t-test on logR: t={t_stat:.3f}, p={p_t:.4f}")

    if p_mw < 0.05:
        print(f"\n  → CONFOUND SIGNAL: Puget S=0 cells have significantly higher R")
        print(f"    than other WC S=0 cells. The region is 'hot' regardless of canyons.")
        confound_verdict = "CONFOUND"
    else:
        print(f"\n  → NO CONFOUND: Puget S=0 cells are NOT significantly elevated.")
        print(f"    Canyon geometry is needed for the excess.")
        confound_verdict = "NO_CONFOUND"
else:
    print("  Insufficient cells for comparison")
    confound_verdict = "INSUFFICIENT_DATA"


# ============================================================
# TEST 3: 2×2 INTERACTION MODEL
# ============================================================
print(f"\n{'='*70}")
print("TEST 3: 2×2 INTERACTION MODEL (Region × Canyon → logR)")
print(f"{'='*70}")

# Region indicator: 1 = Puget, 0 = Other
# Canyon indicator: 1 = S>0, 0 = S=0
R_ind = puget_region.astype(float)
C_ind = has_canyon.astype(float)
RC_ind = R_ind * C_ind  # interaction

X = np.column_stack([np.ones(n), C_ind, R_ind, RC_ind])
b, _, _, _ = lstsq(X, logR_arr, rcond=None)

print(f"\n  logR ~ Canyon + Puget + Canyon×Puget")
print(f"  α (intercept)      = {b[0]:+.4f}  (baseline: Other, S=0)")
print(f"  β_Canyon            = {b[1]:+.4f}  (canyon effect outside Puget)")
print(f"  β_Puget             = {b[2]:+.4f}  (Puget effect in S=0 cells)")
print(f"  β_Canyon×Puget      = {b[3]:+.4f}  (extra canyon effect IN Puget)")

# Cell means
print(f"\n  Predicted cell means:")
print(f"    Other, S=0:   logR = {b[0]:.3f}           → R = {np.exp(b[0]):.2f}")
print(f"    Other, S>0:   logR = {b[0]+b[1]:.3f}      → R = {np.exp(b[0]+b[1]):.2f}")
print(f"    Puget, S=0:   logR = {b[0]+b[2]:.3f}      → R = {np.exp(b[0]+b[2]):.2f}")
print(f"    Puget, S>0:   logR = {b[0]+b[1]+b[2]+b[3]:.3f}  → R = {np.exp(b[0]+b[1]+b[2]+b[3]):.2f}")

# F-test for each term
yhat = X @ b
ss_res_full = np.sum((logR_arr - yhat)**2)
df_full = n - 4

print(f"\n  F-tests for each term:")
for drop_idx, name in [(1, 'Canyon'), (2, 'Puget'), (3, 'Canyon×Puget')]:
    X_reduced = np.delete(X, drop_idx, axis=1)
    b_red, _, _, _ = lstsq(X_reduced, logR_arr, rcond=None)
    ss_res_red = np.sum((logR_arr - X_reduced @ b_red)**2)
    F_val = ((ss_res_red - ss_res_full) / 1) / (ss_res_full / df_full)
    p_val = 1 - f_dist.cdf(F_val, 1, df_full)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'NS'
    print(f"    {name:20s}: F({1},{df_full}) = {F_val:6.2f}, p = {p_val:.4f} {sig}")

# Bootstrap CIs
N_BOOT = 5000
rng = np.random.RandomState(42)
boot_coefs = []
for _ in range(N_BOOT):
    idx = rng.choice(n, n, replace=True)
    try:
        bb, _, _, _ = lstsq(X[idx], logR_arr[idx], rcond=None)
        boot_coefs.append(bb)
    except:
        pass
boot_coefs = np.array(boot_coefs)

names = ['α', 'β_Canyon', 'β_Puget', 'β_Canyon×Puget']
print(f"\n  Bootstrap 95% CIs:")
for i, name in enumerate(names):
    lo = np.percentile(boot_coefs[:, i], 2.5)
    hi = np.percentile(boot_coefs[:, i], 97.5)
    print(f"    {name:20s}: {b[i]:+.4f}  [{lo:+.4f}, {hi:+.4f}]")


# ============================================================
# TEST 4: COMPARISON WITH SIMILAR NAVY REGIONS
# ============================================================
print(f"\n{'='*70}")
print("TEST 4: PUGET vs OTHER NAVY REGIONS (San Diego)")
print(f"{'='*70}")

for rname, rmask in [('Puget', puget_region), ('San Diego', sandiego_region),
                      ('SF Bay', sfbay_region), ('Oregon', oregon_region),
                      ('SoCal', socal_region)]:
    n_r = rmask.sum()
    if n_r == 0:
        continue
    s_pos = (rmask & has_canyon).sum()
    s_zero = (rmask & ~has_canyon).sum()
    agg_rate = O_arr[rmask].sum() / E_arr[rmask].sum() if E_arr[rmask].sum() > 0 else 0
    agg_rate_s0 = (O_arr[rmask & ~has_canyon].sum() / E_arr[rmask & ~has_canyon].sum()
                   if (rmask & ~has_canyon).sum() > 0 and E_arr[rmask & ~has_canyon].sum() > 0 else None)
    agg_rate_s1 = (O_arr[rmask & has_canyon].sum() / E_arr[rmask & has_canyon].sum()
                   if (rmask & has_canyon).sum() > 0 and E_arr[rmask & has_canyon].sum() > 0 else None)

    print(f"\n  {rname} (n={n_r}, S>0:{s_pos}, S=0:{s_zero}):")
    print(f"    Overall ΣO/ΣE = {agg_rate:.3f}")
    if agg_rate_s0 is not None:
        print(f"    S=0 cells ΣO/ΣE = {agg_rate_s0:.3f}")
    if agg_rate_s1 is not None:
        print(f"    S>0 cells ΣO/ΣE = {agg_rate_s1:.3f}")
    if agg_rate_s0 is not None and agg_rate_s1 is not None:
        ratio = agg_rate_s1 / agg_rate_s0 if agg_rate_s0 > 0 else float('inf')
        print(f"    Canyon uplift (S>0 / S=0): {ratio:.2f}×")


# ============================================================
# VERDICT
# ============================================================
print(f"\n{'='*70}")
print("VERDICT")
print(f"{'='*70}")

puget_s0_agg = O_arr[puget_s0].sum() / E_arr[puget_s0].sum() if puget_s0.sum() > 0 else 0
puget_s1_agg = O_arr[puget_region & has_canyon].sum() / E_arr[puget_region & has_canyon].sum()
other_s0_agg = O_arr[other_s0].sum() / E_arr[other_s0].sum()
other_s1_agg = O_arr[other_wc & has_canyon].sum() / E_arr[other_wc & has_canyon].sum() if (other_wc & has_canyon).sum() > 0 else 0

print(f"\n  Rate table (ΣO / ΣE):")
print(f"  {'':20s} {'S=0':>8} {'S>0':>8} {'Uplift':>8}")
print(f"  {'-'*48}")
print(f"  {'Puget':20s} {puget_s0_agg:8.3f} {puget_s1_agg:8.3f} {puget_s1_agg/puget_s0_agg if puget_s0_agg > 0 else 0:8.2f}×")
print(f"  {'Other WC':20s} {other_s0_agg:8.3f} {other_s1_agg:8.3f} {other_s1_agg/other_s0_agg if other_s0_agg > 0 else 0:8.2f}×")

print(f"\n  Confound test result: {confound_verdict}")

if confound_verdict == "CONFOUND":
    print(f"""
  INTERPRETATION:
    Puget S=0 cells already have elevated rates compared to other WC S=0 cells.
    The entire Puget region has more UAP reports per capita than expected,
    regardless of canyon presence. The S×Puget interaction is likely driven by
    a regional confound (Navy bases, reporting culture, observer density) rather
    than canyon geometry per se.

    This favors Path B: the CTH prediction is falsified in its general form.
    The Puget signal may be real but is not attributable to canyons.""")
elif confound_verdict == "NO_CONFOUND":
    print(f"""
  INTERPRETATION:
    Puget S=0 cells have normal rates — the excess occurs only in canyon cells.
    This favors the hypothesis that canyon geometry specifically drives the
    excess in Puget Sound, not a general regional effect.

    However, the mechanism still doesn't generalize: other WC canyon cells
    don't show the same pattern. CTH as a universal prediction is falsified,
    but a Puget-specific version survives.""")


# ============================================================
# SAVE
# ============================================================
results = {
    "test": "Puget confound test",
    "question": "Is Puget UAP excess driven by canyons or regional confound?",
    "group_rates": group_stats,
    "confound_test": {
        "puget_s0_n": int(puget_s0.sum()),
        "other_s0_n": int(other_s0.sum()),
        "puget_s0_aggregate_rate": round(puget_s0_agg, 4),
        "other_s0_aggregate_rate": round(other_s0_agg, 4),
        "mann_whitney_p": round(float(p_mw), 4) if confound_verdict != "INSUFFICIENT_DATA" else None,
        "verdict": confound_verdict,
    },
    "interaction_2x2": {
        "alpha": round(float(b[0]), 4),
        "beta_Canyon": round(float(b[1]), 4),
        "beta_Puget": round(float(b[2]), 4),
        "beta_CanyonxPuget": round(float(b[3]), 4),
        "bootstrap_CIs": {
            names[i]: {
                "point": round(float(b[i]), 4),
                "ci_lo": round(float(np.percentile(boot_coefs[:, i], 2.5)), 4),
                "ci_hi": round(float(np.percentile(boot_coefs[:, i], 97.5)), 4),
            }
            for i in range(4)
        },
    },
    "rate_table": {
        "puget_s0": round(puget_s0_agg, 4),
        "puget_s1": round(puget_s1_agg, 4),
        "other_s0": round(other_s0_agg, 4),
        "other_s1": round(other_s1_agg, 4),
    },
}

out_file = os.path.join(BASE_DIR, "phase_ev2", "phase_e_puget_confound.json")
with open(out_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_file}")

import shutil
repo_out = os.path.join(BASE_DIR, "uap-canyon-analysis", "results", "phase_ev2")
shutil.copy2(out_file, repo_out)
print(f"Copied to repo")
