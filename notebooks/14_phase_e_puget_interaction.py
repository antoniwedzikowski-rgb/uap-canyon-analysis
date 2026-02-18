#!/usr/bin/env python3
"""
Phase E: Puget Interaction Test
================================
OLS: logR ~ S + Puget + S×Puget

Interpretation:
  - β_S significant, β_interaction NS → S works everywhere, Puget just raises intercept
  - β_interaction significant → geometry works differently/sharper in Puget (nearshore mechanism)

Uses haversine-corrected cell_details from phase_e_red_v2_evaluation.json (primary 200km).
"""

import json, os
import numpy as np
from numpy.linalg import lstsq

BASE_DIR = "/Users/antoniwedzikowski/Desktop/UAP research"

# ============================================================
# LOAD cell details from E-RED v2 primary
# ============================================================
with open(os.path.join(BASE_DIR, "phase_ev2", "phase_e_red_v2_evaluation.json")) as f:
    data = json.load(f)

cells = data["primary_200km"]["cell_details"]
print(f"Loaded {len(cells)} testable cells (primary 200km, haversine-corrected)")

S_arr    = np.array([c['S'] for c in cells])
logR_arr = np.array([c['logR'] for c in cells])
lat_arr  = np.array([c['lat'] for c in cells])
n = len(cells)

# ============================================================
# Define Puget indicator
# ============================================================
# Same definition as E-RED: S > 1.8 AND 46 < lat < 50
puget_mask = (S_arr > 1.8) & (lat_arr > 46) & (lat_arr < 50)
P_arr = puget_mask.astype(float)
n_puget = int(P_arr.sum())
n_other = n - n_puget
print(f"Puget cells: {n_puget}, Other: {n_other}")
print(f"Puget cells: {[(c['lat'], c['lon'], c['S']) for c, p in zip(cells, puget_mask) if p]}")

# ============================================================
# MODEL 1: logR ~ S  (baseline, no Puget)
# ============================================================
X1 = np.column_stack([np.ones(n), S_arr])
b1, res1, _, _ = lstsq(X1, logR_arr, rcond=None)
yhat1 = X1 @ b1
ss_res1 = np.sum((logR_arr - yhat1)**2)
ss_tot = np.sum((logR_arr - logR_arr.mean())**2)
r2_1 = 1 - ss_res1 / ss_tot

print(f"\n{'='*60}")
print("MODEL 1: logR ~ S")
print(f"{'='*60}")
print(f"  α (intercept) = {b1[0]:.4f}")
print(f"  β_S            = {b1[1]:.4f}")
print(f"  R²             = {r2_1:.4f}")

# ============================================================
# MODEL 2: logR ~ S + Puget  (additive)
# ============================================================
X2 = np.column_stack([np.ones(n), S_arr, P_arr])
b2, res2, _, _ = lstsq(X2, logR_arr, rcond=None)
yhat2 = X2 @ b2
ss_res2 = np.sum((logR_arr - yhat2)**2)
r2_2 = 1 - ss_res2 / ss_tot

print(f"\n{'='*60}")
print("MODEL 2: logR ~ S + Puget")
print(f"{'='*60}")
print(f"  α (intercept) = {b2[0]:.4f}")
print(f"  β_S            = {b2[1]:.4f}")
print(f"  β_Puget        = {b2[2]:.4f}")
print(f"  R²             = {r2_2:.4f}")

# ============================================================
# MODEL 3: logR ~ S + Puget + S×Puget  (interaction)
# ============================================================
SP_arr = S_arr * P_arr  # interaction term
X3 = np.column_stack([np.ones(n), S_arr, P_arr, SP_arr])
b3, res3, _, _ = lstsq(X3, logR_arr, rcond=None)
yhat3 = X3 @ b3
ss_res3 = np.sum((logR_arr - yhat3)**2)
r2_3 = 1 - ss_res3 / ss_tot

print(f"\n{'='*60}")
print("MODEL 3: logR ~ S + Puget + S×Puget")
print(f"{'='*60}")
print(f"  α (intercept)  = {b3[0]:.4f}")
print(f"  β_S            = {b3[1]:.4f}")
print(f"  β_Puget        = {b3[2]:.4f}")
print(f"  β_S×Puget      = {b3[3]:.4f}")
print(f"  R²             = {r2_3:.4f}")

# ============================================================
# BOOTSTRAP CIs for Model 3
# ============================================================
N_BOOT = 5000
rng = np.random.RandomState(42)
boot_coefs = []
for _ in range(N_BOOT):
    idx = rng.choice(n, n, replace=True)
    Xb = X3[idx]
    yb = logR_arr[idx]
    try:
        bb, _, _, _ = lstsq(Xb, yb, rcond=None)
        boot_coefs.append(bb)
    except:
        pass

boot_coefs = np.array(boot_coefs)
param_names = ['α (intercept)', 'β_S', 'β_Puget', 'β_S×Puget']

print(f"\n{'='*60}")
print("MODEL 3: Bootstrap 95% CIs (n=5000)")
print(f"{'='*60}")
for i, name in enumerate(param_names):
    lo = np.percentile(boot_coefs[:, i], 2.5)
    hi = np.percentile(boot_coefs[:, i], 97.5)
    med = np.percentile(boot_coefs[:, i], 50)
    # Fraction of bootstraps where coefficient has same sign as point estimate
    same_sign_frac = np.mean(np.sign(boot_coefs[:, i]) == np.sign(b3[i]))
    print(f"  {name:16s} = {b3[i]:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
          f"(p_boot ≈ {2*min(same_sign_frac, 1-same_sign_frac):.4f})")

# ============================================================
# F-test: Model 2 vs Model 3 (does interaction improve fit?)
# ============================================================
df_2 = n - 3  # Model 2 has 3 params
df_3 = n - 4  # Model 3 has 4 params
F_stat = ((ss_res2 - ss_res3) / 1) / (ss_res3 / df_3)

# Approximate p-value from F distribution
from scipy.stats import f as f_dist
p_F = 1 - f_dist.cdf(F_stat, 1, df_3)

print(f"\n{'='*60}")
print("F-TEST: Model 2 vs Model 3 (interaction term)")
print(f"{'='*60}")
print(f"  F({1}, {df_3}) = {F_stat:.3f}")
print(f"  p = {p_F:.4f}")
if p_F < 0.05:
    print(f"  → Interaction IS significant (p < 0.05)")
    print(f"  → Geometry works differently in Puget")
else:
    print(f"  → Interaction is NOT significant (p >= 0.05)")
    print(f"  → S works similarly everywhere; Puget just raises intercept")

# ============================================================
# INTERPRETATION
# ============================================================
print(f"\n{'='*60}")
print("INTERPRETATION")
print(f"{'='*60}")

print(f"\n  In Model 3:")
print(f"    Non-Puget cells: E[logR] = {b3[0]:.3f} + {b3[1]:.3f}·S")
print(f"    Puget cells:     E[logR] = {b3[0]+b3[2]:.3f} + {b3[1]+b3[3]:.3f}·S")
print(f"")
print(f"    β_S = {b3[1]:.3f}: effect of S outside Puget")
print(f"    β_S + β_S×Puget = {b3[1]+b3[3]:.3f}: effect of S inside Puget")
print(f"    β_Puget = {b3[2]:.3f}: Puget intercept shift (more reports generally)")
print(f"    β_S×Puget = {b3[3]:.3f}: extra S-slope in Puget")

# Effective slopes
slope_other = b3[1]
slope_puget = b3[1] + b3[3]
print(f"\n  Effective slopes:")
print(f"    Outside Puget: 1 unit S → exp({slope_other:.3f}) = {np.exp(slope_other):.2f}x rate")
print(f"    Inside Puget:  1 unit S → exp({slope_puget:.3f}) = {np.exp(slope_puget):.2f}x rate")

# ============================================================
# SAVE
# ============================================================
results = {
    "test": "Puget interaction",
    "n_cells": n,
    "n_puget": n_puget,
    "n_other": n_other,
    "puget_definition": "S > 1.8 AND 46 < lat < 50",
    "model_1_baseline": {
        "formula": "logR ~ S",
        "alpha": round(float(b1[0]), 4),
        "beta_S": round(float(b1[1]), 4),
        "R2": round(r2_1, 4),
    },
    "model_2_additive": {
        "formula": "logR ~ S + Puget",
        "alpha": round(float(b2[0]), 4),
        "beta_S": round(float(b2[1]), 4),
        "beta_Puget": round(float(b2[2]), 4),
        "R2": round(r2_2, 4),
    },
    "model_3_interaction": {
        "formula": "logR ~ S + Puget + S×Puget",
        "alpha": round(float(b3[0]), 4),
        "beta_S": round(float(b3[1]), 4),
        "beta_Puget": round(float(b3[2]), 4),
        "beta_SxPuget": round(float(b3[3]), 4),
        "R2": round(r2_3, 4),
        "bootstrap_CIs": {
            name: {
                "point": round(float(b3[i]), 4),
                "ci_lo": round(float(np.percentile(boot_coefs[:, i], 2.5)), 4),
                "ci_hi": round(float(np.percentile(boot_coefs[:, i], 97.5)), 4),
            }
            for i, name in enumerate(param_names)
        },
    },
    "F_test_interaction": {
        "F_statistic": round(F_stat, 4),
        "df1": 1,
        "df2": df_3,
        "p_value": round(p_F, 4),
        "significant": bool(p_F < 0.05),
    },
    "effective_slopes": {
        "outside_puget": round(slope_other, 4),
        "inside_puget": round(slope_puget, 4),
        "exp_outside": round(float(np.exp(slope_other)), 4),
        "exp_inside": round(float(np.exp(slope_puget)), 4),
    },
}

out_file = os.path.join(BASE_DIR, "phase_ev2", "phase_e_puget_interaction.json")
with open(out_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_file}")

import shutil
repo_out = os.path.join(BASE_DIR, "uap-canyon-analysis", "results", "phase_ev2")
os.makedirs(repo_out, exist_ok=True)
shutil.copy2(out_file, repo_out)
print(f"Copied to repo")
