#!/usr/bin/env python3
"""
Phase E: Puget Interaction — Sanity Checks
============================================
Two checks on the "unnaturally sharp" β_Puget=-21.6, β_S×Puget=+11.05:

1. CENTERING: Re-run interaction model with S centered at grand mean.
   Makes intercepts interpretable (logR at mean S, not at S=0).
   The interaction slope won't change, but we'll see if the "big numbers"
   are just an artefact of extrapolation to S=0.

2. LEVERAGE: Cook's distance for each cell. If one cell (Vancouver 49.2N,
   R_i=72) drives the interaction, the result is fragile.

3. LEAVE-ONE-OUT (LOO): Drop each Puget cell, re-fit, check stability
   of β_S×Puget.

4. RANGE CHECK: How much of the β_S×Puget is just "narrow S range in Puget"?
   Compare marginal predictions at actual S values, not extrapolated.
"""

import json, os
import numpy as np
from numpy.linalg import lstsq
from scipy.stats import f as f_dist

BASE_DIR = "/Users/antoniwedzikowski/Desktop/UAP research"

# ============================================================
# LOAD
# ============================================================
with open(os.path.join(BASE_DIR, "phase_ev2", "phase_e_red_v2_evaluation.json")) as f:
    data = json.load(f)

cells = data["primary_200km"]["cell_details"]
S_arr    = np.array([c['S'] for c in cells])
logR_arr = np.array([c['logR'] for c in cells])
lat_arr  = np.array([c['lat'] for c in cells])
lon_arr  = np.array([c['lon'] for c in cells])
R_arr    = np.array([c['R_i'] for c in cells])
n = len(cells)

puget_mask = (S_arr > 1.8) & (lat_arr > 46) & (lat_arr < 50)
P_arr = puget_mask.astype(float)
n_puget = int(P_arr.sum())

print(f"n = {n}, Puget = {n_puget}")
print(f"S range (Puget):  [{S_arr[puget_mask].min():.3f}, {S_arr[puget_mask].max():.3f}]")
print(f"S range (Other):  [{S_arr[~puget_mask].min():.3f}, {S_arr[~puget_mask].max():.3f}]")
print(f"S range (All):    [{S_arr.min():.3f}, {S_arr.max():.3f}]")


# ============================================================
# SANITY CHECK 1: CENTERING
# ============================================================
print(f"\n{'='*60}")
print("SANITY CHECK 1: CENTERED MODEL")
print(f"{'='*60}")

S_mean = S_arr.mean()
S_c = S_arr - S_mean  # centered S
SP_c = S_c * P_arr    # centered interaction

print(f"  Grand mean S = {S_mean:.4f}")

# Uncentered model (for comparison)
X_unc = np.column_stack([np.ones(n), S_arr, P_arr, S_arr * P_arr])
b_unc, _, _, _ = lstsq(X_unc, logR_arr, rcond=None)

# Centered model
X_cen = np.column_stack([np.ones(n), S_c, P_arr, SP_c])
b_cen, _, _, _ = lstsq(X_cen, logR_arr, rcond=None)

param_names_unc = ['α', 'β_S', 'β_Puget', 'β_S×Puget']
param_names_cen = ['α_c', 'β_S', 'β_Puget_c', 'β_S×Puget']

print(f"\n  {'':16s} {'Uncentered':>12s}  {'Centered':>12s}")
print(f"  {'-'*44}")
for i in range(4):
    print(f"  {param_names_unc[i]:16s} {b_unc[i]:+12.4f}  {b_cen[i]:+12.4f}")

print(f"\n  Key insight:")
print(f"    Uncentered α_Puget = {b_unc[2]:+.2f} (at S=0 — DOES NOT EXIST in data)")
print(f"    Centered α_Puget_c = {b_cen[2]:+.2f} (at S=mean={S_mean:.2f} — interpretable)")
print(f"    β_S×Puget is identical: {b_unc[3]:.4f} vs {b_cen[3]:.4f}")

# What logR does the model predict for a typical Puget vs non-Puget cell?
S_puget_mean = S_arr[puget_mask].mean()
pred_puget = b_unc[0] + b_unc[1]*S_puget_mean + b_unc[2]*1 + b_unc[3]*S_puget_mean
pred_other_at_same_S = b_unc[0] + b_unc[1]*S_puget_mean
print(f"\n  At mean Puget S ({S_puget_mean:.3f}):")
print(f"    Predicted logR (Puget):     {pred_puget:.3f}  → R = {np.exp(pred_puget):.2f}")
print(f"    Predicted logR (non-Puget): {pred_other_at_same_S:.3f}  → R = {np.exp(pred_other_at_same_S):.2f}")
print(f"    Difference: {pred_puget - pred_other_at_same_S:.3f} "
      f"(Puget has {np.exp(pred_puget - pred_other_at_same_S):.1f}x more)")


# ============================================================
# SANITY CHECK 2: COOK'S DISTANCE
# ============================================================
print(f"\n{'='*60}")
print("SANITY CHECK 2: COOK'S DISTANCE")
print(f"{'='*60}")

X = X_unc  # use uncentered for Cook's (results are the same)
b_full = b_unc
yhat = X @ b_full
residuals = logR_arr - yhat
p_params = X.shape[1]
mse = np.sum(residuals**2) / (n - p_params)

# Hat matrix H = X (X'X)^{-1} X'
XtX_inv = np.linalg.inv(X.T @ X)
H = X @ XtX_inv @ X.T
h_ii = np.diag(H)  # leverage

# Cook's distance
cooks_d = (residuals**2 / (p_params * mse)) * (h_ii / (1 - h_ii)**2)

# Sort by Cook's distance
idx_sorted = np.argsort(-cooks_d)

print(f"\n  Top 15 by Cook's distance:")
print(f"  {'#':>3} {'lat':>6} {'lon':>7} {'S':>6} {'R_i':>7} {'logR':>7} {'h_ii':>6} {'Cook_d':>8} {'Puget':>6}")
print(f"  {'-'*65}")
for rank, i in enumerate(idx_sorted[:15]):
    tag = "***" if puget_mask[i] else ""
    print(f"  {rank+1:3d} {lat_arr[i]:6.1f} {lon_arr[i]:7.1f} {S_arr[i]:6.3f} "
          f"{R_arr[i]:7.2f} {logR_arr[i]:7.3f} {h_ii[i]:6.3f} {cooks_d[i]:8.4f} {tag}")

# Common threshold: 4/n
threshold = 4/n
n_influential = np.sum(cooks_d > threshold)
n_influential_puget = np.sum(cooks_d[puget_mask] > threshold)
print(f"\n  Cook's D > 4/n ({threshold:.4f}): {n_influential} cells")
print(f"    of which Puget: {n_influential_puget}")

# Max Cook's D
max_cook_idx = np.argmax(cooks_d)
print(f"\n  Most influential cell:")
print(f"    ({lat_arr[max_cook_idx]:.1f}, {lon_arr[max_cook_idx]:.1f}) S={S_arr[max_cook_idx]:.3f} "
      f"R={R_arr[max_cook_idx]:.2f} Cook's D={cooks_d[max_cook_idx]:.4f} "
      f"Puget={'YES' if puget_mask[max_cook_idx] else 'NO'}")


# ============================================================
# SANITY CHECK 3: LEAVE-ONE-OUT ON PUGET CELLS
# ============================================================
print(f"\n{'='*60}")
print("SANITY CHECK 3: LEAVE-ONE-OUT (Puget cells)")
print(f"{'='*60}")

puget_indices = np.where(puget_mask)[0]
loo_betas = []

print(f"\n  Drop cell → β_S×Puget")
print(f"  {'lat':>6} {'lon':>7} {'S':>6} {'R_i':>7} {'β_SxP':>8} {'Δ from full':>12}")
print(f"  {'-'*55}")

for drop_idx in puget_indices:
    keep = np.ones(n, dtype=bool)
    keep[drop_idx] = False
    X_loo = X_unc[keep]
    y_loo = logR_arr[keep]
    b_loo, _, _, _ = lstsq(X_loo, y_loo, rcond=None)
    loo_betas.append(b_loo[3])
    delta = b_loo[3] - b_unc[3]
    print(f"  {lat_arr[drop_idx]:6.1f} {lon_arr[drop_idx]:7.1f} {S_arr[drop_idx]:6.3f} "
          f"{R_arr[drop_idx]:7.2f} {b_loo[3]:+8.3f} {delta:+12.3f}")

loo_betas = np.array(loo_betas)
print(f"\n  Full model β_S×Puget:  {b_unc[3]:+.4f}")
print(f"  LOO range:             [{loo_betas.min():+.4f}, {loo_betas.max():+.4f}]")
print(f"  LOO mean:              {loo_betas.mean():+.4f}")
print(f"  LOO std:               {loo_betas.std():.4f}")

# Sign stability
all_positive = np.all(loo_betas > 0)
print(f"  All LOO β_S×Puget > 0: {all_positive}")

# Also check: does interaction remain significant in each LOO?
print(f"\n  LOO F-tests for interaction:")
for i, drop_idx in enumerate(puget_indices):
    keep = np.ones(n, dtype=bool)
    keep[drop_idx] = False
    n_loo = keep.sum()

    # Model 2 (no interaction)
    X2_loo = np.column_stack([np.ones(n_loo), S_arr[keep], P_arr[keep]])
    b2_loo, _, _, _ = lstsq(X2_loo, logR_arr[keep], rcond=None)
    ss_res2 = np.sum((logR_arr[keep] - X2_loo @ b2_loo)**2)

    # Model 3 (with interaction)
    X3_loo = X_unc[keep]
    b3_loo, _, _, _ = lstsq(X3_loo, logR_arr[keep], rcond=None)
    ss_res3 = np.sum((logR_arr[keep] - X3_loo @ b3_loo)**2)

    df3 = n_loo - 4
    F_loo = ((ss_res2 - ss_res3) / 1) / (ss_res3 / df3)
    p_loo = 1 - f_dist.cdf(F_loo, 1, df3)
    sig = "***" if p_loo < 0.001 else "**" if p_loo < 0.01 else "*" if p_loo < 0.05 else "NS"
    print(f"    Drop ({lat_arr[drop_idx]:.1f}, {lon_arr[drop_idx]:.1f}) S={S_arr[drop_idx]:.3f}: "
          f"F={F_loo:.2f} p={p_loo:.4f} {sig}")


# ============================================================
# SANITY CHECK 4: MARGINAL PREDICTIONS AT ACTUAL S VALUES
# ============================================================
print(f"\n{'='*60}")
print("SANITY CHECK 4: MARGINAL PREDICTIONS")
print(f"{'='*60}")

print(f"\n  Actual Puget cells vs model predictions:")
print(f"  {'lat':>6} {'lon':>7} {'S':>6} {'logR_obs':>9} {'logR_pred':>10} {'residual':>9}")
print(f"  {'-'*55}")
for i in puget_indices:
    pred = b_unc[0] + b_unc[1]*S_arr[i] + b_unc[2]*1 + b_unc[3]*S_arr[i]
    resid = logR_arr[i] - pred
    print(f"  {lat_arr[i]:6.1f} {lon_arr[i]:7.1f} {S_arr[i]:6.3f} "
          f"{logR_arr[i]:+9.3f} {pred:+10.3f} {resid:+9.3f}")

# What's the actual observed Puget pattern?
print(f"\n  Observed Puget pattern:")
print(f"    Spearman(S, logR) within Puget: ", end="")
from scipy.stats import spearmanr
if n_puget >= 5:
    rho_p, p_p = spearmanr(S_arr[puget_mask], logR_arr[puget_mask])
    print(f"rho = {rho_p:.3f}, p = {p_p:.4f}")
else:
    print("too few cells")

print(f"    Spearman(S, logR) outside Puget (S>0 only): ", end="")
other_hot = (~puget_mask) & (S_arr > 0)
if other_hot.sum() >= 5:
    rho_o, p_o = spearmanr(S_arr[other_hot], logR_arr[other_hot])
    print(f"rho = {rho_o:.3f}, p = {p_o:.4f} (n={other_hot.sum()})")
else:
    print(f"too few cells (n={other_hot.sum()})")


# ============================================================
# SAVE
# ============================================================
results = {
    "test": "Puget interaction sanity checks",
    "centering": {
        "grand_mean_S": round(float(S_mean), 4),
        "centered_beta_Puget": round(float(b_cen[2]), 4),
        "uncentered_beta_Puget": round(float(b_unc[2]), 4),
        "note": "Centered β_Puget is logR difference at mean S; uncentered is at S=0 (nonexistent)",
        "pred_puget_at_mean_puget_S": round(float(pred_puget), 4),
        "pred_other_at_mean_puget_S": round(float(pred_other_at_same_S), 4),
    },
    "cooks_distance": {
        "threshold_4n": round(threshold, 4),
        "n_influential": int(n_influential),
        "n_influential_puget": int(n_influential_puget),
        "max_cook_cell": {
            "lat": float(lat_arr[max_cook_idx]),
            "lon": float(lon_arr[max_cook_idx]),
            "S": float(S_arr[max_cook_idx]),
            "R_i": float(R_arr[max_cook_idx]),
            "cooks_d": round(float(cooks_d[max_cook_idx]), 4),
        },
    },
    "leave_one_out": {
        "full_beta_SxPuget": round(float(b_unc[3]), 4),
        "loo_range": [round(float(loo_betas.min()), 4), round(float(loo_betas.max()), 4)],
        "loo_mean": round(float(loo_betas.mean()), 4),
        "all_positive": bool(all_positive),
    },
    "within_puget_spearman": {
        "rho": round(float(rho_p), 4) if n_puget >= 5 else None,
        "p": round(float(p_p), 4) if n_puget >= 5 else None,
        "n": n_puget,
    },
}

out_file = os.path.join(BASE_DIR, "phase_ev2", "phase_e_puget_sanity.json")
with open(out_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_file}")

import shutil
repo_out = os.path.join(BASE_DIR, "uap-canyon-analysis", "results", "phase_ev2")
shutil.copy2(out_file, repo_out)
print(f"Copied to repo")
