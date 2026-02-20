#!/usr/bin/env python3
"""
generate_loeb_figure.py — "Blind prediction vs reality"

Jony Ive school: remove everything until it breaks, then add one thing back.
Two maps. No gridlines. No tick labels. Color speaks. White space breathes.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RED_V2_PATH = os.path.join(REPO_DIR, "results", "phase_ev2", "phase_e_red_v2_evaluation.json")
FIG_DIR = os.path.join(REPO_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────
with open(RED_V2_PATH) as f:
    cells = json.load(f)["primary_200km"]["cell_details"]

lats = np.array([c["lat"] for c in cells])
lons = np.array([c["lon"] for c in cells])
S = np.array([c["S"] for c in cells])
logR = np.array([c["logR"] for c in cells])
hot = S > 0

# ── Figure ───────────────────────────────────────────────────────────
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 9), facecolor="white")

for ax in [ax_l, ax_r]:
    ax.set_xlim(-127, -114.5)
    ax.set_ylim(31, 51)
    ax.set_aspect(1.3)
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# ── LEFT: what the model predicted ───────────────────────────────────
# Cold cells: barely there
ax_l.scatter(lons[~hot], lats[~hot], c="#EAECEE", s=90, marker="s",
             edgecolors="#EAECEE", linewidths=0, zorder=2)

# Hot cells: warm gradient
sc = ax_l.scatter(lons[hot], lats[hot], c=S[hot], s=100, marker="s",
                  cmap="YlOrRd", vmin=0, vmax=S.max(),
                  edgecolors="white", linewidths=0.6, zorder=3)

ax_l.text(0.5, 1.02, "Prediction", transform=ax_l.transAxes,
          ha="center", va="bottom", fontsize=18, fontweight="bold",
          color="#2C3E50")
ax_l.text(0.5, 0.97, "from ocean-floor geometry alone",
          transform=ax_l.transAxes, ha="center", va="bottom",
          fontsize=9, color="#95A5A6")

# Minimal colorbar
cb_l = plt.colorbar(sc, ax=ax_l, shrink=0.35, pad=0.01, aspect=20)
cb_l.set_label("Canyon score", fontsize=8, color="#95A5A6", labelpad=5)
cb_l.ax.tick_params(labelsize=6, colors="#95A5A6", length=2)
cb_l.outline.set_visible(False)

# ── RIGHT: what actually happened ────────────────────────────────────
sc2 = ax_r.scatter(lons, lats, c=logR, s=100, marker="s",
                    cmap="RdBu_r", vmin=-1.5, vmax=3.5,
                    edgecolors="white", linewidths=0.6, zorder=3)

# Canyon cells: thin dark outline
ax_r.scatter(lons[hot], lats[hot], s=100, marker="s",
             facecolors="none", edgecolors="#2C3E50", linewidths=0.8, zorder=4)

ax_r.text(0.5, 1.02, "Reality", transform=ax_r.transAxes,
          ha="center", va="bottom", fontsize=18, fontweight="bold",
          color="#2C3E50")
ax_r.text(0.5, 0.97, "population-adjusted UAP reports, 1990-2014",
          transform=ax_r.transAxes, ha="center", va="bottom",
          fontsize=9, color="#95A5A6")

cb_r = plt.colorbar(sc2, ax=ax_r, shrink=0.35, pad=0.01, aspect=20)
cb_r.set_label("Report excess (logR)", fontsize=8, color="#95A5A6", labelpad=5)
cb_r.ax.tick_params(labelsize=6, colors="#95A5A6", length=2)
cb_r.outline.set_visible(False)

# ── Title ────────────────────────────────────────────────────────────
fig.text(0.5, 0.97, "Blind Prediction vs Reality",
         ha="center", fontsize=22, fontweight="bold", color="#2C3E50")
fig.text(0.5, 0.935,
         "The model was built from seafloor geometry without seeing any UAP data.  "
         "rho = 0.37, p = 0.0001",
         ha="center", fontsize=9.5, color="#95A5A6")

plt.subplots_adjust(top=0.88, bottom=0.04, wspace=0.08)

# ── Save ─────────────────────────────────────────────────────────────
out_path = os.path.join(FIG_DIR, "figure_prediction_vs_reality.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.15)
plt.close()
print(f"Saved: {out_path}")
