#!/usr/bin/env python3
"""
generate_scorecard.py — Robustness scorecard.

Ive school: white space, alignment, no chart junk.
One question per row. YES / NO / NULL. Done.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as FancyBboxPatch
import os

# ── Design ───────────────────────────────────────────────────────────
C_DARK = "#2C3E50"
C_MED = "#95A5A6"
C_LIGHT = "#BDC3C7"
C_BG = "#F8F9FA"
C_GREEN = "#27AE60"
C_RED = "#E74C3C"
C_GRAY = "#95A5A6"
C_BLUE = "#2980B9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
})

# ── Data ─────────────────────────────────────────────────────────────
rows = [
    ("Stable across time?",           "YES", C_GREEN,
     "4 temporal splits  |  rho = 0.32-0.36  |  all p < 0.003"),

    ("Replicated in new data?",       "YES", C_GREEN,
     "Post-2014 holdout  |  5,946 new reports  |  rho = 0.35, p = 0.0001"),

    ("Threshold cherry-picked?",      "NO",  C_GREEN,
     "All thresholds 20-100 m/km tested  |  rho = 0.37-0.41  |  all significant"),

    ("Blind prediction works?",       "YES", C_GREEN,
     "Model built from geometry alone  |  Puget rho = 0.60  |  SoCal rho = 0.47"),

    ("Explained by population?",      "NO",  C_GREEN,
     "Puget non-canyon cells: 0.74x  |  Canyon cells: 5.0x  |  Same metro area"),

    ("Explained by military?",        "NO",  C_GREEN,
     "Monterey canyon cells 127-192 km from nearest OPAREA  |  Still logR = 0.75"),

    ("Explained by other confounds?", "NO",  C_GREEN,
     "Ports, depth, upwelling, magnetic, shore type  |  All nested F-tests: S dominant"),

    ("Effect uniform across coast?",  "NO",  C_RED,
     "Puget 6.8x  |  San Diego 9.8x  |  Monterey 2.75-4.80x  |  Rest of WC: 1.4x"),

    ("Detected on East Coast?",       "NULL", C_GRAY,
     "rho = 0.06  |  Canyons 100-400 km offshore  |  Testability limitation"),
]

# ── Layout ───────────────────────────────────────────────────────────
n = len(rows)
row_h = 1.0
fig_w = 13
fig_h = n * row_h + 3.2

fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
ax.set_xlim(0, fig_w)
ax.set_ylim(-0.8, n * row_h + 2.2)
ax.axis("off")

# ── Header ───────────────────────────────────────────────────────────
ax.text(fig_w / 2, n * row_h + 1.8,
        "Does the canyon-UAP association survive hostile testing?",
        ha="center", va="center", fontsize=17, fontweight="bold", color=C_DARK)

# Primary result badge
badge_y = n * row_h + 0.95
badge = plt.Rectangle((fig_w / 2 - 3.8, badge_y - 0.25), 7.6, 0.5,
                       facecolor="#EBF5FB", edgecolor=C_BLUE,
                       linewidth=0.8, zorder=2, clip_on=False)
ax.add_patch(badge)
ax.text(fig_w / 2, badge_y,
        "rho = 0.37    95% CI [0.19, 0.53]    p = 0.0001    n = 102",
        ha="center", va="center", fontsize=11, fontweight="bold",
        color=C_BLUE, zorder=3)

# Divider
div_y = n * row_h + 0.45
ax.plot([0.8, fig_w - 0.8], [div_y, div_y], color=C_LIGHT, linewidth=0.5)

# Column positions
x_q = 0.8       # question
x_badge = 8.6   # YES/NO badge center
x_ev = 9.5      # evidence

# Column headers
head_y = n * row_h + 0.15
ax.text(x_q, head_y, "TEST", fontsize=7.5, fontweight="bold", color=C_MED)
ax.text(x_badge, head_y, "", fontsize=7.5, fontweight="bold", color=C_MED, ha="center")
ax.text(x_ev, head_y, "EVIDENCE", fontsize=7.5, fontweight="bold", color=C_MED)

# ── Rows ─────────────────────────────────────────────────────────────
for i, (question, answer, color, evidence) in enumerate(rows):
    y = (n - 1 - i) * row_h + 0.3

    # Alternating background
    if i % 2 == 0:
        bg = plt.Rectangle((0.4, y - 0.38), fig_w - 0.8, row_h - 0.05,
                            facecolor=C_BG, edgecolor="none", zorder=0)
        ax.add_patch(bg)

    # Question
    ax.text(x_q, y, question, fontsize=11, va="center", ha="left",
            color=C_DARK, fontweight="medium")

    # Answer badge
    badge_w = 0.65 if answer != "NULL" else 0.75
    badge_h = 0.32
    badge_rect = plt.Rectangle((x_badge - badge_w / 2, y - badge_h / 2),
                                badge_w, badge_h,
                                facecolor=color, edgecolor="none",
                                alpha=0.12, zorder=1)
    ax.add_patch(badge_rect)
    ax.text(x_badge, y, answer, fontsize=9.5, fontweight="bold",
            va="center", ha="center", color=color, zorder=2)

    # Evidence
    ax.text(x_ev, y, evidence, fontsize=7.5, va="center", ha="left",
            color=C_MED, linespacing=1.3)

# ── Footer ───────────────────────────────────────────────────────────
foot_y = -0.35
ax.plot([0.8, fig_w - 0.8], [foot_y + 0.3, foot_y + 0.3],
        color=C_LIGHT, linewidth=0.5)
ax.text(fig_w / 2, foot_y,
        "7 passed    1 honest limitation (regional concentration)    "
        "1 null control (East Coast, as expected)",
        ha="center", va="center", fontsize=9, color=C_DARK, fontweight="bold")

# ── Save ─────────────────────────────────────────────────────────────
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path = os.path.join(REPO_DIR, "figures", "figure_scorecard.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.2)
plt.close()
print(f"Saved: {out_path}")
