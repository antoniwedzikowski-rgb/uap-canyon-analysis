#!/usr/bin/env python3
"""
generate_hero_figure.py — Media-quality hero map: bathymetry + canyon cells + UAP density.

Produces a wide (12x8 in) West Coast map with:
  - Color bathymetry (ETOPO1 blue gradient, land in tan/warm gray)
  - Red/orange markers for submarine canyon cells (gradient > 60 m/km, 0 to -500 m)
  - UAP report density as translucent scatter overlay
  - Named canyon labels (Puget Sound, Monterey, La Jolla/Scripps, Mugu)
  - Professional cartographic styling (Nature / Science quality)

Saves: figures/figure_hero_bathymetry.png at 300 DPI.
"""

import os
import ssl
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker

# Cartopy
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.environ.get(
    "UAP_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
DATA_DIR = os.path.join(BASE_DIR, "data")
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

ETOPO_PATH = os.path.join(DATA_DIR, "etopo_subset.nc")
NUFORC_PATH = os.path.join(DATA_DIR, "nuforc_reports.csv")

# ── Design constants ─────────────────────────────────────────────────────
ACCENT = "#D35400"          # warm orange-red for canyon markers
ACCENT_DARK = "#C0392B"     # darker red for labels
UAP_COLOR = "#FDEDEC"       # pale rose for UAP scatter
LAND_COLOR = "#F5F0E8"      # warm parchment tan
LAND_EDGE = "#B0A898"       # muted warm gray for coastline
GRAY_DARK = "#3C3C3C"
GRAY_MED = "#808080"

# West Coast extent
LON_MIN, LON_MAX = -130, -115
LAT_MIN, LAT_MAX = 30, 52

# Canyon detection
GRADIENT_THRESHOLD = 60.0   # m/km
DEPTH_MIN = -500            # shallowest bound (most negative)
DEPTH_MAX = 0               # sea level

# ── Font setup ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "mathtext.default": "regular",
})

# =====================================================================
# 1. LOAD BATHYMETRY
# =====================================================================
print("Loading ETOPO bathymetry ...")
ds = xr.open_dataset(ETOPO_PATH)
lats_full = ds["lat"].values
lons_full = ds["lon"].values
elev_full = ds["z"].values
ds.close()

# Subset to West Coast with margin
margin = 1.0
lat_mask = (lats_full >= LAT_MIN - margin) & (lats_full <= LAT_MAX + margin)
lon_mask = (lons_full >= LON_MIN - margin) & (lons_full <= LON_MAX + margin)
lats = lats_full[lat_mask]
lons = lons_full[lon_mask]
elev = elev_full[np.ix_(lat_mask, lon_mask)]
lon_grid, lat_grid = np.meshgrid(lons, lats)

print(f"  Bathymetry grid: {elev.shape[0]} x {elev.shape[1]}")

# =====================================================================
# 2. COMPUTE CANYON CELLS (gradient > 60 m/km on shelf 0 to -500 m)
# =====================================================================
print("Computing bathymetric gradient ...")
dy = np.gradient(elev, axis=0)
dx = np.gradient(elev, axis=1)
dlat_km = 111.0 * (lats[1] - lats[0]) if len(lats) > 1 else 1.85
dlon_km = (
    111.0 * np.cos(np.radians(np.mean(lats))) * (lons[1] - lons[0])
    if len(lons) > 1
    else 1.4
)
grad_mag = np.sqrt((dy / dlat_km) ** 2 + (dx / dlon_km) ** 2)

shelf_mask = (elev <= DEPTH_MAX) & (elev >= DEPTH_MIN)
canyon_cells = (grad_mag >= GRADIENT_THRESHOLD) & shelf_mask

canyon_lats = lat_grid[canyon_cells]
canyon_lons = lon_grid[canyon_cells]
print(f"  Canyon cells (>{GRADIENT_THRESHOLD} m/km, {DEPTH_MIN} to {DEPTH_MAX} m): {len(canyon_lats)}")

# =====================================================================
# 3. LOAD NUFORC REPORTS
# =====================================================================
print("Loading NUFORC reports ...")
col_names = [
    "datetime", "city", "state", "country", "shape",
    "duration_sec", "duration_text", "description",
    "date_posted", "latitude", "longitude",
]
df = pd.read_csv(NUFORC_PATH, header=None, names=col_names, low_memory=False)
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df = df.dropna(subset=["latitude", "longitude"])

# Filter to analysis period (consistent with Phase E primary analysis)
df["_dt"] = pd.to_datetime(df["datetime"], errors="coerce")
df["_year"] = df["_dt"].dt.year
df = df[(df["_year"] >= 1990) & (df["_year"] <= 2014)]

# Filter to West Coast bounding box (with small buffer for scatter)
buf = 2.0
wc = df[
    (df["latitude"] >= LAT_MIN - buf) & (df["latitude"] <= LAT_MAX + buf)
    & (df["longitude"] >= LON_MIN - buf) & (df["longitude"] <= LON_MAX + buf)
]
# Further restrict to US
wc = wc[wc["country"] == "us"]
print(f"  West Coast US reports: {len(wc)}")

# =====================================================================
# 4. BUILD BATHYMETRY COLORMAP
# =====================================================================
# Custom ocean-depth colormap: light blue (shallow) -> deep navy (deep)
ocean_colors = [
    "#E8F4FD",  # very shallow / near-shore — pale ice blue
    "#B3D9F2",  # shallow shelf
    "#6BAED6",  # mid-shelf
    "#3182BD",  # continental slope
    "#1B5DA0",  # deep slope
    "#08306B",  # abyss — dark navy
]
ocean_cmap = mcolors.LinearSegmentedColormap.from_list(
    "ocean_depth", ocean_colors, N=256
)

# =====================================================================
# 5. CREATE FIGURE
# =====================================================================
print("Rendering hero figure ...")
fig = plt.figure(figsize=(12, 8), facecolor="white")
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

# --- 5a. Bathymetry (ocean only) ---
# Mask land (elev > 0) to NaN so it doesn't color
bathy_display = np.where(elev <= 0, elev, np.nan)

# Depth range for color normalization
vmin, vmax = -4000, 0
im = ax.pcolormesh(
    lon_grid, lat_grid, bathy_display,
    cmap=ocean_cmap,
    vmin=vmin, vmax=vmax,
    transform=ccrs.PlateCarree(),
    shading="auto",
    rasterized=True,
    zorder=0,
)

# --- 5b. Land and political features ---
ax.add_feature(
    cfeature.LAND, facecolor=LAND_COLOR, edgecolor="none", zorder=1
)
ax.add_feature(
    cfeature.COASTLINE, linewidth=0.6, edgecolor=LAND_EDGE, zorder=4
)
ax.add_feature(
    cfeature.STATES, linewidth=0.25, edgecolor="#C8C0B8", zorder=3
)
ax.add_feature(
    cfeature.BORDERS, linewidth=0.4, edgecolor=LAND_EDGE, zorder=3
)

# --- 5c. Isobaths ---
isobath_levels = [-2000, -1000, -500, -200, -100]
ax.contour(
    lon_grid, lat_grid, elev,
    levels=isobath_levels,
    colors=["#A0C4E0"] * len(isobath_levels),
    linewidths=[0.2, 0.25, 0.3, 0.35, 0.4],
    alpha=0.45,
    transform=ccrs.PlateCarree(),
    zorder=2,
)

# --- 5d. UAP report scatter (subtle background layer) ---
# Two-pass rendering: faint large halo + smaller opaque dot for density feel
ax.scatter(
    wc["longitude"].values,
    wc["latitude"].values,
    s=8,
    c="#E6B0AA",
    alpha=0.06,
    transform=ccrs.PlateCarree(),
    zorder=5,
    rasterized=True,
    linewidths=0,
)
ax.scatter(
    wc["longitude"].values,
    wc["latitude"].values,
    s=1.0,
    c="#E74C3C",
    alpha=0.15,
    transform=ccrs.PlateCarree(),
    zorder=5,
    rasterized=True,
    linewidths=0,
)

# --- 5e. Canyon cells (prominent) ---
# Use gradient magnitude for color intensity
canyon_grads = grad_mag[canyon_cells]
canyon_norm = mcolors.Normalize(vmin=60, vmax=200, clip=True)

# Create a warm gradient: orange -> bright red
canyon_cmap = mcolors.LinearSegmentedColormap.from_list(
    "canyon_heat", ["#F39C12", "#E74C3C", "#C0392B"], N=256
)

ax.scatter(
    canyon_lons,
    canyon_lats,
    s=1.0,
    c=canyon_grads,
    cmap=canyon_cmap,
    norm=canyon_norm,
    alpha=0.8,
    transform=ccrs.PlateCarree(),
    zorder=6,
    rasterized=True,
    linewidths=0,
)

# --- 5f. Named canyon labels ---
# Halo effect for readability
halo = [pe.withStroke(linewidth=2.5, foreground="white")]

canyons = {
    "Puget Sound\n& Juan de Fuca": {
        "xy": (-123.5, 48.2),
        "text": (-127.8, 50.5),
    },
    "Astoria\nCanyon": {
        "xy": (-124.6, 46.2),
        "text": (-128.5, 47.2),
    },
    "Monterey\nCanyon": {
        "xy": (-122.0, 36.8),
        "text": (-127.5, 37.8),
    },
    "Mugu\nCanyon": {
        "xy": (-119.1, 34.05),
        "text": (-124.5, 34.5),
    },
    "La Jolla /\nScripps Canyon": {
        "xy": (-117.3, 32.85),
        "text": (-122.5, 31.5),
    },
}

for name, pos in canyons.items():
    ax.annotate(
        name,
        xy=pos["xy"],
        xytext=pos["text"],
        fontsize=8.5,
        fontweight="bold",
        color=ACCENT_DARK,
        transform=ccrs.PlateCarree(),
        arrowprops=dict(
            arrowstyle="-|>",
            color=ACCENT_DARK,
            lw=1.0,
            connectionstyle="arc3,rad=-0.15",
        ),
        bbox=dict(
            boxstyle="round,pad=0.25",
            fc="white",
            ec=ACCENT_DARK,
            alpha=0.88,
            lw=0.6,
        ),
        zorder=10,
        path_effects=halo,
    )

# --- 5g. Gridlines ---
gl = ax.gridlines(
    draw_labels=True,
    linewidth=0.2,
    alpha=0.25,
    linestyle="--",
    color=GRAY_MED,
)
gl.top_labels = False
gl.right_labels = False
gl.xlocator = mticker.FixedLocator(range(-130, -114, 3))
gl.ylocator = mticker.FixedLocator(range(30, 54, 3))
gl.xlabel_style = {"size": 8, "color": GRAY_DARK}
gl.ylabel_style = {"size": 8, "color": GRAY_DARK}

# --- 5h. Title ---
ax.set_title(
    "Submarine Canyon Bathymetry and UAP Report Distribution\n"
    "US West Coast  |  NOAA ETOPO 2022  |  NUFORC 1990\u20132014",
    fontsize=13,
    fontweight="bold",
    color=GRAY_DARK,
    pad=14,
    linespacing=1.4,
)

# --- 5i. Colorbar for bathymetry ---
cax = fig.add_axes([0.92, 0.25, 0.015, 0.5])  # [left, bottom, width, height]
cb = fig.colorbar(im, cax=cax, orientation="vertical")
cb.set_label("Ocean Depth (m)", fontsize=9, color=GRAY_DARK, labelpad=8)
cb.ax.tick_params(labelsize=8, colors=GRAY_DARK)
cb.set_ticks([-4000, -3000, -2000, -1000, -500, 0])
cb.set_ticklabels(["-4000", "-3000", "-2000", "-1000", "-500", "0"])
cb.outline.set_edgecolor(GRAY_MED)
cb.outline.set_linewidth(0.5)

# --- 5j. Legend ---
from matplotlib.lines import Line2D

legend_elements = [
    Line2D(
        [0], [0],
        marker="o", color="w", markerfacecolor=ACCENT,
        markersize=6, label="Canyon cell (gradient > 60 m/km)",
        markeredgewidth=0,
    ),
    Line2D(
        [0], [0],
        marker="o", color="w", markerfacecolor="#E74C3C",
        markersize=5, alpha=0.4,
        label=f"UAP report (n = {len(wc):,})",
        markeredgewidth=0,
    ),
]
leg = ax.legend(
    handles=legend_elements,
    loc="lower left",
    fontsize=8.5,
    frameon=True,
    fancybox=True,
    framealpha=0.9,
    edgecolor=GRAY_MED,
    borderpad=0.8,
)
leg.get_frame().set_linewidth(0.5)

# --- 5k. Attribution footnote ---
fig.text(
    0.5, 0.01,
    "Bathymetry: NOAA ETOPO 2022  |  Canyon cells: shelf 0 to \u2212500 m, gradient > 60 m/km  |  "
    f"Reports: NUFORC (n = {len(wc):,} West Coast US)",
    ha="center",
    fontsize=7.5,
    color=GRAY_MED,
    style="italic",
)

# =====================================================================
# 6. SAVE
# =====================================================================
out_path = os.path.join(FIG_DIR, "figure_hero_bathymetry.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.3)
plt.close()
print(f"\nHero figure saved: {out_path}")
print(f"  Canyon cells plotted: {len(canyon_lats)}")
print(f"  UAP reports plotted: {len(wc)}")
print("Done.")
