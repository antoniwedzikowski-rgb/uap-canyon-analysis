# Data Sources

This document describes all data sources used in the analysis. **Core datasets** for the primary pipeline (scoring + evaluation) are included in the repository. **External datasets** required for confound tests, robustness checks, and international replication must be downloaded separately or are fetched at runtime by the relevant scripts.

### Included in repository
- `nuforc_reports.csv` — primary UAP report data (13 MB)
- `census_county_pop.json` — US Census 2020 population (164 KB)
- `county_centroids_pop.csv` — county geographic centroids (158 KB)
- `military_bases_us.csv` — DoD installation locations (6 KB)
- `port_coords_cache.npz` — port/marina coordinates (122 KB)

### External (not included, downloaded separately or at runtime)
- ETOPO1 bathymetry (52 MB netCDF — see download instructions below)
- EMAG2v3 magnetic anomaly grid (used by `phase_e_magnetic_confound.py`)
- MODIS chlorophyll-a (fetched at runtime by `phase_e_chla_confound.py` via ERDDAP)
- NOAA ESI shoreline classification (used by `phase_e_esi_shoretype.py`)
- Navy OPAREA polygons (fetched at runtime by `phase_e_oparea_confound.py` via MarineCadastre API)
- SRTM30 Norway bathymetry (used by `phase_e_norway_replication.py`)
- Post-2014 NUFORC data (used by `phase_e_replication_suite.py`, sourced from HuggingFace kcimc/NUFORC)
- USGS earthquake catalog, Kp geomagnetic index (used by earlier Phase C/D scripts, not part of final pipeline)

## NUFORC Reports
- **File**: `nuforc_reports.csv` (13 MB, included)
- **Source**: National UFO Reporting Center, via [planetsig/ufo-reports](https://github.com/planetsig/ufo-reports) (public geocoded scrape)
- **URL**: https://nuforc.org/webreports/
- **Records**: 80,332 total reports (scraped snapshot ending May 2014; NUFORC has since added ~60K newer reports not included here)
- **Filtering applied in scripts**: Bounding box (20°N–55°N, 135°W–55°W), years 1990–2014, valid lat/lon → 42,008 coastal reports used in analysis
- **Format**: Headerless CSV
- **Columns**: datetime_str, city, state, country, shape, duration_seconds, duration_text, description, date_posted, lat, lon
- **Note**: lat/lon columns contain mixed types; scripts use `pd.to_numeric(errors='coerce')`

## ETOPO1 Bathymetry
- **File**: `etopo_subset.nc` (52 MB, NOT included — download separately)
- **Source**: NOAA ETOPO1 Global Relief Model
- **URL**: https://www.ngdc.noaa.gov/mgg/global/global.html
- **Direct download**: https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/bedrock/grid_registered/netcdf/ETOPO1_Bed_g_gmt4.grd.gz
- **Resolution**: 1 arc-minute (~1.8 km)
- **Subset**: US coastal waters, used to compute bathymetric gradient (m/km) per grid cell
- **Canyon proxy**: cells with gradient >60 m/km (85% overlap with mapped submarine canyons within 25 km)
- **How to create the subset**: Download the global ETOPO1 file, then subset to the analysis bounding box (20°N–55°N, 135°W–55°W):
  ```python
  import xarray as xr
  ds = xr.open_dataset("ETOPO1_Bed_g_gmt4.grd")
  subset = ds.sel(lat=slice(20, 55), lon=slice(-135, -55))
  subset.to_netcdf("etopo_subset.nc")
  ```
  The resulting file should contain variable `z` (elevation in meters) with dimensions `lat` and `lon`.

## US Census Population
- **File**: `census_county_pop.json` (164 KB, included)
- **Source**: US Census Bureau, 2020 Decennial Census (P1_001N total population)
- **URL**: https://data.census.gov/
- **Description**: County-level population counts used for control point generation and population density covariates

## County Centroids
- **File**: `county_centroids_pop.csv` (158 KB, included)
- **Source**: US Census Bureau Gazetteer Files
- **URL**: https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html
- **Description**: Geographic centroids (lat/lon) and population for all US counties

## Military Bases
- **File**: `military_bases_us.csv` (6 KB, included)
- **Source**: Compiled from public DoD installation data
- **Records**: 171 bases (58 AFB, 33 Army, 27 Navy, 14 DOE, 12 NG, 12 Marines, 7 DOD, 5 USCG, 3 Test)
- **Columns**: name, lat, lon, type
- **Note**: Measures distance to physical installations, not offshore training ranges (OPAREAs)

## Military Operating Areas (OPAREAs)
- **File**: `oparea_polygons.json` (cached GeoJSON, downloaded at runtime)
- **Source**: NOAA MarineCadastre Military Operating Area Boundaries
- **URL**: https://marinecadastre.gov/downloads/data/mc/MilitaryCollection.zip
- **API**: https://coast.noaa.gov/arcgis/rest/services/Hosted/MilitaryOperatingAreas/FeatureServer/0
- **Records**: 35 polygon features (Navy Common Operating Picture, published Dec 2018)
- **Description**: Offshore military operating/training area boundaries used to test whether UAP report density correlates with proximity to Navy operations rather than canyon topography
- **Key West Coast OPAREAs**: SOCAL Range Complex (traces SD coastline), PMSR (104 km offshore), PACNORWEST (~18 km offshore), Navy 3/7 and Carr Inlet (small Puget Sound waterway areas)
- **Note**: Does NOT include all military testing facilities (e.g., Dabob Bay NUWC is absent). SOCAL Range Complex boundary traces the actual San Diego coastline, making dist_to_OPAREA a proxy for coastal distance in SoCal — see regional analysis in phase_e_oparea_confound.py

## Port/Marina Locations
- **File**: `port_coords_cache.npz` (122 KB, included)
- **Source**: OpenStreetMap via Overpass API
- **Records**: 7,747 port and marina locations along US coastline
- **Description**: Cached numpy array of coordinates; originally downloaded at runtime via Overpass API query for amenity=port/marina within CONUS coastal bounding box
