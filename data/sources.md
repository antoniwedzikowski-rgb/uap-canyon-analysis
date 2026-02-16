# Data Sources

All data files are included in this repository except ETOPO1 (52 MB netCDF), which must be downloaded separately.

## NUFORC Reports
- **File**: `nuforc_reports.csv` (13 MB, included)
- **Source**: National UFO Reporting Center
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

## Port/Marina Locations
- **File**: `port_coords_cache.npz` (122 KB, included)
- **Source**: OpenStreetMap via Overpass API
- **Records**: 7,747 port and marina locations along US coastline
- **Description**: Cached numpy array of coordinates; originally downloaded at runtime via Overpass API query for amenity=port/marina within CONUS coastal bounding box
