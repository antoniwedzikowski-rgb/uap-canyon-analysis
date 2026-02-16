# Data Sources

Raw data files are excluded from this repository due to size. Below are download instructions.

## NUFORC Reports
- **File**: `nuforc_reports.csv` (~14 MB)
- **Source**: National UFO Reporting Center (https://nuforc.org)
- **Description**: 80,332 sighting reports with datetime, location, shape, duration
- **Format**: Headerless CSV with columns: datetime_str, city, state, country, shape, duration_seconds, duration_text, description, date_posted, lat, lon

## ETOPO1 Bathymetry
- **File**: `etopo_subset.nc` (~52 MB)
- **Source**: NOAA ETOPO1 Global Relief Model
- **URL**: https://www.ngdc.noaa.gov/mgg/global/
- **Description**: 1 arc-minute resolution bathymetry/topography grid, subset to US coastal waters

## US Census Population
- **File**: `census_county_pop.json` (~165 KB)
- **Source**: US Census Bureau
- **Description**: County-level population counts for control point generation

## County Centroids
- **File**: `county_centroids_pop.csv` (~160 KB)
- **Source**: US Census Bureau Gazetteer Files
- **Description**: Geographic centroids and population for all US counties

## Military Bases
- **File**: `military_bases_us.csv` (~6 KB)
- **Source**: Compiled from public DoD data
- **Description**: US military installation locations (171 bases)

## Port/Marina Locations
- **File**: Downloaded at runtime via Overpass API
- **Source**: OpenStreetMap
- **Description**: 7,747 port and marina locations along US coastline
- **Cache**: `port_coords_cache.npz`
