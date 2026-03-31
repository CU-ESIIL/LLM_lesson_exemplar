# Geospatial Harmonization Examples

This directory contains runnable examples that show how an LLM can use this repository to harmonize geospatial datasets from user-provided URLs.

The current example demonstrates a real Colorado workflow using shared Python code from `src/geospatial_harmonizer.py`.

## Current Example

### `colorado_harmonization.py`

This example shows how to harmonize multiple datasets for Colorado using a common CRS, extent, and target resolution.

It demonstrates a workflow using:

* wildfire hazard potential
* MTBS burned area boundaries
* Microsoft building footprints

The example:

* downloads data from URLs
* extracts archives when needed
* identifies raster and vector inputs
* reprojects data to a shared CRS
* clips data to Colorado
* rasterizes vector layers when needed
* saves harmonized outputs
* generates a visualization

## Quick Start

Install dependencies:

```bash
pip install -r ../requirements.txt
```

Run the example:

```bash
python colorado_harmonization.py
```

## How This Directory Fits Into the Repository

This directory is for concrete, runnable examples.

* `AGENTS.md` defines how the LLM should behave
* `src/geospatial_harmonizer.py` contains shared harmonization logic
* `examples/` contains real worked examples

## Finding Data URLs

Many geospatial workflows begin with remote datasets. Common sources include:

| Source                                                                   | Description                                                  |
| ------------------------------------------------------------------------ | ------------------------------------------------------------ |
| [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) | Large catalog of geospatial collections exposed through STAC |
| [AWS Open Data](https://registry.opendata.aws/?search=tags:geospatial)   | Public geospatial datasets including imagery and terrain     |
| [NASA Earthdata](https://search.earthdata.nasa.gov/)                     | Earth observation and environmental datasets                 |
| [USGS National Map](https://apps.nationalmap.gov/downloader/)            | Elevation, hydrography, land cover, and related products     |

## Example: Using STAC for Data Discovery

```python
import pystac_client

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

search = catalog.search(
    collections=["nlcd"],
    bbox=[-106.0, 38.5, -103.5, 40.5],
    datetime="2019-01-01/2019-12-31"
)

for item in search.items():
    print(f"{item.id}: {item.assets['data'].href}")
```

## Expected Output Structure

After running the example, outputs should look something like:

```text
colorado_harmonized_output/
  colorado_harmonized_wildfire_hazard.tif
  colorado_harmonized_mtbs_burned_areas.tif
  colorado_harmonized_building_footprints.tif
  colorado_harmonized_visualization.png
```

## Notes

This directory is intentionally example-oriented. It is meant to show users how the repository can be used with real datasets and URL-based inputs.

For general LLM behavior and workflow rules, see the repository-level `AGENTS.md`.
