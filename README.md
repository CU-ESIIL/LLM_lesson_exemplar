# Geospatial Harmonization with LLMs

This repository demonstrates how large language models (LLMs) can be used to harmonize geospatial datasets from user-provided URLs.

Instead of writing custom scripts for each dataset, the workflow is:

1. A user provides dataset URLs
2. The LLM inspects the datasets
3. The LLM decides how to harmonize them (CRS, extent, resolution)
4. Shared Python functions perform the harmonization
5. Outputs and maps are generated

---

## What This Repository Does

The harmonization workflow supports:

* downloading datasets from URLs
* extracting archives (e.g., ZIP files)
* identifying raster and vector inputs
* reprojecting to a common CRS
* clipping to a shared extent
* aligning raster resolution
* optionally rasterizing vector data
* saving harmonized outputs
* generating visualizations

---

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the example:

```bash
python examples/colorado_harmonization.py
```

---

## Example: Colorado Fire Risk

This repository includes a worked example that harmonizes:

* wildfire hazard potential (raster)
* MTBS burned area boundaries (vector)
* Microsoft building footprints (vector)

All datasets are harmonized to:

* CRS: EPSG:4326
* Extent: Colorado
* Common resolution

Goal:

> Identify where wildfire risk overlaps with past fires and human infrastructure.

See:

* `examples/colorado_harmonization.py`
* `notebooks/colorado_harmonization_demo.ipynb`

---

## How to Use This Repository with an LLM

You can prompt an LLM with something like:

> “Download these datasets, harmonize them to EPSG:4326 over Colorado, and generate a map.”

The LLM should:

* download and inspect the datasets
* determine raster vs vector inputs
* ask about resolution mismatches if needed
* reproject and clip datasets
* optionally rasterize vector data
* generate harmonized outputs and a visualization

The expected behavior is defined in:

```
AGENTS.md
```

---

## Repository Structure

```text
src/
  geospatial_harmonizer.py   # core harmonization logic

examples/
  colorado_harmonization.py  # runnable example

docs/
  # website and documentation
```

---

## Python API

You can also run harmonization directly in Python:

```python
from src.geospatial_harmonizer import GeospatialHarmonizer, HarmonizationConfig

config = HarmonizationConfig(
    input_files=["file1.tif", "file2.tif"],
    output_crs="EPSG:4326",
    output_extent=(-109.05, 36.99, -102.04, 41.01),
    output_dir="./output"
)

harmonizer = GeospatialHarmonizer(config)
output_files = harmonizer.harmonize()
```

---

## Design Philosophy

* The LLM handles decision-making and orchestration
* The Python code handles geospatial processing
* Examples demonstrate real workflows
* The system is reusable across datasets

---

## Documentation Website

This repository includes a documentation site built with MkDocs.

To preview locally:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Then open:

```
http://127.0.0.1:8000
```

---

## Notes

This repository is designed as a teaching and demonstration tool.

It shows how LLMs can:

* reason about geospatial data
* make harmonization decisions
* orchestrate reusable processing code

rather than requiring custom scripts for each dataset.