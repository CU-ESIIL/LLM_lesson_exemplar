# Examples

This page contains examples for using the geospatial harmonization tools in this repository.

---

## Geospatial Data Harmonization

The [`geospatial_harmonizer`](../src/geospatial_harmonizer.py) module helps harmonize multiple geospatial datasets by projecting them to a common CRS, clipping to a common extent, and creating visualizations.

### Running the Colorado Example

The main example harmonizes three datasets for Colorado fire risk analysis:

```bash
python examples/colorado_harmonization.py
```

This downloads and processes:
- Wildfire Hazard Potential (raster) from GeoPlatform ImageServer
- MTBS Burned Area Boundaries (vector, rasterized) from USGS
- Microsoft Building Footprints (vector, rasterized) for Colorado

Output is saved to `output/colorado_harmonized_output/`.

### Programmatic Usage

Import the harmonization functions and run a custom workflow:

```python
from pathlib import Path
from src.geospatial_harmonizer import (
    DatasetSpec,
    ExampleWorkflow,
    run_harmonization_example,
)

# Define your target grid
TARGET_CRS = "EPSG:4326"
TARGET_EXTENT = (-109.05, 36.99, -102.04, 41.01)  # Colorado bounding box
TARGET_RESOLUTION = 0.00243  # ~270m in degrees
OUTPUT_DIR = Path("./my_harmonized_output")

# Define datasets to harmonize
DATASETS = [
    DatasetSpec(
        name="my_raster",
        url="https://example.com/data.tif",
        data_type="raster",
    ),
    DatasetSpec(
        name="my_vector",
        url="https://example.com/data.zip",
        data_type="vector",
        rasterize=True,
        burn_value=1,
    ),
]

# Run the workflow
workflow = ExampleWorkflow(
    name="my_workflow",
    datasets=DATASETS,
    target_crs=TARGET_CRS,
    target_extent=TARGET_EXTENT,
    target_resolution=TARGET_RESOLUTION,
    output_dir=OUTPUT_DIR,
    create_visualization=True,
    verbose=True,
)

output_files = run_harmonization_example(workflow)
```

### Supported Data Types

- **Raster**: GeoTIFF, COG, IMG (downloaded and harmonized)
- **Vector**: GeoJSON, Shapefile (rasterized to match raster grid)
- **Archives**: ZIP files (automatically extracted)

### ArcGIS ImageServer Support

The harmonizer can download directly from ArcGIS ImageServer endpoints:

```python
DATASETS = [
    DatasetSpec(
        name="wildfire_hazard",
        url="https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/USFS_EDW_RMRS_WildfireHazardPotentialContinuous/ImageServer",
        data_type="raster",
    ),
]
```

### Output Structure

After running, the output directory contains:

```
output/
└── colorado_harmonized_output/
    ├── harmonized_wildfire_hazard.tif
    ├── harmonized_mtbs_burned_areas.tif
    ├── harmonized_building_footprints.tif
    └── harmonized_visualization.png
```

All harmonized rasters share:
- Common CRS (e.g., EPSG:4326)
- Common extent (bounding box)
- Common resolution (pixel size)

---

## Core Functions

### `build_grid_spec(target_crs, target_extent, target_resolution)`

Creates a `GridSpec` defining the target coordinate system and resolution.

### `download_file(url, output_dir, verbose)`

Downloads a file from a URL to the specified directory.

### `harmonize_raster(input_path, grid, output_path, verbose)`

Reprojects and resamples a raster to match the target grid.

### `rasterize_vector_to_grid(input_path, grid, output_path, burn_value, verbose)`

Rasterizes vector geometries onto the target grid.

### `create_visualization(outputs, output_path, verbose)`

Creates a multi-panel matplotlib visualization of harmonized outputs.

### `run_harmonization_example(workflow)`

Main entry point that orchestrates the complete harmonization workflow.
