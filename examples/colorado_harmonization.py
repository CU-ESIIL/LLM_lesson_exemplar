#!/usr/bin/env python3
"""
Colorado fire risk harmonization example.

This example demonstrates how to harmonize multiple geospatial datasets
for Colorado using shared functions from src/geospatial_harmonizer.py.

Datasets:
1. FBFM40 Fire Behavior Fuel Models (raster) - Landfire 2024
2. MTBS Burned Area Boundaries (vector, rasterized)
3. Microsoft Building Footprints (vector, rasterized)

All outputs are harmonized to:
- CRS: EPSG:4326
- Extent: Colorado
- Resolution: ~270 m (in degrees, approximated for Colorado)
"""

from pathlib import Path
from src.geospatial_harmonizer import (
    DatasetSpec,
    ExampleWorkflow,
    run_harmonization_example,
)

# Colorado bounding box in EPSG:4326
COLORADO_EXTENT = (-109.05, 36.99, -102.04, 41.01)

# Common output settings
TARGET_CRS = "EPSG:4326"

# Approximate degree resolution for ~270 m at Colorado latitude
TARGET_RESOLUTION = 0.00243

OUTPUT_DIR = Path("./output/colorado_harmonized_output")

# Example datasets
# Using Landfire FBFM40 (Scott and Burgan Fire Behavior Fuel Models) - CONUS 2024
# Note: WMS returns rendered images (not raw data), so we use direct download
# The harmonizer now caches results, so subsequent runs are fast
DATASETS = [
    DatasetSpec(
        name="fbfm40_fuel_models",
        url="https://www.landfire.gov/data-downloads/CONUS_LF2024/LF2024_FBFM40_CONUS.zip",
        data_type="raster",
    ),
    DatasetSpec(
        name="mtbs_burned_areas",
        url="https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip",
        data_type="vector",
        rasterize=True,
        burn_value=1,
    ),
    DatasetSpec(
        name="building_footprints",
        url="https://minedbuildings.z5.web.core.windows.net/legacy/usbuildings-v2/Colorado.geojson.zip",
        data_type="vector",
        rasterize=True,
        burn_value=1,
    ),
]

def main() -> int:
    workflow = ExampleWorkflow(
        name="colorado_fire_risk",
        datasets=DATASETS,
        target_crs=TARGET_CRS,
        target_extent=COLORADO_EXTENT,
        target_resolution=TARGET_RESOLUTION,
        output_dir=OUTPUT_DIR,
        create_visualization=True,
        verbose=True,
    )

    output_files = run_harmonization_example(workflow)

    print("\nHarmonization complete.")
    print(f"\nOutputs saved to: {OUTPUT_DIR.resolve()}")

    print("\nGenerated files:")
    for path in output_files:
        print(f" - {path.name}")

    viz_path = OUTPUT_DIR / "harmonized_visualization.png"
    html_path = OUTPUT_DIR / "harmonized_visualization.html"
    
    print("\nGenerated visualizations:")
    if viz_path.exists():
        print(f" - Static PNG: {viz_path.name}")
    if html_path.exists():
        print(f" - Interactive HTML: {html_path.name}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())