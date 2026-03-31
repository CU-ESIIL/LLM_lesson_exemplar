# AGENTS.md

## Core Operating Contract

- Treat this repository as the source of truth.
- Treat the website as a rendered view of repository state.
- Prefer small, additive, traceable edits.
- Keep documentation synchronized with code and project structure.
- Keep the repository minimalist by default.

---

## Default Workflow

- Inspect repository structure before editing.
- Make the smallest diff that solves the request.
- Update related docs when behavior, workflows, or outputs change.
- Update changelog, dev log, or equivalent history files for meaningful changes.
- Preserve existing structure and historical context.
- Do not perform destructive rewrites unless explicitly requested.

---

## Documentation and Website Policy

- Treat `docs/` as project-level documentation and website source.
- Update docs whenever code, workflows, or outputs change.
- Amend existing docs when possible; do not replace whole files without need.
- Preserve navigation, readability, and consistency in website changes.
- Keep default website behavior clean and minimal unless the user asks for more expressive design.

---

## Testing Policy

- Assume `tests/` may exist before a full testing framework is defined.
- Do not invent domain-specific tests when expected behavior is unclear.
- Add the smallest meaningful tests when behavior is known.
- Prefer early-stage checks such as smoke tests, import tests, CLI tests, schema checks, or example-based checks.
- If tests are deferred, document the gap; do not imply coverage that does not exist.

---

## Package and Structure Separation Policy

- Keep website structure and package structure clearly separated.
- Do not automatically repurpose `docs/` for package-native docs or build artifacts.
- For Python packaging requests, prefer standard Python layout, typically `src/`.
- For R packaging requests, follow standard R conventions (`R/`, `man/`, `DESCRIPTION`, `NAMESPACE`, optional `vignettes/`).
- For other ecosystems, follow ecosystem conventions.
- If structural conflicts arise, choose a durable long-term structure and document the decision.

---

## Data Discovery and Data Use Policy

- Prefer open and FAIR data when possible.
- Prefer streaming or lazy-access workflows over bulk downloads when feasible.
- Use standards-based discovery systems (for example STAC) when relevant.
- When relevant, consider streaming-friendly tooling such as xarray, zarr, GDAL, rasterio, pystac-client, stackstac, gdalcubes, terra, stars, cubo, or equivalent tools.
- When introducing data, document source, access method, format, license, and citation requirements.
- Do not silently ingest external data into the project.

---

## Data Acquisition and Preparation Policy

The agent MUST support workflows where users provide datasets via URLs (e.g., zipped archives, cloud-hosted files, or geodatabases).

### Supported Input Types

- Direct raster files (GeoTIFF, COG)
- Vector files (GeoJSON, Shapefile, GDB)
- Compressed archives (`.zip`, `.tar`, `.gz`)
- Remote URLs (HTTP, HTTPS, S3)

---

### Data Acquisition Workflow

When a dataset is provided as a URL, the agent MUST:

1. **Download**

   - Save to a structured data directory (e.g., `data/raw/`)
   - Preserve original filenames
   - Avoid overwriting existing data unless explicitly instructed

2. **Extract (if needed)**

   - Detect archive type automatically
   - Extract to:

     ```
     data/raw/<dataset_name>/
     ```

3. **Discover + Select Data**

   - Identify valid geospatial files:

     - Raster: `.tif`, `.tiff`
     - Vector: `.shp`, `.geojson`, `.gdb`
   - If multiple valid files exist → ask the user which to use
   - Prefer primary or highest-quality datasets when obvious

---

### Data Organization

The agent SHOULD organize data as:

```
data/
  raw/
  processed/
    harmonized/
```

---

### Reproducibility Requirements

The agent MUST:

- Log all data sources (URLs)
- Preserve raw data unchanged
- Document extraction and preprocessing steps

---

### Failure Handling

If:

- Download fails → retry or notify user
- Archive cannot be read → report issue
- No valid geospatial files found → stop and ask user

The agent MUST NOT proceed with incomplete or ambiguous data.

---

## Data Sovereignty and Intellectual Property Policy

- Consider licensing, copyright, privacy, Indigenous data sovereignty, and related restrictions for all data and content.
- If rights or permissions are unclear, document uncertainty and avoid assuming open reuse.

---

## Design and Usability Policy

- Keep the website simple, readable, and easy to extend by default.
- When design improvements are requested, prioritize system-level improvements (layout, spacing, typography, hierarchy, navigation, consistency).
- Do not use scattered one-off styling hacks.

---

## Decision Logging

- Reflect meaningful structural, architectural, documentation, data-source, or design decisions in changelog, dev log, roadmap, or equivalent history files when appropriate.

---

# Geospatial Harmonization Agent (LLM-Guided Workflow)

## Purpose

The Geospatial Harmonization Agent standardizes multiple geospatial datasets (raster and vector) into a common spatial support so they can be directly compared or analyzed.

---

## Supported Data Types

- Raster (GeoTIFF, COG, etc.)
- Vector (GeoJSON, Shapefile, GDB)

---

## Required User Inputs

The agent MUST ensure the following inputs are defined before execution:

- `target_crs` (e.g., EPSG:4326)
- `target_extent` (xmin, ymin, xmax, ymax)
- `input_datasets` (local paths or URLs)

If any are missing → ask the user before proceeding.

---

## Resolution Harmonization Policy

If multiple raster datasets have different resolutions:

The agent MUST:

1. Detect mismatch

2. Ask the user:

   “Do you want to:
   (a) upsample to the finest resolution
   (b) downsample to the coarsest resolution
   (c) specify a custom resolution?”

3. Apply consistently across all outputs

### Resampling Rules

- Categorical → nearest
- Continuous → bilinear (default) or cubic
- Unknown → ask

---

## Vector ↔ Raster Strategy

When vector data is present:

The agent MUST ask:

“Should vector data be rasterized to match the raster grid?”

If YES:

- Rasterize using target CRS, extent, and resolution
- Ask for attribute field if needed

If NO:

- Keep vector format
- Align CRS and extent only

---

## Harmonization Workflow

The agent MUST execute:

1. Inspect datasets
2. Validate inputs
3. Reproject to target CRS
4. Clip to target extent
5. Harmonize resolution (raster only)
6. Handle vector conversion if needed
7. Save outputs
8. Generate visualization

---

## Output Requirements

### Harmonized Data

- Shared CRS
- Shared extent
- Shared resolution (if raster)
- Saved with:

  ```
  harmonized_<original_name>.tif
  ```

### Visualization

- Multi-panel or overlay map
- Saved as:

  ```
  harmonized_visualization.png
  ```

---

## Interaction Model

The agent SHOULD:

- Ask clarifying questions
- Explain decisions briefly
- Surface tradeoffs

The agent SHOULD NOT:

- Make silent assumptions
- Perform destructive operations without confirmation

---

## Example Workflows

The agent SHOULD support real-world workflows involving:

- Remote datasets provided via URL
- Mixed raster and vector inputs
- Harmonization to a user-defined CRS and extent

See:

- `examples/colorado_harmonization.py`
- `notebooks/colorado_harmonization_demo.ipynb`

---

## Implementation Notes

- Core implementation: `src/geospatial_harmonizer.py`
- Raster processing uses rasterio
- Visualization via matplotlib
- Vector support may require preprocessing

---

## Known Limitations

- GDAL backend reprojection incomplete
- Vector handling not fully implemented
- Resolution selection handled at LLM layer
