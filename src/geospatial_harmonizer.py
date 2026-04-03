#!/usr/bin/env python3
"""
Reusable geospatial harmonization utilities for URL-driven examples.

This module supports mixed raster/vector workflows where datasets are:
1. downloaded from URLs
2. extracted if needed
3. harmonized to a common CRS, extent, and resolution
4. saved to disk
5. visualized as aligned outputs

Supports WCS (Web Coverage Service) for efficient subsetting of large raster datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
import shutil
import urllib.request
import urllib.parse
import zipfile
import io
import base64
import xml.etree.ElementTree as ET

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio import mask
from shapely.geometry import box

try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False


BBox = tuple[float, float, float, float]

# Global color map discovered from source data (used by visualization functions)
DISCOVERED_COLOR_MAP: dict | None = None
# Global labels map: integer value -> short label string (e.g. 101 -> "GR1")
DISCOVERED_LABELS: dict | None = None


@dataclass
class DatasetSpec:
    name: str
    url: str
    data_type: Literal["raster", "vector"]
    rasterize: bool = False
    burn_value: int = 1
    is_wcs: bool = False
    wcs_layer: str = None
    is_wms: bool = False
    wms_layer: str = None
    labels_url: str | None = None  # URL to a CSV with VALUE,LABEL columns for legend labels


@dataclass
class ExampleWorkflow:
    name: str
    datasets: list[DatasetSpec]
    target_crs: str
    target_extent: BBox
    target_resolution: float
    output_dir: Path
    create_visualization: bool = True
    verbose: bool = True


@dataclass
class GridSpec:
    crs: str
    extent: BBox
    resolution: float
    width: int
    height: int
    transform: object


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def build_grid_spec(target_crs: str, target_extent: BBox, target_resolution: float) -> GridSpec:
    xmin, ymin, xmax, ymax = target_extent
    width = int(round((xmax - xmin) / target_resolution))
    height = int(round((ymax - ymin) / target_resolution))
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    return GridSpec(
        crs=target_crs,
        extent=target_extent,
        resolution=target_resolution,
        width=width,
        height=height,
        transform=transform,
    )


def download_file(url: str, output_dir: Path, verbose: bool = True) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(url).name

    if output_path.exists():
        _log(f"Using existing download: {output_path}", verbose)
        return output_path

    _log(f"Downloading: {url}", verbose)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(output_path, "wb") as f:
        shutil.copyfileobj(response, f)

    return output_path


def download_arcgis_image_server(
    url: str,
    output_dir: Path,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    verbose: bool = True,
) -> Path:
    """Download raster from ArcGIS ImageServer REST API.
    
    Args:
        url: ImageServer REST endpoint
        output_dir: Directory to save output
        bbox: Bounding box (xmin, ymin, xmax, ymax) in EPSG:4326
        width: Output width in pixels
        height: Output height in pixels
        verbose: Print progress messages
        
    Returns:
        Path to downloaded TIFF file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "wildfire_hazard.tif"
    
    if output_path.exists():
        _log(f"Using existing download: {output_path}", verbose)
        return output_path
    
    xmin, ymin, xmax, ymax = bbox
    export_url = (
        f"{url}/exportImage"
        f"?bbox={xmin},{ymin},{xmax},{ymax}"
        f"&bboxSR=4326"
        f"&size={width},{height}"
        f"&format=tif"
        f"&imageSR=4326"
        f"&f=image"
    )
    
    _log(f"Downloading from ArcGIS ImageServer: {url}", verbose)
    request = urllib.request.Request(export_url, headers={"User-Agent": "Mozilla/5.0"})
    
    # Download to temp file first
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name
        with urllib.request.urlopen(request) as response:
            shutil.copyfileobj(response, tmp)
    
    # Read the downloaded file and add geotransform
    xmin, ymin, xmax, ymax = bbox
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    
    with rasterio.open(tmp_path) as src:
        data = src.read()
        meta = src.meta.copy()
        meta.update({
            "crs": "EPSG:4326",
            "transform": transform,
            "height": height,
            "width": width,
        })
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(data)
    
    # Clean up temp file
    Path(tmp_path).unlink()
    
    return output_path


def download_wms_coverage(
    wms_url: str,
    layer: str,
    bbox: tuple[float, float, float, float],
    crs: str,
    output_dir: Path,
    output_filename: str,
    width: int = None,
    height: int = None,
    verbose: bool = True,
) -> Path:
    """Download raster coverage from a WMS (Web Map Service).
    
    Args:
        wms_url: Base WMS endpoint URL
        layer: Layer name to request
        bbox: Bounding box (xmin, ymin, xmax, ymax) in the specified CRS
        crs: CRS code (e.g., "EPSG:4326")
        output_dir: Directory to save output
        output_filename: Name for the output file
        width: Optional width in pixels (height calculated proportionally if not provided)
        height: Optional height in pixels (width calculated proportionally if not provided)
        verbose: Print progress messages
        
    Returns:
        Path to downloaded GeoTIFF file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    
    if output_path.exists():
        _log(f"Using existing WMS download: {output_path}", verbose)
        return output_path
    
    # Calculate dimensions if not provided
    if width is None and height is None:
        width = 1000
    elif width is None:
        xmin, ymin, xmax, ymax = bbox
        aspect = (xmax - xmin) / (ymax - ymin)
        width = int(height * aspect)
    elif height is None:
        xmin, ymin, xmax, ymax = bbox
        aspect = (xmax - xmin) / (ymax - ymin)
        height = int(width / aspect)
    
    # Build WMS GetMap request
    params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": layer,
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "crs": crs,
        "width": width,
        "height": height,
        "format": "image/geotiff",
    }
    
    query_string = urllib.parse.urlencode(params)
    request_url = f"{wms_url}?{query_string}"
    
    _log(f"Downloading WMS coverage: {layer}", verbose)
    _log(f"  URL: {request_url[:100]}...", verbose)
    
    try:
        request = urllib.request.Request(request_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=300) as response:
            with open(output_path, "wb") as f:
                shutil.copyfileobj(response, f)
        _log(f"  Downloaded {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)", verbose)
        return output_path
    except Exception as e:
        _log(f"WMS request failed: {e}", verbose)
        raise RuntimeError(f"Failed to download WMS coverage: {e}")


def download_wcs_coverage(
    wcs_url: str,
    layer: str,
    bbox: tuple[float, float, float, float],
    crs: str,
    output_dir: Path,
    output_filename: str,
    width: int = None,
    height: int = None,
    verbose: bool = True,
) -> Path:
    """Download raster coverage from a WCS (Web Coverage Service).
    
    Args:
        wcs_url: Base WCS endpoint URL
        layer: Layer name to request
        bbox: Bounding box (xmin, ymin, xmax, ymax) in the specified CRS
        crs: CRS code (e.g., "EPSG:4326")
        output_dir: Directory to save output
        output_filename: Name for the output file
        width: Optional width in pixels (height calculated proportionally if not provided)
        height: Optional height in pixels (width calculated proportionally if not provided)
        verbose: Print progress messages
        
    Returns:
        Path to downloaded GeoTIFF file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    
    if output_path.exists():
        _log(f"Using existing WCS download: {output_path}", verbose)
        return output_path
    
    # Parse CRS to get EPSG code
    epsg_code = crs.split(":")[-1] if ":" in crs else crs
    
    # Calculate dimensions if not provided
    if width is None and height is None:
        # Default to reasonable size
        width = 1000
    elif width is None:
        # Calculate width based on aspect ratio
        xmin, ymin, xmax, ymax = bbox
        aspect = (xmax - xmin) / (ymax - ymin)
        width = int(height * aspect)
    elif height is None:
        # Calculate height based on aspect ratio
        xmin, ymin, xmax, ymax = bbox
        aspect = (xmax - xmin) / (ymax - ymin)
        height = int(width / aspect)
    
    # Build WCS GetCoverage request
    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "coverageid": layer,
        "format": "image/geotiff",
        "subset": f"Long({bbox[0]},{bbox[2]})",
        "subset": f"Lat({bbox[1]},{bbox[3]})",
        "scalesize": f"Long({width}),Lat({height})",
    }
    
    # For WCS 1.x.x style requests (more commonly supported)
    wcs_111_params = {
        "service": "WCS",
        "version": "1.1.1",
        "request": "GetCoverage",
        "coverage": layer,
        "format": "GeoTIFF",
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "crs": crs,
        "width": width,
        "height": height,
    }
    
    # Try WCS 1.1.1 request first (more widely supported)
    query_string = urllib.parse.urlencode(wcs_111_params)
    request_url = f"{wcs_url}?{query_string}"
    
    _log(f"Downloading WCS coverage: {layer}", verbose)
    _log(f"  URL: {request_url[:100]}...", verbose)
    
    try:
        request = urllib.request.Request(request_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=300) as response:
            with open(output_path, "wb") as f:
                shutil.copyfileobj(response, f)
        _log(f"  Downloaded {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)", verbose)
        return output_path
    except Exception as e:
        _log(f"WCS 1.1.1 request failed: {e}", verbose)
        # Try WCS 2.0.1 style request
        _log("Trying WCS 2.0.1 request...", verbose)
        params_201 = {
            "service": "WCS",
            "version": "2.0.1",
            "request": "GetCoverage",
            "coverageId": layer,
            "format": "image/geotiff",
        }
        query_string = urllib.parse.urlencode(params_201)
        request_url = f"{wcs_url}?{query_string}"
        
        try:
            request = urllib.request.Request(request_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=300) as response:
                with open(output_path, "wb") as f:
                    shutil.copyfileobj(response, f)
            _log(f"  Downloaded {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)", verbose)
            return output_path
        except Exception as e2:
            _log(f"WCS 2.0.1 request also failed: {e2}", verbose)
            raise RuntimeError(f"Failed to download WCS coverage: {e2}")


def extract_archive_if_needed(path: Path, output_dir: Path, verbose: bool = True) -> Path:
    if path.suffix.lower() != ".zip":
        return path.parent

    extract_dir = output_dir / path.stem
    extract_dir.mkdir(parents=True, exist_ok=True)

    if any(extract_dir.iterdir()):
        _log(f"Using existing extraction: {extract_dir}", verbose)
        return extract_dir

    _log(f"Extracting: {path.name}", verbose)
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(extract_dir)

    return extract_dir


def discover_dataset_file(dataset_dir: Path, data_type: str) -> Path:
    if data_type == "raster":
        candidates = (
            list(dataset_dir.rglob("*.tif"))
            + list(dataset_dir.rglob("*.tiff"))
            + list(dataset_dir.rglob("*.img"))
        )
    else:
        candidates = (
            list(dataset_dir.rglob("*.geojson"))
            + list(dataset_dir.rglob("*.shp"))
        )

    if not candidates:
        raise FileNotFoundError(f"No {data_type} dataset found in {dataset_dir}")

    return candidates[0]


# Known color table URLs for common datasets
COLOR_TABLE_URLS = {
    "fbfm40": "https://landfire.gov/sites/default/files/CSV/2024/LF2024_FBFM40.csv",
    "fbfm13": "https://landfire.gov/sites/default/files/CSV/2024/LF2024_FBFM13.csv",
    "evc": "https://landfire.gov/sites/default/files/CSV/2024/LF2024_EVC.csv",
    "evh": "https://landfire.gov/sites/default/files/CSV/2024/LF2024_EVH.csv",
}


def discover_color_map(dataset_dir: Path, dataset_name: str = "") -> dict | None:
    """Discover and parse color map file from the dataset directory.
    
    Looks for common color map file formats:
    - .clr (ESRI color map files)
    - .txt (text-based color maps)
    - .csv (comma-separated color tables like Landfire)
    - .qml (QGIS style files)
    
    Also tries to download from known URLs for Landfire datasets.
    
    Args:
        dataset_dir: Directory to search for color map files
        dataset_name: Optional name of the dataset to help find color table URLs
        
    Returns:
        Dict mapping category values to RGB colors, or None if not found
    """
    # Look for .clr files (ESRI color map format)
    clr_files = list(dataset_dir.rglob("*.clr"))
    if clr_files:
        return _parse_esri_color_map(clr_files[0])
    
    # Look for .txt files that might be color maps
    txt_files = list(dataset_dir.rglob("*.txt"))
    for txt_file in txt_files:
        # Check if it looks like a color map file
        try:
            with open(txt_file, 'r') as f:
                first_lines = [f.readline() for _ in range(5)]
            # ESRI color maps often have category values as first column
            if any(line.strip().split()[:3] for line in first_lines if line.strip()):
                result = _parse_text_color_map(txt_file)
                if result:
                    return result
        except Exception:
            continue
    
    # Look for CSV files that might be color tables (e.g., Landfire)
    csv_files = list(dataset_dir.rglob("*.csv"))
    for csv_file in csv_files:
        result = _parse_csv_color_map(csv_file)
        if result:
            return result
    
    # Try to download from known URLs based on dataset name
    name_lower = dataset_name.lower()
    for key, url in COLOR_TABLE_URLS.items():
        if key in name_lower:
            _log(f"Attempting to download color table from {url}", True)
            try:
                # Download to a temp file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                    tmp_path = tmp.name
                    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(request) as response:
                        shutil.copyfileobj(response, tmp)
                    result = _parse_csv_color_map(Path(tmp_path))
                    Path(tmp_path).unlink()
                    if result:
                        return result
            except Exception as e:
                _log(f"Could not download color table: {e}", True)
                continue
    
    return None


def _parse_esri_color_map(clr_path: Path) -> dict:
    """Parse ESRI .clr color map file.
    
    Format: category R G B
    """
    color_map = {}
    try:
        with open(clr_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        category = int(parts[0])
                        r = int(parts[1])
                        g = int(parts[2])
                        b = int(parts[3])
                        color_map[category] = (r, g, b)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Warning: Could not parse color map {clr_path}: {e}")
        return None
    
    return color_map if color_map else None


def _parse_text_color_map(txt_path: Path) -> dict | None:
    """Parse text-based color map file.
    
    Attempts to parse common formats.
    """
    color_map = {}
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        # Try to parse as category R G B
                        category = int(parts[0])
                        r = int(parts[1])
                        g = int(parts[2])
                        b = int(parts[3])
                        color_map[category] = (r, g, b)
                    except ValueError:
                        continue
    except Exception:
        return None
    
    return color_map if color_map else None


def _parse_csv_color_map(csv_path: Path) -> dict | None:
    """Parse CSV-based color map file (e.g., Landfire color tables).
    
    Expected format: VALUE,FBFM40,R,G,B,RED,GREEN,BLUE
    or similar with VALUE and R, G, B columns.
    """
    color_map = {}
    try:
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = None
            for row in reader:
                if not row:
                    continue
                # First non-empty row is the header
                if header is None:
                    header = [col.strip().upper() for col in row]
                    # Look for VALUE and R, G, B columns
                    if 'VALUE' not in header or 'R' not in header:
                        return None
                    continue
                
                # Skip rows that don't have enough columns
                if len(row) < len(header):
                    continue
                
                try:
                    # Find column indices
                    value_idx = header.index('VALUE')
                    r_idx = header.index('R')
                    g_idx = header.index('G')
                    b_idx = header.index('B')
                    
                    # Parse values
                    value = int(row[value_idx])
                    r = int(row[r_idx])
                    g = int(row[g_idx])
                    b = int(row[b_idx])
                    
                    # Skip NoData values
                    if value < 0 or value == -9999:
                        continue
                    
                    color_map[value] = (r, g, b)
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"Warning: Could not parse CSV color map {csv_path}: {e}")
        return None
    
    return color_map if color_map else None


def harmonize_raster(input_path: Path, grid: GridSpec, output_path: Path, verbose: bool = True) -> Path:
    """Harmonize raster to target grid, clipping to extent first for efficiency."""
    _log(f"Harmonizing raster: {input_path.name}", verbose)

    with rasterio.open(input_path) as src:
        xmin, ymin, xmax, ymax = grid.extent
        
        # Define target bounds in target CRS
        target_bounds = box(xmin, ymin, xmax, ymax)
        
        # If source CRS differs, transform bounds
        if src.crs != grid.crs:
            from rasterio.warp import transform_bounds
            left, bottom, right, top = transform_bounds(
                grid.crs, src.crs, xmin, ymin, xmax, ymax
            )
            src_bounds = box(left, bottom, right, top)
        else:
            src_bounds = target_bounds
        
        # Try to clip source data to bounds before reprojecting
        try:
            out_image, out_transform = mask.mask(
                src, [src_bounds], crop=True, all_touched=True
            )
            
            if out_image.shape[1] == 0 or out_image.shape[2] == 0:
                # Clipping resulted in empty data, use full source
                _log(f"  Clipping resulted in empty data, using full source", verbose)
                out_image = src.read()
                out_transform = src.transform
        except (ValueError, Exception) as e:
            # Clipping failed (e.g., bounds don't overlap or WMS axis order issue), use full source
            _log(f"  Clipping failed: {e}, using full source", verbose)
            out_image = src.read()
            out_transform = src.transform
        
        # Now reproject the clipped data to target grid
        dst = np.zeros((out_image.shape[0], grid.height, grid.width), dtype=out_image.dtype)

        reproject(
            source=out_image,
            destination=dst,
            src_transform=out_transform,
            src_crs=src.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=Resampling.bilinear,
        )

        meta = src.meta.copy()
        meta.update(
            {
                "driver": "GTiff",
                "height": grid.height,
                "width": grid.width,
                "transform": grid.transform,
                "crs": grid.crs,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **meta) as dst_file:
        dst_file.write(dst)

    return output_path


def rasterize_vector_to_grid(
    input_path: Path,
    grid: GridSpec,
    output_path: Path,
    burn_value: int = 1,
    verbose: bool = True,
) -> Path:
    _log(f"Rasterizing vector: {input_path.name}", verbose)

    gdf = gpd.read_file(input_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    if str(gdf.crs) != grid.crs:
        gdf = gdf.to_crs(grid.crs)

    xmin, ymin, xmax, ymax = grid.extent
    clip_geom = box(xmin, ymin, xmax, ymax)
    gdf = gdf[gdf.intersects(clip_geom)].copy()

    shapes = [(geom, burn_value) for geom in gdf.geometry if geom is not None]
    burned = rasterize(
        shapes=shapes,
        out_shape=(grid.height, grid.width),
        transform=grid.transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )

    meta = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": 1,
        "dtype": "uint8",
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(burned, 1)

    return output_path


def harmonize_vector(
    input_path: Path,
    grid: GridSpec,
    output_path: Path,
    verbose: bool = True,
) -> Path:
    """Harmonize vector to target CRS and clip to target extent.
    
    Args:
        input_path: Path to input vector file
        grid: Target grid specification
        output_path: Path to save harmonized vector
        verbose: Print progress messages
        
    Returns:
        Path to harmonized vector file
    """
    _log(f"Harmonizing vector: {input_path.name}", verbose)
    
    gdf = gpd.read_file(input_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    
    # Reproject to target CRS if needed
    if str(gdf.crs) != grid.crs:
        gdf = gdf.to_crs(grid.crs)
    
    # Clip to target extent
    xmin, ymin, xmax, ymax = grid.extent
    clip_geom = box(xmin, ymin, xmax, ymax)
    gdf = gdf[gdf.intersects(clip_geom)].copy()
    
    # Save harmonized vector
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")
    
    _log(f"  Saved harmonized vector: {output_path.name} ({len(gdf)} features)", verbose)
    
    return output_path


def _create_binary_mask(data: np.ndarray) -> np.ndarray:
    """Create a mask for binary data (0s and 1s), returning only 1s as visible."""
    unique_vals = np.unique(data)
    if len(unique_vals) <= 2 and all(v in [0, 1, data.dtype.type(0), data.dtype.type(1)] for v in unique_vals):
        return np.ma.masked_equal(data, 0)
    return data


def create_visualization(outputs: list[tuple[str, Path]], output_path: Path, verbose: bool = True) -> Path | None:
    """Create a static PNG visualization with subplots for each layer."""
    try:
        return _create_visualization_impl(outputs, output_path, verbose)
    except Exception as e:
        _log(f"Error creating visualization: {e}", verbose)
        import traceback
        traceback.print_exc()
        return None


def _load_color_map_from_output_dir(output_path: Path) -> None:
    """Load DISCOVERED_COLOR_MAP and DISCOVERED_LABELS from saved JSON files if not already set."""
    import json
    import src.geospatial_harmonizer as _self
    if _self.DISCOVERED_COLOR_MAP is None:
        for json_path in output_path.parent.glob("*_color_map.json"):
            with open(json_path) as f:
                color_map_json = json.load(f)
            _self.DISCOVERED_COLOR_MAP = {int(k): tuple(v) for k, v in color_map_json.items()}
            break
    if _self.DISCOVERED_LABELS is None:
        for json_path in output_path.parent.glob("*_labels.json"):
            with open(json_path) as f:
                labels_json = json.load(f)
            _self.DISCOVERED_LABELS = {int(k): v for k, v in labels_json.items()}
            break


def _create_visualization_impl(outputs: list[tuple[str, Path]], output_path: Path, verbose: bool = True) -> Path:
    """Internal implementation of PNG visualization creation."""
    _log("Creating visualization", verbose)
    _load_color_map_from_output_dir(output_path)

    n = len(outputs)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, (name, path) in enumerate(outputs):
        _log(f"  Processing layer {i+1}/{len(outputs)}: {name}", verbose)
        ax = axes[i]
        
        # Handle vector files
        if path.suffix.lower() in ['.geojson', '.json', '.shp']:
            # Get styling for vector layer
            style = _get_vector_style(name)
            
            # Read vector data
            gdf = gpd.read_file(path)
            
            # Plot the vector data
            gdf.plot(ax=ax, color=style["color"], edgecolor='black',
                    linewidth=0.5, alpha=style["fillOpacity"])
            
            ax.set_title(name.replace("_", " ").title())
            ax.axis("off")
            continue
        
        # Handle raster files
        with rasterio.open(path) as src:
            data = src.read(1)
        
        # Get styling for this layer
        style = _get_layer_style(name, data, i)
        
        if style["solid_color"] is not None:
            # Binary data - use solid color with transparency
            # Show color where data == 1, transparent where data == 0
            rgba_image = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.float32)
            r, g, b, a = style["solid_color"]
            rgba_image[:, :, 0] = r
            rgba_image[:, :, 1] = g
            rgba_image[:, :, 2] = b
            # Alpha: 1 where data == 1, 0 where data == 0
            rgba_image[:, :, 3] = (data == 1).astype(float) * style["alpha"]
            
            ax.imshow(rgba_image, origin='upper')
            
            # Add legend for binary data
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=(r, g, b, a), label='Present (1)')]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        else:
            # Continuous/categorical data - use colormap
            import matplotlib.cm as cm
            from matplotlib.colors import BoundaryNorm, ListedColormap
            
            vmin = style["vmin"] if style["vmin"] is not None else np.nanmin(data)
            vmax = style["vmax"] if style["vmax"] is not None else np.nanmax(data)
            
            # Check if this is categorical/discrete data (like fuel models)
            unique_vals = np.unique(data[data > 0])
            is_categorical = len(unique_vals) > 20  # Many unique values suggests categorical
            
            # Mask out zeros for better visualization
            masked_data = np.ma.masked_equal(data, 0)
            
            if is_categorical and "fbfm" in name.lower():
                import src.geospatial_harmonizer as _self
                from matplotlib.patches import Patch
                labels_map = _self.DISCOVERED_LABELS or {}
                all_cats = sorted([int(v) for v in unique_vals.tolist()])
                # Build colormap from ALL present values so no pixels are masked/white
                if "color_map" in style and style["color_map"]:
                    color_map = style["color_map"]
                    colors = [(color_map[c][0]/255, color_map[c][1]/255, color_map[c][2]/255)
                              if c in color_map else (0.7, 0.7, 0.7)
                              for c in all_cats]
                    cmap = ListedColormap(colors)
                    color_bounds = [c - 0.5 for c in all_cats] + [all_cats[-1] + 0.5]
                    norm = BoundaryNorm(color_bounds, cmap.N)
                    im = ax.imshow(masked_data, cmap=cmap, norm=norm, alpha=style["alpha"])
                    # Patch legend: only named categories, in order, 2 columns
                    named_cats = [c for c in all_cats if labels_map.get(c, str(c)) != str(c) and c in color_map]
                    legend_handles = [
                        Patch(facecolor=(color_map[c][0]/255, color_map[c][1]/255, color_map[c][2]/255),
                              label=labels_map[c])
                        for c in named_cats
                    ]
                else:
                    n_colors = len(all_cats)
                    cmap = cm.get_cmap("nipy_spectral", n_colors)
                    color_bounds = [c - 0.5 for c in all_cats] + [all_cats[-1] + 0.5]
                    norm = BoundaryNorm(color_bounds, cmap.N)
                    im = ax.imshow(masked_data, cmap=cmap, norm=norm, alpha=style["alpha"])
                    named_cats = [c for c in all_cats if labels_map.get(c, str(c)) != str(c)]
                    legend_handles = [
                        Patch(facecolor=cmap(norm(c)), label=labels_map[c])
                        for c in named_cats
                    ]
                ax.legend(
                    handles=legend_handles,
                    loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0,
                    fontsize=5.5, ncol=2, frameon=True,
                    title='Fuel Model', title_fontsize=6,
                    handlelength=1, handleheight=1, handletextpad=0.4, columnspacing=0.5,
                )
            else:
                cmap = cm.get_cmap(style["colormap"])
                # Mask out zeros for better visualization
                masked_data = np.ma.masked_equal(data, 0)
                im = ax.imshow(masked_data, cmap=cmap, vmin=vmin, vmax=vmax, alpha=style["alpha"])
                plt.colorbar(im, ax=ax, shrink=0.6)
        
        ax.set_title(name.replace("_", " ").title())
        ax.axis("off")

    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def _is_binary_data(data: np.ndarray) -> bool:
    """Check if data is binary (only contains 0 and 1)."""
    unique_vals = np.unique(data)
    if len(unique_vals) <= 2:
        # Check if values are 0 and 1 (or similar)
        non_zero_vals = [v for v in unique_vals if v != 0 and v != 0.0]
        if len(non_zero_vals) == 1 and non_zero_vals[0] == 1:
            return True
    return False


def _get_layer_style(name: str, data: np.ndarray, index: int) -> dict:
    """Get color scheme and styling for a layer based on its name and data type.
    
    Returns a dict with:
    - colormap: matplotlib colormap name
    - solid_color: RGB tuple for binary data (0-1 range)
    - alpha: transparency
    - vmin, vmax: for colormap normalization
    """
    name_lower = name.lower()
    is_binary = _is_binary_data(data)
    
    # Define distinct color schemes for different dataset types
    if is_binary:
        # Binary data - use solid colors
        if "burn" in name_lower or "mtbs" in name_lower or "fire" in name_lower:
            # Burned areas - use red
            return {
                "colormap": None,
                "solid_color": (1.0, 0.0, 0.0, 1.0),  # Red
                "alpha": 0.8,
                "vmin": None,
                "vmax": None,
            }
        elif "building" in name_lower or "footprint" in name_lower:
            # Buildings - use purple/blue
            return {
                "colormap": None,
                "solid_color": (0.4, 0.2, 0.8, 1.0),  # Purple
                "alpha": 0.8,
                "vmin": None,
                "vmax": None,
            }
        else:
            # Default binary - use red
            return {
                "colormap": None,
                "solid_color": (1.0, 0.0, 0.0, 1.0),  # Red
                "alpha": 0.8,
                "vmin": None,
                "vmax": None,
            }
    else:
        # Continuous/categorical data - use distinct colormaps
        if "fbfm" in name_lower or "fuel" in name_lower:
            # Fuel models - check if we have a discovered color map from source
            if DISCOVERED_COLOR_MAP:
                return {
                    "colormap": None,  # Will use custom color map
                    "color_map": DISCOVERED_COLOR_MAP,
                    "solid_color": None,
                    "alpha": 0.7,
                    "vmin": min(DISCOVERED_COLOR_MAP.keys()),
                    "vmax": max(DISCOVERED_COLOR_MAP.keys()),
                }
            else:
                # Fallback to default colormap
                return {
                    "colormap": "nipy_spectral",  # Good for many categories
                    "color_map": None,
                    "solid_color": None,
                    "alpha": 0.7,
                "vmin": data.min(),
                "vmax": data.max(),
            }
        elif "elevation" in name_lower or "dem" in name_lower:
            # Elevation - use terrain colormap
            return {
                "colormap": "terrain",
                "solid_color": None,
                "alpha": 0.7,
                "vmin": data.min(),
                "vmax": data.max(),
            }
        else:
            # Default continuous - use viridis
            return {
                "colormap": "viridis",
                "solid_color": None,
                "alpha": 0.7,
                "vmin": data.min(),
                "vmax": data.max(),
            }


def _get_vector_style(name: str) -> dict:
    """Get color scheme and styling for a vector layer based on its name.
    
    Returns a dict with:
    - color: Hex color string for the layer
    - weight: Line weight for boundaries
    - fillOpacity: Fill opacity for polygons
    """
    name_lower = name.lower()
    
    if "burn" in name_lower or "mtbs" in name_lower or "fire" in name_lower:
        # Burned areas - use red
        return {
            "color": "#FF0000",
            "weight": 2,
            "fillOpacity": 0.5,
        }
    elif "building" in name_lower or "footprint" in name_lower:
        # Buildings - use purple/blue
        return {
            "color": "#6600CC",
            "weight": 1,
            "fillOpacity": 0.7,
        }
    else:
        # Default - use blue
        return {
            "color": "#3186cc",
            "weight": 2,
            "fillOpacity": 0.5,
        }


def create_interactive_visualization(
    outputs: list[tuple[str, Path]],
    output_path: Path,
    target_extent: tuple[float, float, float, float],
    verbose: bool = True
) -> Path | None:
    """Create an interactive HTML map visualization with folium.
    
    Args:
        outputs: List of (name, path) tuples for each harmonized layer
        output_path: Path to save the HTML file
        target_extent: Bounding box (xmin, xmax, ymin, ymax) for the map view
        verbose: Print progress messages
        
    Returns:
        Path to the created HTML file, or None if creation failed
    """
    if not FOLIUM_AVAILABLE:
        _log("Folium not available, falling back to static visualization", verbose)
        return None
    
    try:
        return _create_interactive_visualization_impl(outputs, output_path, target_extent, verbose)
    except Exception as e:
        _log(f"Error creating interactive visualization: {e}", verbose)
        import traceback
        traceback.print_exc()
        return None


def _create_interactive_visualization_impl(
    outputs: list[tuple[str, Path]],
    output_path: Path,
    target_extent: tuple[float, float, float, float],
    verbose: bool = True
) -> Path:
    """Internal implementation of interactive visualization creation."""
    _log("Creating interactive HTML visualization", verbose)
    _load_color_map_from_output_dir(output_path)

    xmin, ymin, xmax, ymax = target_extent
    center_lat = (ymin + ymax) / 2
    center_lon = (xmin + xmax) / 2
    
    # Create base map with clean styling (CartoDB positron is clean and doesn't compete with data)
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles=None  # We'll add our own base layer
    )
    
    # Add CartoDB positron base layer (clean, light background)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        name="CartoDB Positron",
        max_zoom=19
    ).add_to(m)
    
    # Add each layer - handle raster and vector differently
    for i, (name, path) in enumerate(outputs):
        _log(f"  Processing layer {i+1}/{len(outputs)}: {name}", verbose)
        # Check if this is a vector file (GeoJSON)
        if path.suffix.lower() == '.geojson' or path.suffix.lower() == '.json':
            # Handle vector data as GeoJSON layer
            _log(f"  Adding vector layer: {name}", verbose)
            
            # Get styling for this layer
            style = _get_vector_style(name)
            
            # Read the GeoJSON
            gdf = gpd.read_file(path)

            # Convert datetime columns to strings to avoid JSON serialization issues
            for col in gdf.columns:
                if hasattr(gdf[col].dtype, 'kind') and gdf[col].dtype.kind == 'M':
                    gdf[col] = gdf[col].astype(str)

            file_size_mb = path.stat().st_size / 1e6
            if file_size_mb > 10:
                # Large vector: rasterize to a presence grid and show as an image overlay.
                # This preserves full spatial coverage without sampling or simplification.
                _log(f"  Rasterizing large vector ({file_size_mb:.0f} MB) for HTML display", verbose)
                bounds = (xmin, ymin, xmax, ymax)
                # Build a higher-resolution grid so small features (e.g. buildings) are visible
                aspect = (xmax - xmin) / (ymax - ymin)
                grid_h = 1500
                grid_w = max(1, int(grid_h * aspect))
                transform = from_bounds(xmin, ymin, xmax, ymax, grid_w, grid_h)
                presence = rasterize(
                    [(geom, 1) for geom in gdf.geometry if geom is not None],
                    out_shape=(grid_h, grid_w),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=True,
                )
                from PIL import Image as PILImage
                hex_c = style["color"].lstrip("#")
                r_val, g_val, b_val = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
                rgba = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)
                mask_px = presence == 1
                rgba[mask_px, 0] = r_val
                rgba[mask_px, 1] = g_val
                rgba[mask_px, 2] = b_val
                rgba[mask_px, 3] = int(style["fillOpacity"] * 255)
                img = PILImage.fromarray(rgba, mode='RGBA')
                buf = io.BytesIO()
                img.save(buf, format='PNG', optimize=True)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.getvalue()).decode('ascii')
                # opacity=1.0 because alpha is already baked into the RGBA pixel data
                folium.raster_layers.ImageOverlay(
                    image=f"data:image/png;base64,{img_base64}",
                    bounds=[[ymin, xmin], [ymax, xmax]],
                    name=name.replace("_", " ").title(),
                    opacity=1.0,
                    interactive=False,
                    zindex=i + 10,
                ).add_to(m)
            else:
                # Small vector: add as interactive GeoJSON layer
                geojson_str = gdf.to_json()
                folium.GeoJson(
                    geojson_str,
                    name=name.replace("_", " ").title(),
                    style_function=lambda x, color=style["color"]: {
                        'fillColor': color,
                        'color': color,
                        'weight': style["weight"],
                        'fillOpacity': style["fillOpacity"],
                    },
                    tooltip=name.replace("_", " ").title(),
                ).add_to(m)
        else:
            # Handle raster data as image overlay
            _log(f"  Adding raster layer: {name}", verbose)
            with rasterio.open(path) as src:
                data = src.read(1)
                # Use target extent for bounds to ensure full coverage
                bounds = (xmin, ymin, xmax, ymax)

                # Get styling for this layer
                style = _get_layer_style(name, data, i)

                # Build RGBA array directly with PIL — much smaller than matplotlib figure
                from PIL import Image as PILImage
                h, w = data.shape
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                alpha_val = int(style["alpha"] * 255)

                if style["solid_color"] is not None:
                    r, g, b, _ = style["solid_color"]
                    mask_px = data == 1
                    rgba[mask_px, 0] = int(r * 255)
                    rgba[mask_px, 1] = int(g * 255)
                    rgba[mask_px, 2] = int(b * 255)
                    rgba[mask_px, 3] = alpha_val
                else:
                    import matplotlib.cm as cm
                    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

                    vmin = style["vmin"] if style["vmin"] is not None else float(np.nanmin(data[data > 0]))
                    vmax = style["vmax"] if style["vmax"] is not None else float(np.nanmax(data))
                    unique_vals = np.unique(data[data > 0])
                    is_categorical = len(unique_vals) > 20

                    if is_categorical and "fbfm" in name.lower() and style.get("color_map"):
                        color_map = style["color_map"]
                        # Paint all non-zero pixels light grey first (catches unlabeled artifact codes)
                        artifact_px = (data > 0) & ~np.isin(data, list(color_map.keys()))
                        rgba[artifact_px, 0] = 180
                        rgba[artifact_px, 1] = 180
                        rgba[artifact_px, 2] = 180
                        rgba[artifact_px, 3] = alpha_val
                        # Then paint known categories with their proper colors
                        for cat_val, color in color_map.items():
                            px = data == cat_val
                            rgba[px, 0] = color[0]
                            rgba[px, 1] = color[1]
                            rgba[px, 2] = color[2]
                            rgba[px, 3] = alpha_val
                    else:
                        if is_categorical and "fbfm" in name.lower():
                            n_colors = len(unique_vals)
                            cmap = cm.get_cmap("nipy_spectral", n_colors)
                            norm = BoundaryNorm(
                                np.linspace(vmin - 0.5, vmax + 0.5, n_colors + 1), n_colors
                            )
                        else:
                            cmap = cm.get_cmap(style["colormap"])
                            norm = Normalize(vmin=vmin, vmax=vmax)

                        valid = data > 0
                        normed = norm(data[valid].astype(float))
                        colored = cmap(normed)  # (N, 4) float 0-1
                        rgba[valid, 0] = (colored[:, 0] * 255).astype(np.uint8)
                        rgba[valid, 1] = (colored[:, 1] * 255).astype(np.uint8)
                        rgba[valid, 2] = (colored[:, 2] * 255).astype(np.uint8)
                        rgba[valid, 3] = alpha_val

                # Downsample to max 800px on the long side for web display
                max_dim = 800
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                    img = PILImage.fromarray(rgba, mode='RGBA').resize(
                        (new_w, new_h), PILImage.LANCZOS
                    )
                else:
                    img = PILImage.fromarray(rgba, mode='RGBA')

                buf = io.BytesIO()
                img.save(buf, format='PNG', optimize=True)
                buf.seek(0)

                # Encode to base64
                img_base64 = base64.b64encode(buf.getvalue()).decode('ascii')
                
                # Create image overlay with proper lat/lon bounds using target extent
                # opacity=1.0 because alpha is already baked into the RGBA pixel data
                overlay = folium.raster_layers.ImageOverlay(
                    image=f"data:image/png;base64,{img_base64}",
                    bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                    name=name.replace("_", " ").title(),
                    opacity=1.0,
                    interactive=True,
                    zindex=i + 10  # Above base maps
                )
                overlay.add_to(m)
    
    # Add a rectangle to show the target extent
    folium.Rectangle(
        bounds=[[ymin, xmin], [ymax, xmax]],
        color="#3186cc",
        fill=True,
        fill_color="#3186cc",
        fill_opacity=0.1,
        name="Target Extent"
    ).add_to(m)
    
    # Add layer control with groups
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add fullscreen option
    folium.plugins.Fullscreen().add_to(m)
    
    # Add mouse position display
    folium.plugins.MousePosition().add_to(m)
    
    # Add a legend as a custom HTML element
    legend_html = '''
    <div style="position: fixed;
                bottom: 50px; left: 50px; width: 210px;
                max-height: calc(100vh - 120px); overflow-y: auto;
                border:2px solid grey; z-index:9999; font-size:12px;
                background-color:white; padding: 10px;
                border-radius: 5px;
                ">
    <b>Legend</b><br>
    '''
    
    for i, (name, path) in enumerate(outputs):
        # Check if this is a vector file
        if path.suffix.lower() == '.geojson' or path.suffix.lower() == '.json':
            style = _get_vector_style(name)
            legend_html += f'<div style="display:flex;align-items:center;margin:3px 0;"><i style="background:{style["color"]};width:12px;height:12px;display:inline-block;margin-right:5px;flex-shrink:0;"></i> {name.replace("_", " ").title()}</div>'
        else:
            with rasterio.open(path) as src:
                data = src.read(1)
            style = _get_layer_style(name, data, i)
            
            if style["solid_color"] is not None:
                # Binary data - show solid color
                r, g, b, a = style["solid_color"]
                hex_color = '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
                legend_html += f'<div style="display:flex;align-items:center;margin:3px 0;"><i style="background:{hex_color};width:12px;height:12px;display:inline-block;margin-right:5px;flex-shrink:0;"></i> {name.replace("_", " ").title()}</div>'
            else:
                # Categorical: show a compact labeled swatch list for values present in this layer
                import src.geospatial_harmonizer as _self
                color_map = style.get("color_map") or {}
                labels_map = _self.DISCOVERED_LABELS or {}
                unique_present = sorted(np.unique(data[data > 0]).tolist())
                legend_html += f'<div style="margin:4px 0 2px;font-weight:bold;">{name.replace("_", " ").title()}</div>'
                legend_html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:2px 8px;font-size:11px;">'
                # Only show values that have a real label or a real color (skip unnamed grey entries)
                named_present = [v for v in unique_present if v in color_map or labels_map.get(v, str(v)) != str(v)]
                for v in named_present:
                    label = labels_map.get(v, str(v))
                    if v in color_map:
                        r_c, g_c, b_c = color_map[v][0], color_map[v][1], color_map[v][2]
                        bg = f"rgb({r_c},{g_c},{b_c})"
                    else:
                        bg = "#aaa"
                    legend_html += (
                        f'<div style="display:flex;align-items:center;gap:4px;">'
                        f'<i style="background:{bg};width:12px;height:12px;display:inline-block;flex-shrink:0;border-radius:2px;"></i>'
                        f'{label}</div>'
                    )
                legend_html += '</div>'
    
    legend_html += '</div>'
    
    # Add the legend to the map
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save the map
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(output_path)
    
    _log(f"Interactive visualization saved to: {output_path}", verbose)
    
    return output_path


def run_harmonization_example(workflow: ExampleWorkflow) -> list[Path]:
    grid = build_grid_spec(
        target_crs=workflow.target_crs,
        target_extent=workflow.target_extent,
        target_resolution=workflow.target_resolution,
    )

    workflow.output_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[Path] = []
    viz_inputs: list[tuple[str, Path]] = []
    
    # Global color map discovered from source data
    global_color_map: dict | None = None

    with TemporaryDirectory(prefix=f"{workflow.name}_") as tmp:
        tmp_dir = Path(tmp)

        for dataset in workflow.datasets:
            dataset_dir = tmp_dir / dataset.name
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Check if output already exists BEFORE downloading
            if dataset.data_type == "vector" and not dataset.rasterize:
                output_path = workflow.output_dir / f"harmonized_{dataset.name}.geojson"
            else:
                output_path = workflow.output_dir / f"harmonized_{dataset.name}.tif"
            
            if output_path.exists():
                _log(f"Using existing harmonized file: {output_path.name}", workflow.verbose)
                output_files.append(output_path)
                viz_inputs.append((dataset.name, output_path))
                # Try to load color map from saved file if not already discovered
                if global_color_map is None and dataset.data_type == "raster":
                    color_map_path = workflow.output_dir / f"{dataset.name}_color_map.json"
                    if color_map_path.exists():
                        import json
                        with open(color_map_path) as f:
                            color_map_json = json.load(f)
                            # Convert string keys back to int and lists to tuples
                            color_map = {int(k): tuple(v) for k, v in color_map_json.items()}
                            global_color_map = color_map
                            # Update module-level global for visualization functions
                            import src.geospatial_harmonizer
                            src.geospatial_harmonizer.DISCOVERED_COLOR_MAP = color_map
                            _log(f"Loaded color map with {len(color_map)} categories from file", workflow.verbose)
                # Try to load labels from saved file
                labels_path = workflow.output_dir / f"{dataset.name}_labels.json"
                if labels_path.exists():
                    import json, src.geospatial_harmonizer
                    with open(labels_path) as f:
                        labels_json = json.load(f)
                    src.geospatial_harmonizer.DISCOVERED_LABELS = {int(k): v for k, v in labels_json.items()}
                    _log(f"Loaded {len(labels_json)} labels from file", workflow.verbose)
                continue

            # Check if URL is a WMS endpoint
            if dataset.is_wms and dataset.data_type == "raster":
                # Download from WMS with target extent
                source_file = download_wms_coverage(
                    wms_url=dataset.url,
                    layer=dataset.wms_layer,
                    bbox=workflow.target_extent,
                    crs=workflow.target_crs,
                    output_dir=dataset_dir,
                    output_filename=f"{dataset.name}.tif",
                    width=grid.width,
                    height=grid.height,
                    verbose=workflow.verbose,
                )
            # Check if URL is a WCS endpoint
            elif dataset.is_wcs and dataset.data_type == "raster":
                # Download from WCS with target extent
                source_file = download_wcs_coverage(
                    wcs_url=dataset.url,
                    layer=dataset.wcs_layer,
                    bbox=workflow.target_extent,
                    crs=workflow.target_crs,
                    output_dir=dataset_dir,
                    output_filename=f"{dataset.name}.tif",
                    width=grid.width,
                    height=grid.height,
                    verbose=workflow.verbose,
                )
            # Check if URL is an ArcGIS ImageServer endpoint
            elif "ImageServer" in dataset.url and dataset.data_type == "raster":
                # Download directly from ImageServer with target grid dimensions
                source_file = download_arcgis_image_server(
                    dataset.url,
                    dataset_dir,
                    bbox=grid.extent,
                    width=grid.width,
                    height=grid.height,
                    verbose=workflow.verbose,
                )
            else:
                # Standard download + extract workflow
                downloaded = download_file(dataset.url, dataset_dir, workflow.verbose)
                extracted_dir = extract_archive_if_needed(downloaded, dataset_dir, workflow.verbose)
                source_file = discover_dataset_file(extracted_dir, dataset.data_type)
                
                # Try to discover color map from extracted data
                if global_color_map is None and dataset.data_type == "raster":
                    color_map = discover_color_map(extracted_dir, dataset.name)
                    if color_map:
                        global_color_map = color_map
                        # Update module-level global for visualization functions
                        import src.geospatial_harmonizer
                        src.geospatial_harmonizer.DISCOVERED_COLOR_MAP = color_map
                        _log(f"Discovered color map with {len(color_map)} categories", workflow.verbose)
                        # Save color map to output directory for reference
                        color_map_path = workflow.output_dir / f"{dataset.name}_color_map.json"
                        # Rename old color_map.json if it exists
                        old_color_map_path = workflow.output_dir / "color_map.json"
                        if old_color_map_path.exists():
                            old_color_map_path.rename(color_map_path)
                        import json
                        # Convert tuples to lists for JSON serialization
                        color_map_json = {str(k): list(v) for k, v in color_map.items()}
                        with open(color_map_path, 'w') as f:
                            json.dump(color_map_json, f)
                        _log(f"Saved color map to {color_map_path.name}", workflow.verbose)

            # Download and cache labels CSV if provided
            if dataset.labels_url:
                import json as _json
                labels_path = workflow.output_dir / f"{dataset.name}_labels.json"
                if not labels_path.exists():
                    _log(f"Downloading labels from {dataset.labels_url}", workflow.verbose)
                    try:
                        with urllib.request.urlopen(dataset.labels_url) as resp:
                            csv_text = resp.read().decode("utf-8")
                        labels: dict[str, str] = {}
                        for line in csv_text.splitlines():
                            parts = line.strip().split(",")
                            if len(parts) >= 2:
                                try:
                                    int(parts[0])  # skip non-numeric rows (header / nodata)
                                    labels[parts[0]] = parts[1]
                                except ValueError:
                                    pass
                        with open(labels_path, "w") as f:
                            _json.dump(labels, f)
                        _log(f"Saved {len(labels)} labels to {labels_path.name}", workflow.verbose)
                    except Exception as e:
                        _log(f"Could not download labels: {e}", workflow.verbose)

            # Determine output path based on data type
            if dataset.data_type == "vector" and not dataset.rasterize:
                output_path = workflow.output_dir / f"harmonized_{dataset.name}.geojson"
            else:
                output_path = workflow.output_dir / f"harmonized_{dataset.name}.tif"

            if dataset.data_type == "raster":
                # If downloaded from ImageServer, it's already at target resolution
                # Still need to ensure CRS matches
                harmonize_raster(source_file, grid, output_path, workflow.verbose)
            elif dataset.data_type == "vector" and dataset.rasterize:
                rasterize_vector_to_grid(
                    source_file,
                    grid,
                    output_path,
                    burn_value=dataset.burn_value,
                    verbose=workflow.verbose,
                )
            elif dataset.data_type == "vector" and not dataset.rasterize:
                # Keep vector as vector - reproject and clip to extent
                harmonize_vector(
                    source_file,
                    grid,
                    output_path,
                    verbose=workflow.verbose,
                )
            else:
                raise NotImplementedError(
                    f"Unsupported data type: {dataset.data_type}"
                )

            output_files.append(output_path)
            viz_inputs.append((dataset.name, output_path))

    if workflow.create_visualization and output_files:
        viz_path = workflow.output_dir / "harmonized_visualization.png"
        create_visualization(viz_inputs, viz_path, workflow.verbose)
        
        # Also create interactive HTML visualization
        html_path = workflow.output_dir / "harmonized_visualization.html"
        create_interactive_visualization(
            viz_inputs,
            html_path,
            workflow.target_extent,
            workflow.verbose
        )

    return output_files