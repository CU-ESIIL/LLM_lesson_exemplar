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


def _create_binary_mask(data: np.ndarray) -> np.ndarray:
    """Create a mask for binary data (0s and 1s), returning only 1s as visible."""
    unique_vals = np.unique(data)
    if len(unique_vals) <= 2 and all(v in [0, 1, data.dtype.type(0), data.dtype.type(1)] for v in unique_vals):
        return np.ma.masked_equal(data, 0)
    return data


def create_visualization(outputs: list[tuple[str, Path]], output_path: Path, verbose: bool = True) -> Path:
    """Create a static PNG visualization with subplots for each layer."""
    _log("Creating visualization", verbose)

    n = len(outputs)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, (name, path) in enumerate(outputs):
        with rasterio.open(path) as src:
            data = src.read(1)
        
        # Get styling for this layer
        style = _get_layer_style(name, data, i)
        
        ax = axes[i]
        
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
                # For fuel models: create discrete colorbar with boundaries
                n_colors = len(unique_vals)
                # Create discrete colormap
                cmap = cm.get_cmap("nipy_spectral", n_colors)
                # Create boundaries for discrete colorbar
                bounds = np.linspace(vmin - 0.5, vmax + 0.5, n_colors + 1)
                norm = BoundaryNorm(bounds, cmap.N)
                im = ax.imshow(masked_data, cmap=cmap, norm=norm, alpha=style["alpha"])
                cbar = plt.colorbar(im, ax=ax, shrink=0.6, ticks=unique_vals[::10], orientation='horizontal', pad=0.05)
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
            # Fuel models - use a colormap that handles many categories
            return {
                "colormap": "nipy_spectral",  # Good for many categories
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


def create_interactive_visualization(
    outputs: list[tuple[str, Path]],
    output_path: Path,
    target_extent: tuple[float, float, float, float],
    verbose: bool = True
) -> Path:
    """Create an interactive HTML map visualization with folium.
    
    Args:
        outputs: List of (name, path) tuples for each harmonized layer
        output_path: Path to save the HTML file
        target_extent: Bounding box (xmin, xmax, ymin, ymax) for the map view
        verbose: Print progress messages
        
    Returns:
        Path to the created HTML file
    """
    if not FOLIUM_AVAILABLE:
        _log("Folium not available, falling back to static visualization", verbose)
        return None
    
    _log("Creating interactive HTML visualization", verbose)
    
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
    
    # Add CartoDB dark matter as alternative
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        name="CartoDB Dark",
        max_zoom=19
    ).add_to(m)
    
    # Add OpenStreetMap as another alternative
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        max_zoom=19
    ).add_to(m)
    
    # Add each layer as an image overlay
    for i, (name, path) in enumerate(outputs):
        with rasterio.open(path) as src:
            data = src.read(1)
            bounds = src.bounds
            
            # Get styling for this layer
            style = _get_layer_style(name, data, i)
            
            # Create a temporary PNG for the overlay
            fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
            
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
                
                ax.imshow(rgba_image, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
                         origin='upper')
            else:
                # Continuous/categorical data - use colormap
                import matplotlib.cm as cm
                from matplotlib.colors import BoundaryNorm
                
                vmin = style["vmin"] if style["vmin"] is not None else np.nanmin(data)
                vmax = style["vmax"] if style["vmax"] is not None else np.nanmax(data)
                
                # Check if this is categorical/discrete data (like fuel models)
                unique_vals = np.unique(data[data > 0])
                is_categorical = len(unique_vals) > 20  # Many unique values suggests categorical
                
                # Mask out zeros for better visualization
                masked_data = np.ma.masked_equal(data, 0)
                
                if is_categorical and "fbfm" in name.lower():
                    # For fuel models: create discrete colormap
                    n_colors = len(unique_vals)
                    cmap = cm.get_cmap("nipy_spectral", n_colors)
                    color_bounds = np.linspace(vmin - 0.5, vmax + 0.5, n_colors + 1)
                    norm = BoundaryNorm(color_bounds, cmap.N)
                    im = ax.imshow(masked_data,
                                  extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
                                  origin='upper',
                                  cmap=cmap,
                                  norm=norm,
                                  alpha=style["alpha"])
                else:
                    cmap = cm.get_cmap(style["colormap"])
                    im = ax.imshow(masked_data,
                                  extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
                                  origin='upper',
                              alpha=style["alpha"],
                              cmap=cmap,
                              vmin=vmin,
                              vmax=vmax)
                # Don't add colorbar here - it makes the image too large for overlay
                # We'll add an external legend to the HTML instead
            
            ax.set_xlim(bounds.left, bounds.right)
            ax.set_ylim(bounds.bottom, bounds.top)
            # Don't set title here - it will overlap in the HTML overlay
            # The layer name in folium.LayerControl serves as the title
            ax.axis("off")
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', transparent=True)
            buf.seek(0)
            plt.close()
            
            # Encode to base64 - remove any whitespace/newlines
            img_base64 = base64.b64encode(buf.getvalue()).decode('ascii')
            
            # Create image overlay with proper lat/lon bounds
            # Use branca colormap for proper rendering
            overlay = folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{img_base64}",
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                name=name.replace("_", " ").title(),
                opacity=style["alpha"],
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
                bottom: 50px; left: 50px; width: 180px; height: auto;
                border:2px solid grey; z-index:9999; font-size:12px;
                background-color:white; padding: 10px;
                border-radius: 5px;
                ">
    <b>Legend</b><br>
    '''
    
    for i, (name, path) in enumerate(outputs):
        with rasterio.open(path) as src:
            data = src.read(1)
        style = _get_layer_style(name, data, i)
        
        if style["solid_color"] is not None:
            # Binary data - show solid color
            r, g, b, a = style["solid_color"]
            hex_color = '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
            legend_html += f'<div style="display:flex;align-items:center;margin:3px 0;"><i style="background:{hex_color};width:12px;height:12px;display:inline-block;margin-right:5px;flex-shrink:0;"></i> {name.replace("_", " ").title()}</div>'
        else:
            # Continuous/categorical - show colormap name on same line as gradient
            legend_html += f'<div style="display:flex;align-items:center;margin:3px 0;"><span style="margin-right:5px;white-space:nowrap;">{name.replace("_", " ").title()}</span><i style="background:linear-gradient(to right, blue, cyan, green, yellow, red);width:100px;height:12px;display:inline-block;flex-grow:1;"></i></div>'
    
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

    with TemporaryDirectory(prefix=f"{workflow.name}_") as tmp:
        tmp_dir = Path(tmp)

        for dataset in workflow.datasets:
            dataset_dir = tmp_dir / dataset.name
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Check if output already exists BEFORE downloading
            output_path = workflow.output_dir / f"harmonized_{dataset.name}.tif"
            if output_path.exists():
                _log(f"Using existing harmonized file: {output_path.name}", workflow.verbose)
                output_files.append(output_path)
                viz_inputs.append((dataset.name, output_path))
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
            else:
                raise NotImplementedError(
                    "Vector outputs are currently supported only through rasterization."
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