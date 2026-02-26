"""
Evaluate simulated vs. observed flood extents by sub-catchment.

For each simulation raster:
  1. Threshold water level → binary flood/no-flood
  2. Build contingency codes:
       0 = correct rejection
       1 = miss
       2 = false alarm
       3 = hit
  3. Count pixels per code inside each catchment polygon
  4. Compute CSI, TSI, Bias, FAR, POD
  5. Write out the contingency-code raster and a shapefile of statistics

Author: Aman Arora
Created: 2025-04-10
Inspired by: Nabil, Juliette, and Pierre
"""

import os
import time
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask


def read_and_align(ref_fp, sim_fp):
    """
    Read reference and simulation rasters.
    Resample reference onto simulation grid if grids differ.

    Parameters
    ----------
    ref_fp : str or Path
        Path to reference (ground truth) raster.
    sim_fp : str or Path
        Path to simulation (predicted) raster.

    Returns
    -------
    ref_data : np.ndarray
        Aligned reference raster array.
    sim_data : np.ndarray
        Simulation raster array.
    sim_meta : dict
        Rasterio profile/metadata of simulation raster.
    """
    with rasterio.open(sim_fp) as sim_src:
        sim_data = sim_src.read(1)
        sim_meta = sim_src.profile

    with rasterio.open(ref_fp) as ref_src:
        ref_data = ref_src.read(1)
        ref_meta = ref_src.profile

    # If extents/CRS/shape differ, resample reference onto simulation grid
    if (ref_meta['crs'] != sim_meta['crs'] or
        ref_meta['transform'] != sim_meta['transform'] or
        ref_meta['width'] != sim_meta['width'] or
        ref_meta['height'] != sim_meta['height']):

        aligned_ref = np.empty_like(sim_data, dtype=ref_data.dtype)
        reproject(
            source=ref_data,
            destination=aligned_ref,
            src_transform=ref_meta['transform'],
            src_crs=ref_meta['crs'],
            dst_transform=sim_meta['transform'],
            dst_crs=sim_meta['crs'],
            dst_res=(sim_meta['transform'][0], -sim_meta['transform'][4]),
            resampling=Resampling.nearest
        )
        ref_data = aligned_ref

    return ref_data, sim_data, sim_meta


def contingency_raster(ref_arr, sim_arr, thr=0.01):
    """
    Build contingency code raster.

    Codes:
        0 = correct rejection (both no flood)
        1 = miss (ref=flood, sim=no flood)
        2 = false alarm (ref=no flood, sim=flood)
        3 = hit (both flood)

    Parameters
    ----------
    ref_arr : np.ndarray
        Reference binary flood raster.
    sim_arr : np.ndarray
        Simulated/predicted flood raster.
    thr : float
        Threshold to binarize simulation raster.

    Returns
    -------
    np.ndarray
        Contingency code raster (values 0-3).
    """
    sim_mask = np.where(sim_arr < thr, 0, 2)
    ref_mask = np.where(np.isnan(ref_arr) | (ref_arr <= 0), 0, 1)
    return sim_mask + ref_mask


def count_per_catchment(cont_code, meta, basins_gdf, id_field="BV_ID"):
    """
    Count contingency pixels per catchment polygon.

    Parameters
    ----------
    cont_code : np.ndarray
        Contingency code raster (0-3).
    meta : dict
        Rasterio profile metadata.
    basins_gdf : GeoDataFrame
        Catchment polygons.
    id_field : str
        Column name for catchment ID.

    Returns
    -------
    pd.DataFrame
        Pixel counts [cr, miss, fa, hit] per catchment.
    """
    counts = []
    transform = meta['transform']

    for _, row in basins_gdf.iterrows():
        geom = row.geometry
        mask = geometry_mask(
            [geom],
            transform=transform,
            invert=True,
            out_shape=(meta['height'], meta['width'])
        )
        vals = cont_code[mask]
        unique, freq = np.unique(vals, return_counts=True)
        d = dict(zip(unique, freq))
        counts.append({
            id_field: row[id_field],
            'cr':   d.get(0, 0),
            'miss': d.get(1, 0),
            'fa':   d.get(2, 0),
            'hit':  d.get(3, 0)
        })

    return pd.DataFrame(counts)


def compute_scores(df):
    """
    Compute flood evaluation metrics from contingency counts.

    Metrics:
        CSI  = hit / (hit + miss + fa)
        TSI  = (miss + fa) / (hit + miss)
        Bias = (hit + fa) / (miss + hit)
        FAR  = fa / (fa + hit)
        POD  = hit / (miss + hit)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: hit, miss, fa, cr.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added metric columns.
    """
    df = df.copy()
    df['CSI']  = df['hit'] / (df['hit'] + df['miss'] + df['fa'])
    df['TSI']  = (df['miss'] + df['fa']) / (df['hit'] + df['miss'])
    df['Bias'] = (df['hit'] + df['fa']) / (df['miss'] + df['hit'])
    df['FAR']  = df['fa'] / (df['fa'] + df['hit'])
    df['POD']  = df['hit'] / (df['miss'] + df['hit'])
    return df


def evaluate_flood_extent(basin_shp, ref_raster, sim_pattern,
                          output_dir, basin_name="Basin",
                          thr=0.01, id_field="BV_ID"):
    """
    Full evaluation pipeline for one basin.

    Parameters
    ----------
    basin_shp : str or Path
        Path to catchment shapefile.
    ref_raster : str or Path
        Path to reference (ground truth) raster.
    sim_pattern : str or Path
        Glob pattern to find simulation rasters.
    output_dir : str or Path
        Directory to save outputs.
    basin_name : str
        Name of the basin (for logging).
    thr : float
        Threshold for binarizing simulation raster.
    id_field : str
        Column name for catchment ID in shapefile.

    Returns
    -------
    dict
        Dictionary of {simulation_name: scores_dataframe}.
    """
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    # Read catchment polygons
    basins_gdf = gpd.read_file(basin_shp)
    ref_fp = str(ref_raster)

    # Find simulation rasters
    sims = glob.glob(str(sim_pattern))
    all_results = {}

    for sim_fp in sims:
        sim_name = os.path.splitext(os.path.basename(sim_fp))[0]
        print(f"Processing {sim_name} ...")

        # Load and align
        ref_arr, sim_arr, meta = read_and_align(ref_fp, sim_fp)

        # Build contingency raster
        cont_code = contingency_raster(ref_arr, sim_arr, thr)

        # Count per catchment
        df_counts = count_per_catchment(cont_code, meta, basins_gdf, id_field)
        df_counts = compute_scores(df_counts)

        # Join back to GeoDataFrame
        basins_out = basins_gdf.merge(df_counts, on=id_field)

        # Write contingency raster
        out_tif = os.path.join(output_dir, f"{basin_name}_cont_{sim_name}.tif")
        with rasterio.open(out_tif, "w", **meta) as dst:
            dst.write(cont_code.astype(np.uint8), 1)
        print(f"  → Wrote raster: {out_tif}")

        # Write shapefile with stats
        out_shp = os.path.join(output_dir, f"{basin_name}_scores_{sim_name}.shp")
        basins_out.to_file(out_shp, driver="ESRI Shapefile")
        print(f"  → Wrote shapefile: {out_shp}\n")

        all_results[sim_name] = df_counts

    print(f"Done for {basin_name} in {time.time() - t0:.1f}s\n")
    return all_results
