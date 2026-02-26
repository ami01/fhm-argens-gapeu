"""
Shared data loading and preparation functions.

Used by:
  - models/random_forest/01_training.py
  - models/xgboost/01_training.py
  - models/random_forest/04_extrapolation.py
  - models/xgboost/04_extrapolation.py

Author: Aman Arora
"""

import os
import glob
import numpy as np
import pandas as pd
import rasterio


def load_and_prepare_data(filepath, features, target):
    """
    Load training CSV and prepare feature matrix and target.

    Parameters
    ----------
    filepath : str or Path
        Path to training CSV file.
    features : list of str
        List of feature column names to select.
    target : str
        Target column name (e.g., 'CID').

    Returns
    -------
    X : pd.DataFrame
        Cleaned feature matrix (NaN imputed, constants removed).
    y : pd.Series
        Target variable.

    Example
    -------
    >>> X, y = load_and_prepare_data("data/raw/train.csv",
    ...     ["HAND", "Discharge_N"], "CID")
    """
    df = pd.read_csv(filepath)

    # Verify columns exist
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns not found in data: {missing}\n"
            f"Available: {df.columns.tolist()}"
        )

    # Select features
    X = df[features].copy()

    # Remove constant columns
    constant_cols = X.columns[X.nunique() <= 1].tolist()
    if constant_cols:
        print(f"  ⚠️ Removing constant columns: {constant_cols}")
        X = X.drop(columns=constant_cols)

    # Replace infinite values with NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Impute missing values with median
    n_missing = X.isna().sum().sum()
    if n_missing > 0:
        print(f"  ⚠️ Imputing {n_missing} missing values with median")
        X = X.fillna(X.median())

    y = df[target]

    return X, y


def find_feature_rasters(raster_dir, feature_names):
    """
    Find and validate raster files matching feature names.

    Parameters
    ----------
    raster_dir : str or Path
        Directory containing .tif raster files.
    feature_names : list of str
        Keywords to match in filenames (in training order).

    Returns
    -------
    selected : list of str
        Ordered list of matched file paths.

    Raises
    ------
    FileNotFoundError
        If any feature raster is not found.

    Example
    -------
    >>> files = find_feature_rasters("data/raw/rasters",
    ...     ["HAND", "Discharge_N"])
    """
    all_tifs = sorted(glob.glob(os.path.join(str(raster_dir), "*.tif")))
    selected = []

    for feat in feature_names:
        matches = [fp for fp in all_tifs
                   if feat in os.path.basename(fp)]
        if not matches:
            raise FileNotFoundError(
                f"No TIFF found for feature '{feat}' in {raster_dir}\n"
                f"Available files: {[os.path.basename(f) for f in all_tifs]}"
            )
        if len(matches) > 1:
            print(f"  ⚠️ Multiple matches for '{feat}', using: {os.path.basename(matches[0])}")
        selected.append(matches[0])

    print(f"  Found {len(selected)} feature rasters:")
    for i, (feat, fp) in enumerate(zip(feature_names, selected)):
        print(f"    [{i}] {feat:20s} → {os.path.basename(fp)}")

    return selected


def load_validation_data(shapefile_path, target_col="CID"):
    """
    Load validation shapefile and verify target column exists.

    Parameters
    ----------
    shapefile_path : str or Path
        Path to validation shapefile.
    target_col : str
        Name of the ground truth column.

    Returns
    -------
    gdf : GeoDataFrame
        Loaded validation data.

    Raises
    ------
    ValueError
        If target column is missing.
    """
    import geopandas as gpd

    gdf = gpd.read_file(shapefile_path)
    print(f"  Loaded {len(gdf)} validation points")

    if target_col not in gdf.columns:
        raise ValueError(
            f"Missing '{target_col}' column.\n"
            f"Available: {gdf.columns.tolist()}"
        )

    return gdf
