"""
Shared raster I/O and processing functions.

Used by:
  - models/random_forest/02_prediction.py
  - models/random_forest/04_extrapolation.py
  - models/xgboost/02_prediction.py
  - models/xgboost/04_extrapolation.py
  - models/*/05_extrapolation_validation.py
  - utils/csi_evaluation.py

Author: Aman Arora
"""

import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt


def predict_to_raster(model, rasters, output_path, compress="lzw"):
    """
    Apply trained model to rasters block-by-block and write output.
    Handles nodata and infinite values.

    Parameters
    ----------
    model : sklearn/xgboost estimator
        Trained classifier with predict_proba method.
    rasters : list
        List of open rasterio datasets (one per feature).
    output_path : str or Path
        Output raster path.
    compress : str
        Compression method for output.

    Example
    -------
    >>> srcs = [rasterio.open(f) for f in feature_files]
    >>> predict_to_raster(model, srcs, "output.tif")
    """
    ref = rasters[0]
    meta = ref.meta.copy()
    meta.update(count=1, dtype="float32", compress=compress)

    n_blocks = sum(1 for _ in ref.block_windows(1))
    print(f"  Processing {n_blocks} blocks...")

    with rasterio.open(str(output_path), "w", **meta) as dst:
        for block_idx, (_, window) in enumerate(ref.block_windows(1)):
            # Read each feature band
            bands = [src.read(1, window=window) for src in rasters]
            stacked = np.stack(bands, axis=0)
            n_feats, H, W = stacked.shape

            # Reshape to (n_pixels, n_features)
            pix = stacked.reshape(n_feats, -1).T

            # Create nodata mask
            nodata_masks = []
            for band, src in zip(bands, rasters):
                if src.nodata is not None:
                    nodata_masks.append(band == src.nodata)
                else:
                    nodata_masks.append(np.isnan(band))

            mask_flat = np.any(
                np.stack(nodata_masks, axis=0), axis=0
            ).reshape(-1)

            # Also mask infinite values
            mask_flat |= ~np.all(np.isfinite(pix), axis=1)

            # Initialize with nodata
            nodata_val = ref.nodata if ref.nodata is not None else -9999
            probs = np.full(pix.shape[0], nodata_val, dtype=np.float32)

            # Predict only valid pixels
            if (~mask_flat).any():
                probs[~mask_flat] = model.predict_proba(
                    pix[~mask_flat]
                )[:, 1]

            # Write
            out = probs.reshape(H, W).astype("float32")
            dst.write(out, 1, window=window)

            if (block_idx + 1) % 100 == 0:
                print(f"    Block {block_idx + 1}/{n_blocks}")

    print(f"  ✅ Written to: {output_path}")


def sample_raster_at_points(raster_path, gdf):
    """
    Sample raster values at point locations from a GeoDataFrame.

    Parameters
    ----------
    raster_path : str or Path
        Path to raster file.
    gdf : GeoDataFrame
        Point geometries.

    Returns
    -------
    np.ndarray
        Sampled values at point locations.

    Example
    -------
    >>> gdf = gpd.read_file("points.shp")
    >>> vals = sample_raster_at_points("prediction.tif", gdf)
    """
    with rasterio.open(str(raster_path)) as src:
        coords = [(geom.x, geom.y) for geom in gdf.geometry]
        values = [val[0] for val in src.sample(coords)]
    return np.array(values, dtype=np.float64)


def plot_roc_curve(fpr, tpr, auc, model_name,
                   color='blue', save_path=None, dpi=300):
    """
    Plot and optionally save ROC curve.

    Parameters
    ----------
    fpr : array
        False positive rates.
    tpr : array
        True positive rates.
    auc : float
        Area under curve.
    model_name : str
        Model name for legend and title.
    color : str
        Line color.
    save_path : Path, optional
        Path to save figure.
    dpi : int
        Figure resolution.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color=color, linewidth=2,
             label=f'{model_name} (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve — {model_name}', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=dpi, bbox_inches='tight')
        print(f"  → Saved: {save_path}")

    plt.show()


def clean_predictions(y_true, y_pred_prob):
    """
    Remove invalid (NaN, Inf) predictions and corresponding truth values.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred_prob : array-like
        Predicted probabilities.

    Returns
    -------
    y_true_clean : np.ndarray
        Cleaned ground truth.
    y_pred_clean : np.ndarray
        Cleaned predictions.
    n_removed : int
        Number of removed samples.
    """
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob, dtype=np.float64)

    valid_mask = np.isfinite(y_pred_prob)
    n_removed = (~valid_mask).sum()

    if n_removed > 0:
        print(f"  ⚠️ Removing {n_removed} invalid samples")

    return y_true[valid_mask], y_pred_prob[valid_mask], n_removed
