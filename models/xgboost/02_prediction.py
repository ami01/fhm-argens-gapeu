"""
XGBoost — Raster Prediction

Steps:
  1. Load trained XGBoost model
  2. Read input feature rasters
  3. Predict flood probability per pixel (block-wise)
  4. Write prediction raster

Author: Aman Arora
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import os
import glob
import numpy as np
import joblib
import rasterio

from config import RAW_DATA_DIR, RESULTS_DIR

# ── Paths ───────────────────────────────────────────────
MODEL_PATH = RESULTS_DIR / "xgboost" / "xgb_model_4dtsts.pkl"
RASTER_DIR = RAW_DATA_DIR / "rasters"
OUTPUT_DIR = RESULTS_DIR / "xgboost"
OUTPUT_PATH = OUTPUT_DIR / "XGB_4dtsts_prediction.tif"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature rasters (must match training feature order) ─
SELECTED_RASTERS = ['Discharge_N', 'Dist2stream_N', 'HAND', 'River_Slope']


def load_feature_rasters(raster_dir, feature_names):
    """
    Find and open feature rasters matching the given names.

    Parameters
    ----------
    raster_dir : Path
        Directory containing .tif raster files.
    feature_names : list
        Keywords to match in filenames.

    Returns
    -------
    list
        List of open rasterio dataset objects.
    list
        List of matched filenames (for verification).
    """
    all_files = sorted(glob.glob(str(raster_dir / "*.tif")))
    selected = [
        f for f in all_files
        if any(name in os.path.basename(f) for name in feature_names)
    ]

    if len(selected) != len(feature_names):
        raise ValueError(
            f"Expected {len(feature_names)} rasters, found {len(selected)}.\n"
            f"Looking for: {feature_names}\n"
            f"Found: {[os.path.basename(f) for f in selected]}"
        )

    filenames = [os.path.basename(f) for f in selected]
    print(f"Found {len(selected)} feature rasters:")
    for f in filenames:
        print(f"  {f}")

    return [rasterio.open(f) for f in selected], filenames


def predict_to_raster(model, rasters, output_path):
    """
    Apply trained model to rasters block-by-block and write output.

    Parameters
    ----------
    model : sklearn/xgboost estimator
        Trained classifier with predict_proba method.
    rasters : list
        List of open rasterio datasets (one per feature).
    output_path : Path
        Output raster path.
    """
    ref = rasters[0]
    meta = ref.meta.copy()
    meta.update({
        "count": 1,
        "dtype": "float32"
    })

    n_blocks = sum(1 for _ in ref.block_windows(1))
    print(f"  Processing {n_blocks} blocks...")

    with rasterio.open(output_path, 'w', **meta) as dst:
        for block_idx, (ji, window) in enumerate(ref.block_windows(1)):
            # Read same window from each raster
            window_data = [src.read(1, window=window) for src in rasters]

            # Stack: (n_features, height, width)
            stacked = np.stack(window_data, axis=0)
            height, width = stacked.shape[1], stacked.shape[2]

            # Reshape to (n_pixels, n_features)
            data_2d = stacked.reshape(len(rasters), -1).T

            # Handle NaN/Inf values
            mask_valid = np.all(np.isfinite(data_2d), axis=1)
            predicted_probs = np.zeros(data_2d.shape[0], dtype=np.float32)

            if mask_valid.any():
                predicted_probs[mask_valid] = model.predict_proba(
                    data_2d[mask_valid]
                )[:, 1]

            # Reshape back and write
            predicted_window = predicted_probs.reshape(height, width)
            dst.write(predicted_window.astype('float32'), 1, window=window)

            # Progress
            if (block_idx + 1) % 100 == 0:
                print(f"    Block {block_idx + 1}/{n_blocks}")

    print(f"  ✅ Prediction raster written to: {output_path}")


def main():
    print("=" * 60)
    print("XGBOOST — RASTER PREDICTION")
    print("=" * 60)

    # 1. Load model
    print("\n1. Loading trained model...")
    model = joblib.load(MODEL_PATH)
    print(f"   Loaded from: {MODEL_PATH}")

    # 2. Load feature rasters
    print("\n2. Loading feature rasters...")
    rasters, filenames = load_feature_rasters(RASTER_DIR, SELECTED_RASTERS)

    # 3. Verify feature order matches training
    print("\n   Feature order verification:")
    for i, (feat, fname) in enumerate(zip(SELECTED_RASTERS, filenames)):
        print(f"   [{i}] Training: {feat:20s} → Raster: {fname}")

    # 4. Predict
    print("\n3. Generating predictions...")
    predict_to_raster(model, rasters, OUTPUT_PATH)

    # 5. Close rasters
    for r in rasters:
        r.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
