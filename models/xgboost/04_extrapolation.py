"""
XGBoost — Extrapolation to Unseen Region (Gapeau Basin)

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
RASTER_DIR = RAW_DATA_DIR / "rasters_gapeau"
OUTPUT_DIR = RESULTS_DIR / "xgboost"
OUTPUT_PATH = OUTPUT_DIR / "Gapeau_XGB_4dtsts_extrapolated.tif"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = ["Discharge_N", "Dist2stream_N", "HAND", "River_Slope"]


def find_feature_files(raster_dir, feature_names):
    """Find raster files matching feature names in order."""
    all_tifs = sorted(glob.glob(str(raster_dir / "*.tif")))
    selected = []
    for feat in feature_names:
        matches = [fp for fp in all_tifs if feat in os.path.basename(fp)]
        if not matches:
            raise FileNotFoundError(
                f"No TIFF found for feature '{feat}' in {raster_dir}"
            )
        selected.append(matches[0])
    return selected


def extrapolate(model, feature_paths, output_path):
    """Apply trained model to new region rasters block-by-block."""
    srcs = [rasterio.open(fp) for fp in feature_paths]
    meta = srcs[0].meta.copy()
    meta.update(count=1, dtype="float32", compress="lzw")

    n_blocks = sum(1 for _ in srcs[0].block_windows(1))
    print(f"  Processing {n_blocks} blocks...")

    with rasterio.open(output_path, "w", **meta) as dst:
        for block_idx, (_, window) in enumerate(srcs[0].block_windows(1)):
            bands = [src.read(1, window=window) for src in srcs]
            arr = np.stack(bands, axis=0)
            n_feats, H, W = arr.shape
            pix = arr.reshape(n_feats, -1).T

            # Nodata mask
            nodata_masks = []
            for band, src in zip(bands, srcs):
                if src.nodata is not None:
                    nodata_masks.append(band == src.nodata)
                else:
                    nodata_masks.append(np.isnan(band))

            mask_flat = np.any(
                np.stack(nodata_masks, axis=0), axis=0
            ).reshape(-1)
            mask_flat |= ~np.all(np.isfinite(pix), axis=1)

            nodata_val = srcs[0].nodata if srcs[0].nodata is not None else -9999
            probs = np.full(pix.shape[0], nodata_val, dtype=np.float32)

            if (~mask_flat).any():
                probs[~mask_flat] = model.predict_proba(
                    pix[~mask_flat]
                )[:, 1]

            out = probs.reshape(H, W).astype("float32")
            dst.write(out, 1, window=window)

            if (block_idx + 1) % 100 == 0:
                print(f"    Block {block_idx + 1}/{n_blocks}")

    for src in srcs:
        src.close()

    print(f"  ✅ Written to: {output_path}")


def main():
    print("=" * 60)
    print("XGBOOST — EXTRAPOLATION TO GAPEAU BASIN")
    print("=" * 60)

    print("\n1. Loading trained model...")
    model = joblib.load(MODEL_PATH)
    print(f"   Loaded from: {MODEL_PATH}")

    print("\n2. Finding feature rasters for Gapeau...")
    feature_files = find_feature_files(RASTER_DIR, FEATURE_NAMES)
    for i, (feat, fp) in enumerate(zip(FEATURE_NAMES, feature_files)):
        print(f"   [{i}] {feat:20s} → {os.path.basename(fp)}")

    print("\n3. Extrapolating predictions...")
    extrapolate(model, feature_files, OUTPUT_PATH)

    print("\nDone!")


if __name__ == "__main__":
    main()
