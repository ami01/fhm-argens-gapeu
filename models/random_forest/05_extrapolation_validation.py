"""
Random Forest — Extrapolation Validation (Gapeau Basin)

Validates RF predictions extrapolated to unseen Gapeau basin.

Author: Aman Arora
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from config import (
    RAW_DATA_DIR, RESULTS_DIR,
    FIGURES_DIR, CLASSIFICATION_THRESHOLD
)
from utils.data_loader import load_validation_data
from utils.raster_utils import (
    sample_raster_at_points,
    plot_roc_curve,
    clean_predictions
)
from utils.metrics import compute_all_metrics, print_metrics, save_metrics

# ── Paths ───────────────────────────────────────────────
PREDICTION_RASTER = RESULTS_DIR / "random_forest" / "Gapeau_RF_4dtsts_extrapolated.tif"
VALIDATION_SHP    = RAW_DATA_DIR / "inventory" / "Validation_FnF_Gapeau.shp"
OUTPUT_DIR        = RESULTS_DIR / "random_forest"
TARGET_COL        = "CID"
THRESHOLD         = CLASSIFICATION_THRESHOLD


def main():
    print("=" * 60)
    print("RANDOM FOREST — EXTRAPOLATION VALIDATION (GAPEAU)")
    print("=" * 60)

    # 1. Load validation data
    print("\n1. Loading Gapeau validation data...")
    val_gdf = load_validation_data(VALIDATION_SHP, TARGET_COL)

    # 2. Sample predictions
    print("\n2. Sampling extrapolated prediction raster...")
    y_pred_prob = sample_raster_at_points(PREDICTION_RASTER, val_gdf)
    y_true = val_gdf[TARGET_COL].values

    # 3. Clean
    print("\n3. Cleaning predictions...")
    y_true, y_pred_prob, n_removed = clean_predictions(y_true, y_pred_prob)

    # 4. Compute metrics
    print("\n4. Computing metrics...")
    metrics = compute_all_metrics(y_true, y_pred_prob, THRESHOLD)
    print_metrics(metrics, "Random Forest — Gapeau Extrapolation")

    # 5. Plot ROC
    print("\n5. Plotting ROC curve...")
    plot_roc_curve(
        fpr=metrics['fpr'],
        tpr=metrics['tpr'],
        auc=metrics['auc'],
        model_name="Random Forest — Gapeau Extrapolation",
        color='blue',
        save_path=FIGURES_DIR / "rf_extrapolation_roc.png"
    )

    # 6. Save results
    print("\n6. Saving results...")
    save_metrics(
        metrics=metrics,
        y_true=y_true,
        y_pred_prob=y_pred_prob,
        output_dir=OUTPUT_DIR,
        prefix="rf",
        stage="extrapolation"
    )

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
