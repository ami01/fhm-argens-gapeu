"""
CSI evaluation for XGBoost predictions.
Compares XGB predicted flood raster against hydrodynamic ground truth.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from config import RAW_DATA_DIR, RESULTS_DIR
from utils.csi_evaluation import evaluate_flood_extent

# ── Paths (only these change per model) ────────────────
BASIN_SHP   = RAW_DATA_DIR / "shapefiles" / "Gapeau_BV_selected.shp"
REF_RASTER  = RAW_DATA_DIR / "reference" / "Gapeau_Q1000_F_nF_rsmpld.tif"
SIM_PATTERN = RESULTS_DIR / "xgboost" / "*_XGB_*_Thrshld.tif"
OUTPUT_DIR  = RESULTS_DIR / "xgboost" / "csi_evaluation"

if __name__ == "__main__":
    results = evaluate_flood_extent(
        basin_shp=BASIN_SHP,
        ref_raster=REF_RASTER,
        sim_pattern=SIM_PATTERN,
        output_dir=OUTPUT_DIR,
        basin_name="Gapeau",
        thr=0.01,
        id_field="BV_ID"
    )

    for sim_name, df in results.items():
        print(f"\nResults: {sim_name}")
        print(df[['BV_ID', 'CSI', 'POD', 'FAR', 'Bias']].to_string(index=False))
