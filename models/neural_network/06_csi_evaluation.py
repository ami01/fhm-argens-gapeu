"""
CSI evaluation for Neural Network predictions.
Compares RF predicted flood raster against hydrodynamic ground truth.
"""
import sys
from pathlib import Path

# Add repo root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from config import RAW_DATA_DIR, RESULTS_DIR
from utils.csi_evaluation import evaluate_flood_extent

# ── Paths (all relative to repo root) ──────────────────
BASIN_SHP   = RAW_DATA_DIR / "shapefiles" / "Gapeau_BV_selected.shp"
REF_RASTER  = RAW_DATA_DIR / "reference" / "Gapeau_Q1000_F_nF_rsmpld.tif"
SIM_PATTERN = RESULTS_DIR / "random_forest" / "*_Nnet_*_Thrshld.tif"
OUTPUT_DIR  = RESULTS_DIR / "random_forest" / "csi_evaluation"

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

    # Print summary
    for sim_name, df in results.items():
        print(f"\n{'='*50}")
        print(f"Results: {sim_name}")
        print(f"{'='*50}")
        print(df[['BV_ID', 'CSI', 'POD', 'FAR', 'Bias']].to_string(index=False))
