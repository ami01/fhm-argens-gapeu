"""
Central configuration for all paths and parameters.
All paths are relative to the repository root.
"""
from pathlib import Path

# ── Directory Paths ────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"

# ── Sub-directories ────────────────────────────────────
SHAPEFILES_DIR = RAW_DATA_DIR / "shapefiles"
REFERENCE_DIR = RAW_DATA_DIR / "reference"
INVENTORY_DIR = RAW_DATA_DIR / "inventory"
RASTER_DIR = RAW_DATA_DIR / "rasters"

# Create all directories
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR,
          FIGURES_DIR, SHAPEFILES_DIR, REFERENCE_DIR,
          INVENTORY_DIR, RASTER_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Parameters ─────────────────────────────────────────
RANDOM_SEED = 42
FIGURE_DPI = 300
WATER_LEVEL_THRESHOLD = 0.01
CLASSIFICATION_THRESHOLD = 0.5
