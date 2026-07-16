# Models

This directory contains the machine learning model implementations 
for flood hazard mapping. Three models are evaluated:

## Models Overview

| Model | Language | Directory |
|-------|----------|-----------|
| Random Forest (RF) | Python | `random_forest/` |
| eXtreme Gradient Boosting (XGB) | Python | `xgboost/` |
| Feed-Forward Neural Network (NNet) | R | `neural_network/` |

## Workflow Stages

Each model follows the same six-stage pipeline; a shared post-hoc
calibration step is run once across all models.

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | 01_training | Train the model on the Argens basin |
| 2 | 02_prediction | Test model on held-out test set |
| 3 | 03_validation | Validate on the full Argens domain |
| 4 | 04_extrapolation | Apply trained model to unseen Gapeau basin |
| 5 | 05_extrapolation_validation | Validate extrapolated predictions |
| 6 | 06_csi_evaluation | Calculate CSI using target vs predicted rasters |
| — | utils/calibration.py | Brier score + reliability diagrams on the Gapeau samples (all models) |

Feature selection is CSI-optimised (`utils/feature_selection.py`): a greedy
forward search that retains each candidate feature by its incremental
contribution to the CSI of the retrieved flood extent.

## How to Run

### Python Models (RF, XGB)

    cd models/random_forest/     # or models/xgboost/
    python 01_training.py
    python 02_prediction.py
    python 03_validation.py
    python 04_extrapolation.py
    python 05_extrapolation_validation.py
    python 06_csi_evaluation.py

### R Model (NNet)

    cd models/neural_network/
    Rscript 01_training.R
    Rscript 02_prediction.R
    Rscript 03_validation.R
    Rscript 04_extrapolation.R
    Rscript 05_extrapolation_validation.R
    python 06_csi_evaluation.py

### Calibration (all models)

    python -m utils.calibration


## Study Areas

- **Training (Seen):** Argens basin, southeastern France
- **Extrapolation (Unseen):** Gapeau basin, southeastern France
- **Flood Scenario:** 1000-year return period

## Ground Truth
High-resolution 2D hydrodynamic simulation (binary: 0 = no flood, 1 = flood)
