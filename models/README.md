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

Each model follows the same six-stage pipeline:

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | 01_training | Train the model on the Argens basin |
| 2 | 02_testing | Test model on held-out test set |
| 3 | 03_validation | Validate on the full Argens domain |
| 4 | 04_extrapolation | Apply trained model to unseen Gapeau basin |
| 5 | 05_extrapolation_validation | Validate extrapolated predictions |
| 6 | 06_csi_evaluation | Calculate CSI using target vs predicted rasters |

## How to Run

### Python Models (RF, XGB)


### R Model (NNet)


## Study Areas

- **Training (Seen):** Argens basin, southeastern France
- **Extrapolation (Unseen):** Gapeau basin, southeastern France
- **Flood Scenario:** 1000-year return period

## Ground Truth
High-resolution 2D hydrodynamic simulation (binary: 0 = no flood, 1 = flood)
