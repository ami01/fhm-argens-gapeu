Gapeau calibration samples (reliability diagram, Fig_Reliability_Diagrams)
Balanced 10,000 flood / 10,000 non-flood pixels, numpy seed=42.
Reference: Gapeau_Q1000_F_nF_rsmpld.tif; flood_label = (reference depth > 0).
Columns: pixel_index (flattened on ref grid HxW=5087x9658), row, col, x_utm, y_utm (EPSG:32631 cell centre), flood_label, pred_prob.

Model  n      AUROC   Brier   file
Nnet   20000 0.9738  0.0655  Gapeau_Nnet_calibration_sample_seed42.csv
XGB    20000 0.9761  0.0648  Gapeau_XGB_calibration_sample_seed42.csv
RF     20000 0.9722  0.0714  Gapeau_RF_calibration_sample_seed42.csv
