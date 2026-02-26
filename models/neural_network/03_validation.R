# ══════════════════════════════════════════════════════════
# Neural Network (nnet) — Validation
#
# Steps:
#   1. Load prediction raster
#   2. Load validation shapefile
#   3. Extract predicted values at validation points
#   4. Compute AUC, Accuracy, Confusion Matrix
#   5. Plot ROC curve
#   6. Save results
#
# Author: Aman Arora
# ══════════════════════════════════════════════════════════

# ── Setup: Find repo root ────────────────────────────────
get_repo_root <- function() {
  script_dir <- dirname(sys.frame(1)$ofile)
  if (is.null(script_dir) || script_dir == "") {
    script_dir <- getwd()
  }
  normalizePath(file.path(script_dir, "..", ".."), mustWork = FALSE)
}

REPO_ROOT <- get_repo_root()
cat("Repo root:", REPO_ROOT, "\n")

# ── Paths ─────────────────────────────────────────────────
DATA_DIR    <- file.path(REPO_ROOT, "data", "raw")
RESULTS_DIR <- file.path(REPO_ROOT, "results", "neural_network")
FIGURES_DIR <- file.path(REPO_ROOT, "figures")

dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

# Input files
PREDICTION_RASTER <- file.path(RESULTS_DIR, "NNET_4dtsts_prediction.tif")
VALIDATION_SHP    <- file.path(DATA_DIR, "inventory", "Validation_FnF.shp")

# Output files
RESULTS_TXT <- file.path(RESULTS_DIR, "nnet_validation_summary.txt")
RESULTS_CSV <- file.path(RESULTS_DIR, "nnet_validation_predictions.csv")

# ── Parameters ────────────────────────────────────────────
TARGET_COL <- "CID"
THRESHOLD  <- 0.5

# ══════════════════════════════════════════════════════════
# 1. Load Libraries
# ══════════════════════════════════════════════════════════
library(raster)
library(sf)
library(pROC)
library(caret)
library(ggplot2)

cat("✅ Libraries loaded\n")

# ══════════════════════════════════════════════════════════
# 2. Load Prediction Raster
# ══════════════════════════════════════════════════════════
cat("\n1. Loading prediction raster...\n")

pred_raster <- raster(PREDICTION_RASTER)
cat("   Loaded:", PREDICTION_RASTER, "\n")
cat("   CRS:", as.character(crs(pred_raster)), "\n")
cat("   Resolution:", res(pred_raster), "\n")

# ══════════════════════════════════════════════════════════
# 3. Load Validation Points
# ══════════════════════════════════════════════════════════
cat("\n2. Loading validation shapefile...\n")

val_sf <- st_read(VALIDATION_SHP, quiet = TRUE)
cat("   Points:", nrow(val_sf), "\n")

# Check target column exists
if (!TARGET_COL %in% names(val_sf)) {
  stop(paste(
    "Validation shapefile missing '", TARGET_COL,
    "' column. Available:", paste(names(val_sf), collapse = ", ")
  ))
}

# ══════════════════════════════════════════════════════════
# 4. Extract Predicted Values at Validation Points
# ══════════════════════════════════════════════════════════
cat("\n3. Extracting predictions at validation points...\n")

# Convert sf to Spatial for raster::extract
val_sp <- as(val_sf, "Spatial")

# Extract raster values
predicted_vals <- raster::extract(pred_raster, val_sp)

# Combine with ground truth
y_true     <- val_sf[[TARGET_COL]]
y_pred_prob <- predicted_vals

# Remove NAs
valid_mask <- !is.na(y_pred_prob)
if (sum(!valid_mask) > 0) {
  cat("   ⚠️ Removing", sum(!valid_mask), "NA predictions\n")
  y_true      <- y_true[valid_mask]
  y_pred_prob <- y_pred_prob[valid_mask]
}

cat("   Valid predictions:", length(y_pred_prob), "\n")

# ══════════════════════════════════════════════════════════
# 5. Compute Metrics
# ══════════════════════════════════════════════════════════
cat("\n4. Computing metrics...\n")

# Binary predictions
y_pred_binary <- ifelse(y_pred_prob >= THRESHOLD, 1, 0)

# ROC and AUC
roc_obj <- roc(y_true, y_pred_prob, quiet = TRUE)
auc_val <- auc(roc_obj)

# Confusion matrix
cm <- confusionMatrix(
  as.factor(y_pred_binary),
  as.factor(y_true)
)

cat("\n   ┌─────────────────────────┐\n")
cat("   │ ROC AUC: ", sprintf("%.4f", auc_val), "       │\n")
cat("   │ Accuracy:", sprintf("%.4f", cm$overall["Accuracy"]), "       │\n")
cat("   └─────────────────────────┘\n")

cat("\n   Confusion Matrix:\n")
print(cm$table)

# ══════════════════════════════════════════════════════════
# 6. Plot ROC Curve
# ══════════════════════════════════════════════════════════
cat("\n5. Plotting ROC curve...\n")

roc_plot_path <- file.path(FIGURES_DIR, "nnet_roc_curve.png")

roc_df <- data.frame(
  FPR = 1 - roc_obj$specificities,
  TPR = roc_obj$sensitivities
)

p <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(color = "red", linewidth = 1) +
  geom_abline(linetype = "dashed", color = "gray") +
  labs(
    title = paste0("ROC Curve — NNet (AUC = ",
                   sprintf("%.4f", auc_val), ")"),
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.title = element_text(size = 12)
  ) +
  coord_equal()

ggsave(roc_plot_path, p, width = 8, height = 6, dpi = 300)
cat("   ✅ ROC curve saved to:", roc_plot_path, "\n")

# Display
print(p)

# ══════════════════════════════════════════════════════════
# 7. Save Results
# ══════════════════════════════════════════════════════════
cat("\n6. Saving results...\n")

# Save summary
sink(RESULTS_TXT)
cat("Neural Network Validation Summary\n")
cat("==================================\n\n")
cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
cat("Prediction Raster:", PREDICTION_RASTER, "\n")
cat("Validation Points:", VALIDATION_SHP, "\n")
cat("Threshold:", THRESHOLD, "\n\n")
cat("ROC AUC:", sprintf("%.4f", auc_val), "\n")
cat("Accuracy:", sprintf("%.4f", cm$overall["Accuracy"]), "\n\n")
cat("Confusion Matrix:\n")
print(cm$table)
cat("\nFull Report:\n")
print(cm)
sink()
cat("   ✅ Summary saved to:", RESULTS_TXT, "\n")

# Save predictions to CSV
results_df <- data.frame(
  y_true = y_true,
  y_pred_prob = y_pred_prob,
  y_pred_binary = y_pred_binary
)
write.csv(results_df, RESULTS_CSV, row.names = FALSE)
cat("   ✅ Predictions saved to:", RESULTS_CSV, "\n")

cat("\n✅ Done!\n")