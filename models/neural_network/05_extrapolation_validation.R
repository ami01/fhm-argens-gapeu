# ══════════════════════════════════════════════════════════
# Neural Network — Extrapolation Validation (Gapeau Basin)
#
# Steps:
#   1. Load extrapolated prediction raster
#   2. Load Gapeau validation shapefile
#   3. Extract predictions at validation points
#   4. Compute AUC, Accuracy, Sensitivity, Specificity,
#      Precision, F1, TSS
#   5. Plot ROC curve
#   6. Save all results
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

# Input
PREDICTION_RASTER <- file.path(RESULTS_DIR, "Gapeau_NNET_4dtsts_extrapolated.tif")
VALIDATION_SHP    <- file.path(DATA_DIR, "inventory", "Validation_FnF_Gapeau.shp")

# Output
RESULTS_TXT <- file.path(RESULTS_DIR, "nnet_extrapolation_summary.txt")
RESULTS_CSV <- file.path(RESULTS_DIR, "nnet_extrapolation_predictions.csv")

# ── Parameters ────────────────────────────────────────────
TARGET_COL <- "CID"
THRESHOLD  <- 0.5

# ══════════════════════════════════════════════════════════
# 1. Load Libraries
# ══════════════════════════════════════════════════════════
library(raster)
library(sf)
library(pROC)
library(ggplot2)

cat("✅ Libraries loaded\n")

# ══════════════════════════════════════════════════════════
# 2. Load Prediction Raster
# ══════════════════════════════════════════════════════════
cat("\n1. Loading extrapolated prediction raster...\n")

pred_raster <- raster(PREDICTION_RASTER)
cat("   Loaded:", PREDICTION_RASTER, "\n")

# ══════════════════════════════════════════════════════════
# 3. Load Validation Points
# ══════════════════════════════════════════════════════════
cat("\n2. Loading validation shapefile...\n")

val_sf <- st_read(VALIDATION_SHP, quiet = TRUE)
cat("   Points:", nrow(val_sf), "\n")

if (!TARGET_COL %in% names(val_sf)) {
  stop(paste("Missing '", TARGET_COL, "' column in shapefile"))
}

# ══════════════════════════════════════════════════════════
# 4. Extract Predictions at Validation Points
# ══════════════════════════════════════════════════════════
cat("\n3. Extracting predictions at points...\n")

val_sp <- as(val_sf, "Spatial")
predicted_vals <- raster::extract(pred_raster, val_sp)

y_true      <- val_sf[[TARGET_COL]]
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

y_pred_bin <- ifelse(y_pred_prob >= THRESHOLD, 1, 0)

# Confusion matrix
cm <- table(Truth = y_true, Pred = y_pred_bin)

# Extract values safely
tp <- ifelse("1" %in% rownames(cm) & "1" %in% colnames(cm), cm["1", "1"], 0)
tn <- ifelse("0" %in% rownames(cm) & "0" %in% colnames(cm), cm["0", "0"], 0)
fp <- ifelse("0" %in% rownames(cm) & "1" %in% colnames(cm), cm["0", "1"], 0)
fn <- ifelse("1" %in% rownames(cm) & "0" %in% colnames(cm), cm["1", "0"], 0)

accuracy    <- (tp + tn) / (tp + tn + fp + fn)
sensitivity <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
specificity <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
precision   <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
f1_score    <- ifelse((precision + sensitivity) > 0,
                      2 * precision * sensitivity / (precision + sensitivity), 0)
tss         <- sensitivity + specificity - 1

# ROC
roc_obj   <- roc(y_true, y_pred_prob, quiet = TRUE)
auc_value <- auc(roc_obj)

# Print results
cat("\n   ┌──────────────────────────────┐\n")
cat("   │ ROC AUC:    ", sprintf("%.4f", auc_value),   "         │\n")
cat("   │ Accuracy:   ", sprintf("%.4f", accuracy),     "         │\n")
cat("   │ Sensitivity:", sprintf("%.4f", sensitivity),   "         │\n")
cat("   │ Specificity:", sprintf("%.4f", specificity),   "         │\n")
cat("   │ Precision:  ", sprintf("%.4f", precision),     "         │\n")
cat("   │ F1 Score:   ", sprintf("%.4f", f1_score),      "         │\n")
cat("   │ TSS:        ", sprintf("%.4f", tss),           "         │\n")
cat("   └──────────────────────────────┘\n")

cat("\n   Confusion Matrix:\n")
print(cm)

# ══════════════════════════════════════════════════════════
# 6. Plot ROC Curve
# ══════════════════════════════════════════════════════════
cat("\n5. Plotting ROC curve...\n")

roc_plot_path <- file.path(FIGURES_DIR, "nnet_extrapolation_roc.png")

roc_df <- data.frame(
  FPR = 1 - roc_obj$specificities,
  TPR = roc_obj$sensitivities
)

p <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(color = "red", linewidth = 1) +
  geom_abline(linetype = "dashed", color = "gray") +
  labs(
    title = paste0("ROC Curve — NNet Extrapolation (AUC = ",
                   sprintf("%.4f", auc_value), ")"),
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
cat("   ✅ Saved:", roc_plot_path, "\n")
print(p)

# ══════════════════════════════════════════════════════════
# 7. Save Results
# ══════════════════════════════════════════════════════════
cat("\n6. Saving results...\n")

# Summary text
sink(RESULTS_TXT)
cat("NNet Extrapolation Validation Summary\n")
cat("=====================================\n\n")
cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
cat("Prediction Raster:", PREDICTION_RASTER, "\n")
cat("Validation Points:", VALIDATION_SHP, "\n")
cat("Threshold:", THRESHOLD, "\n\n")
cat("ROC AUC:     ", sprintf("%.4f", auc_value), "\n")
cat("Accuracy:    ", sprintf("%.4f", accuracy), "\n")
cat("Sensitivity: ", sprintf("%.4f", sensitivity), "\n")
cat("Specificity: ", sprintf("%.4f", specificity), "\n")
cat("Precision:   ", sprintf("%.4f", precision), "\n")
cat("F1 Score:    ", sprintf("%.4f", f1_score), "\n")
cat("TSS:         ", sprintf("%.4f", tss), "\n\n")
cat("Confusion Matrix:\n")
print(cm)
sink()
cat("   ✅ Saved:", RESULTS_TXT, "\n")

# Predictions CSV
results_df <- data.frame(
  y_true = y_true,
  y_pred_prob = y_pred_prob,
  y_pred_binary = y_pred_bin
)
write.csv(results_df, RESULTS_CSV, row.names = FALSE)
cat("   ✅ Saved:", RESULTS_CSV, "\n")

cat("\n✅ Done!\n")