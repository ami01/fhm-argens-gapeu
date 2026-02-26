# ══════════════════════════════════════════════════════════
# Neural Network (nnet) — Raster Prediction
#
# Steps:
#   1. Load trained nnet model
#   2. Stack input feature rasters
#   3. Predict flood probability per pixel
#   4. Write prediction raster
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
RASTER_DIR  <- file.path(DATA_DIR, "rasters")

dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

# Input files
MODEL_PATH  <- file.path(RESULTS_DIR, "nnet_model_4dtsts.rds")

# Output files
OUTPUT_TIF  <- file.path(RESULTS_DIR, "NNET_4dtsts_prediction.tif")

# ── Parameters ────────────────────────────────────────────
SELECTED_RASTERS <- c("Discharge_N", "Dist2stream_N", "HAND", "River_Slope")

# ══════════════════════════════════════════════════════════
# 1. Load Libraries
# ══════════════════════════════════════════════════════════
library(raster)
library(caret)

cat("✅ Libraries loaded\n")

# ══════════════════════════════════════════════════════════
# 2. Load Trained Model
# ══════════════════════════════════════════════════════════
cat("\n1. Loading trained model...\n")

nnet_model <- readRDS(MODEL_PATH)
cat("   Loaded from:", MODEL_PATH, "\n")

# ══════════════════════════════════════════════════════════
# 3. Load and Stack Feature Rasters
# ══════════════════════════════════════════════════════════
cat("\n2. Loading feature rasters...\n")

# Find all .tif files in raster directory
all_files <- list.files(
  path = RASTER_DIR,
  pattern = "\\.tif$",
  full.names = TRUE
)

# Filter to selected features
selected_files <- all_files[grepl(
  paste(SELECTED_RASTERS, collapse = "|"),
  all_files
)]

# Verify correct number found
if (length(selected_files) != length(SELECTED_RASTERS)) {
  stop(paste(
    "Expected", length(SELECTED_RASTERS),
    "rasters, found", length(selected_files)
  ))
}

cat("   Found", length(selected_files), "feature rasters:\n")
for (f in selected_files) {
  cat("     ", basename(f), "\n")
}

# Stack rasters
Rasters <- stack(selected_files)

# Fix layer names if needed (remove file extensions, etc.)
# Ensure names match training feature names
cat("\n   Layer names:", paste(names(Rasters), collapse = ", "), "\n")

# Fix Dist2stream name if needed
dist2stream_idx <- which(names(Rasters) == "Dist2stream")
if (length(dist2stream_idx) > 0) {
  names(Rasters)[dist2stream_idx] <- "Dist2stream_N"
  cat("   Renamed 'Dist2stream' → 'Dist2stream_N'\n")
}

cat("   Final layer names:", paste(names(Rasters), collapse = ", "), "\n")

# ══════════════════════════════════════════════════════════
# 4. Predict
# ══════════════════════════════════════════════════════════
cat("\n3. Generating predictions...\n")
cat("   This may take a while for large rasters...\n")

# Predict probability (1 - prob gives flood probability)
prediction_nnet <- (1 - raster::predict(
  model = nnet_model,
  object = Rasters,
  type = "prob"
))

# ══════════════════════════════════════════════════════════
# 5. Write Output Raster
# ══════════════════════════════════════════════════════════
cat("\n4. Writing prediction raster...\n")

writeRaster(
  prediction_nnet,
  filename = OUTPUT_TIF,
  format = "GTiff",
  overwrite = TRUE
)

cat("   ✅ Prediction raster written to:", OUTPUT_TIF, "\n")

# ══════════════════════════════════════════════════════════
# 6. Plot
# ══════════════════════════════════════════════════════════
cat("\n5. Plotting prediction...\n")

# Save plot
plot_path <- file.path(FIGURES_DIR, "nnet_prediction_map.png")
png(plot_path, width = 800, height = 600, res = 150)
plot(prediction_nnet, main = "NNet Flood Probability Prediction")
dev.off()
cat("   ✅ Plot saved to:", plot_path, "\n")

# Display plot
plot(prediction_nnet, main = "NNet Flood Probability Prediction")

cat("\nDone!\n")