# ══════════════════════════════════════════════════════════
# Neural Network — Extrapolation to Unseen Region (Gapeau)
#
# Steps:
#   1. Load trained nnet model (trained on Argens)
#   2. Stack Gapeau feature rasters
#   3. Predict flood probability
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
RASTER_DIR  <- file.path(DATA_DIR, "rasters_gapeau")

dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

# Input
MODEL_PATH <- file.path(RESULTS_DIR, "nnet_model_4dtsts.rds")

# Output
OUTPUT_TIF <- file.path(RESULTS_DIR, "Gapeau_NNET_4dtsts_extrapolated.tif")

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
# 3. Load and Stack Gapeau Feature Rasters
# ══════════════════════════════════════════════════════════
cat("\n2. Loading Gapeau feature rasters...\n")

all_files <- list.files(
  path = RASTER_DIR,
  pattern = "\\.tif$",
  full.names = TRUE
)

selected_files <- all_files[grepl(
  paste(SELECTED_RASTERS, collapse = "|"),
  all_files
)]

if (length(selected_files) != length(SELECTED_RASTERS)) {
  stop(paste(
    "Expected", length(SELECTED_RASTERS),
    "rasters, found", length(selected_files)
  ))
}

cat("   Found", length(selected_files), "rasters:\n")
for (f in selected_files) {
  cat("     ", basename(f), "\n")
}

Rasters <- stack(selected_files)

# Fix layer names if needed
dist2stream_idx <- which(names(Rasters) == "Dist2stream")
if (length(dist2stream_idx) > 0) {
  names(Rasters)[dist2stream_idx] <- "Dist2stream_N"
  cat("   Renamed 'Dist2stream' → 'Dist2stream_N'\n")
}

cat("   Layer names:", paste(names(Rasters), collapse = ", "), "\n")

# ══════════════════════════════════════════════════════════
# 4. Predict
# ══════════════════════════════════════════════════════════
cat("\n3. Generating predictions (this may take a while)...\n")

prediction_nnet <- (1 - raster::predict(
  model = nnet_model,
  object = Rasters,
  type = "prob"
))

# ══════════════════════════════════════════════════════════
# 5. Write Output
# ══════════════════════════════════════════════════════════
cat("\n4. Writing extrapolation raster...\n")

writeRaster(
  prediction_nnet,
  filename = OUTPUT_TIF,
  format = "GTiff",
  overwrite = TRUE
)

cat("   ✅ Written to:", OUTPUT_TIF, "\n")

# ══════════════════════════════════════════════════════════
# 6. Plot
# ══════════════════════════════════════════════════════════
plot_path <- file.path(FIGURES_DIR, "nnet_extrapolation_map.png")
png(plot_path, width = 800, height = 600, res = 150)
plot(prediction_nnet, main = "NNet Extrapolation — Gapeau Basin")
dev.off()
cat("   ✅ Plot saved to:", plot_path, "\n")

plot(prediction_nnet, main = "NNet Extrapolation — Gapeau Basin")

cat("\nDone!\n")